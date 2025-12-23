# Copyright 2025 Pathway Technology, Inc.
# BDH Shakespeare Character Erasure Experiment
# Train on Shakespeare, then erase specific characters by ablating their neurons

import os
import re
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset

import bdh

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"Using device: {device} with dtype {dtype}")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
config = bdh.BDH_GPUConfig(
    n_layer=6,
    n_embd=384,
    n_head=4,  # Changed: 16384 / 4 = 4096 (divisible!)
    n_neurons=16384,
    dropout=0.1,
    vocab_size=256,  # Byte-level
    rope_theta=2**16,
)

BATCH_SIZE = 64
BLOCK_SIZE = 256  # Context length
MAX_EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
LOG_FREQ = 100

os.makedirs("outputs", exist_ok=True)


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
class ShakespeareDataset(Dataset):
    """Character-level Shakespeare dataset."""

    def __init__(self, text, block_size):
        self.text = text
        self.block_size = block_size
        # Byte-level encoding (vocab_size=256)
        self.data = torch.tensor([ord(c) for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_shakespeare(path="input.txt"):
    """Load and split Shakespeare text."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 90/10 train/val split
    n = len(text)
    train_text = text[: int(0.9 * n)]
    val_text = text[int(0.9 * n) :]

    return train_text, val_text, text


def extract_characters(text):
    """Extract character names from Shakespeare format."""
    # Pattern: Name at start of line followed by colon
    pattern = r"^([A-Z][a-zA-Z\s]+):"
    characters = set()
    for line in text.split("\n"):
        match = re.match(pattern, line)
        if match:
            characters.add(match.group(1).strip())
    return sorted(characters)


def find_character_segments(text, character_name):
    """Find all text segments where a character is speaking."""
    segments = []
    lines = text.split("\n")
    in_segment = False
    current_segment = []

    for line in lines:
        if line.startswith(f"{character_name}:"):
            in_segment = True
            current_segment = [line]
        elif in_segment:
            if re.match(r"^[A-Z][a-zA-Z\s]+:", line):
                # New character speaking
                segments.append("\n".join(current_segment))
                in_segment = False
                current_segment = []
            else:
                current_segment.append(line)

    if current_segment:
        segments.append("\n".join(current_segment))

    return segments


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_epoch(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with ctx:
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % LOG_FREQ == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with ctx:
                _, loss = model(x, y)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_text(model, prompt, max_tokens=200, temperature=0.8, top_k=40):
    """Generate text from a prompt."""
    model.eval()

    # Encode prompt
    tokens = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device)

    # Generate
    output = model.generate(tokens, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    # Decode
    generated = "".join([chr(min(t, 255)) for t in output[0].tolist()])
    return generated


# -----------------------------------------------------------------------------
# Neuron Analysis
# -----------------------------------------------------------------------------
@torch.no_grad()
def get_activations_for_text(model, text, max_len=BLOCK_SIZE):
    """Get sparse activations for a piece of text."""
    model.eval()

    # Encode
    tokens = torch.tensor([[ord(c) for c in text[:max_len]]], dtype=torch.long, device=device)

    # Get activations
    v_ast = model.ln(model.wte(tokens).unsqueeze(1))
    x = F.relu(v_ast @ model.decoder_x)  # 1, H, T, N//H

    # Flatten: T, N
    T = tokens.size(1)
    N = model.config.n_neurons
    x_flat = x.squeeze(0).transpose(0, 1).reshape(T, N)

    return x_flat.cpu().numpy()


def collect_character_activations(model, text, characters, max_segments=20):
    """Collect average activations for each character's dialogue."""
    char_activations = {}

    for char in characters:
        segments = find_character_segments(text, char)
        if not segments:
            continue

        all_acts = []
        for seg in segments[:max_segments]:
            if len(seg) > 10:  # Skip very short segments
                acts = get_activations_for_text(model, seg)
                # Mean over positions
                all_acts.append(acts.mean(axis=0))

        if all_acts:
            char_activations[char] = np.stack(all_acts).mean(axis=0)

    return char_activations


def find_character_neurons(char_activations, target_char, threshold=2.0):
    """Find neurons specific to a character using z-score."""
    if target_char not in char_activations:
        return np.array([]), np.array([])

    target = char_activations[target_char]
    others = [v for k, v in char_activations.items() if k != target_char]

    if not others:
        return np.array([]), np.array([])

    others = np.stack(others)
    others_mean = others.mean(axis=0)
    others_std = others.std(axis=0) + 1e-6

    z_scores = (target - others_mean) / others_std
    specific_neurons = np.where(z_scores > threshold)[0]

    return specific_neurons, z_scores


# -----------------------------------------------------------------------------
# Ablation
# -----------------------------------------------------------------------------
def ablate_neurons(model, neuron_indices):
    """Create a copy of the model with specific neurons ablated."""
    import copy

    model_copy = copy.deepcopy(model)
    N_per_head = model.config.n_neurons // model.config.n_head

    with torch.no_grad():
        for n in neuron_indices:
            n = int(n)
            h = n // N_per_head
            local_n = n % N_per_head

            model_copy.decoder_x.data[h, :, local_n] = 0
            model_copy.decoder_y.data[h, :, local_n] = 0
            model_copy.encoder.data[n, :] = 0

    return model_copy


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def visualize_character_activations(char_activations, save_path):
    """Heatmap of neuron activations per character."""
    characters = list(char_activations.keys())
    matrix = np.stack([char_activations[c] for c in characters])

    # Top 300 most variable neurons
    variance = matrix.var(axis=0)
    top_neurons = variance.argsort()[-300:]
    matrix_subset = matrix[:, top_neurons]

    fig, ax = plt.subplots(figsize=(16, max(6, len(characters) * 0.4)))
    im = ax.imshow(matrix_subset, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(characters)))
    ax.set_yticklabels(characters, fontsize=8)
    ax.set_xlabel("Top 300 Most Discriminative Neurons")
    ax.set_title("BDH: Sparse Activations per Character")
    plt.colorbar(im, label="Activation Strength")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_erasure_comparison(original_text, erased_text, char_name, save_path):
    """Side-by-side comparison of generation before/after erasure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    axes[0].text(
        0.05, 0.95, original_text, transform=axes[0].transAxes, fontsize=9,
        verticalalignment="top", fontfamily="monospace", wrap=True
    )
    axes[0].set_title(f"Original Model", fontsize=12)
    axes[0].axis("off")

    axes[1].text(
        0.05, 0.95, erased_text, transform=axes[1].transAxes, fontsize=9,
        verticalalignment="top", fontfamily="monospace", wrap=True
    )
    axes[1].set_title(f"After Erasing '{char_name}'", fontsize=12)
    axes[1].axis("off")

    plt.suptitle(f"Character Erasure Experiment: {char_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BDH Shakespeare Character Erasure Experiment")
    print("=" * 70)

    # Load data
    print("\n[1/7] Loading Shakespeare text...")
    train_text, val_text, full_text = load_shakespeare("input.txt")
    print(f"  Total characters: {len(full_text):,}")
    print(f"  Train: {len(train_text):,}, Val: {len(val_text):,}")

    # Extract characters
    characters = extract_characters(full_text)
    print(f"  Found {len(characters)} speaking characters")
    print(f"  Examples: {characters[:5]}")

    # Create datasets
    train_dataset = ShakespeareDataset(train_text, BLOCK_SIZE)
    val_dataset = ShakespeareDataset(val_text, BLOCK_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Create model
    print("\n[2/7] Creating BDH model...")
    model = bdh.BDH_GPU(config).to(device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Neurons: {config.n_neurons:,}")
    print(f"  Context: {BLOCK_SIZE} characters")

    # Training
    print("\n[3/7] Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, epoch)
        val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Save model
    torch.save(model._orig_mod.state_dict(), "outputs/shakespeare_bdh.pt")
    print("  Saved: outputs/shakespeare_bdh.pt")

    # Load uncompiled model for analysis
    print("\n[4/7] Loading model for analysis...")
    model_uncompiled = bdh.BDH_GPU(config).to(device)
    model_uncompiled.load_state_dict(torch.load("outputs/shakespeare_bdh.pt", weights_only=True))

    # Generate sample text
    print("\n[5/7] Testing generation...")
    prompt = "First Citizen:\n"
    generated = generate_text(model_uncompiled, prompt, max_tokens=300)
    print("  Sample generation:")
    print("-" * 40)
    print(generated[:500])
    print("-" * 40)

    # Analyze character-specific neurons
    print("\n[6/7] Analyzing character-specific neurons...")
    # Focus on most frequent characters
    top_characters = []
    for char in characters:
        segments = find_character_segments(full_text, char)
        if len(segments) >= 5:
            top_characters.append((char, len(segments)))
    top_characters.sort(key=lambda x: -x[1])
    top_characters = [c[0] for c in top_characters[:15]]

    print(f"  Analyzing top {len(top_characters)} characters: {top_characters[:5]}...")

    char_activations = collect_character_activations(model_uncompiled, full_text, top_characters)
    print(f"  Collected activations for {len(char_activations)} characters")

    visualize_character_activations(char_activations, "outputs/shakespeare_activations.png")

    # Character erasure experiment
    print("\n[7/7] Character erasure experiment...")
    
    # Pick a character to erase
    target_char = top_characters[0] if top_characters else "First Citizen"
    print(f"  Target character: {target_char}")

    neurons, z_scores = find_character_neurons(char_activations, target_char, threshold=1.5)
    print(f"  Found {len(neurons)} neurons specific to {target_char}")

    if len(neurons) > 0:
        # Generate with original model
        prompt = f"{target_char}:\n"
        original_gen = generate_text(model_uncompiled, prompt, max_tokens=300, temperature=0.7)

        # Ablate and generate
        erased_model = ablate_neurons(model_uncompiled, neurons)
        erased_gen = generate_text(erased_model, prompt, max_tokens=300, temperature=0.7)

        # Also try with a different prompt
        other_char = [c for c in top_characters if c != target_char][0]
        prompt2 = f"{other_char}:\n"
        original_gen2 = generate_text(model_uncompiled, prompt2, max_tokens=200, temperature=0.7)
        erased_gen2 = generate_text(erased_model, prompt2, max_tokens=200, temperature=0.7)

        print(f"\n  Generation comparison for '{target_char}':")
        print("  " + "=" * 60)
        print("  ORIGINAL MODEL:")
        print("  " + "-" * 60)
        print("  " + original_gen[:400].replace("\n", "\n  "))
        print("  " + "=" * 60)
        print(f"  AFTER ERASING {target_char} ({len(neurons)} neurons):")
        print("  " + "-" * 60)
        print("  " + erased_gen[:400].replace("\n", "\n  "))
        print("  " + "=" * 60)

        print(f"\n  Control test with '{other_char}' (should be similar):")
        print("  " + "-" * 60)
        print("  Original: " + original_gen2[:200].replace("\n", " ")[:150])
        print("  Erased:   " + erased_gen2[:200].replace("\n", " ")[:150])

        # Save comparison
        visualize_erasure_comparison(
            original_gen[:600], erased_gen[:600], target_char,
            "outputs/shakespeare_erasure.png"
        )
    else:
        print("  No specific neurons found for this character")

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    print("\nOutputs:")
    print("  - shakespeare_bdh.pt           : Trained model")
    print("  - shakespeare_activations.png  : Character-neuron heatmap")
    print("  - shakespeare_erasure.png      : Before/after erasure comparison")
    print("=" * 70)

    return model_uncompiled, char_activations


if __name__ == "__main__":
    model, char_acts = main()

