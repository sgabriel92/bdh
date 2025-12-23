# Copyright 2025 Pathway Technology, Inc.
# BDH-GPU Image Training - MNIST experiments for knowledge visualization and erasure

import copy
import os
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import bdh

# -----------------------------------------------------------------------------
# Device and dtype setup (same as train.py)
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
    n_embd=256,
    n_head=4,
    n_neurons=8192,  # Smaller for MNIST (can increase for more capacity)
    dropout=0.05,
    vocab_size=256,  # Pixel values 0-255
    rope_theta=2**16,
)

BATCH_SIZE = 64
MAX_EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
IMG_SIZE = 28  # MNIST is 28x28
SEQ_LEN = IMG_SIZE * IMG_SIZE  # 784 pixels

# Output directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def get_dataloaders():
    """Load MNIST and prepare dataloaders."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_data = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    
    def collate_fn(batch):
        """Convert images to pixel sequences for autoregressive training."""
        images, labels = zip(*batch)
        images = torch.stack(images)  # B, 1, 28, 28
        # Convert to byte sequences (0-255)
        pixels = (images * 255).byte().view(len(images), -1)  # B, 784
        # Autoregressive: input is pixels[:-1], target is pixels[1:]
        inputs = pixels[:, :-1].long()
        targets = pixels[:, 1:].long()
        return inputs, targets, torch.tensor(labels)
    
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    
    return train_loader, test_loader, test_data


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_epoch(model, train_loader, optimizer, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (inputs, targets, labels) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with ctx:
            logits, loss = model(inputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % LOG_FREQ == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def evaluate(model, test_loader):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with ctx:
                logits, loss = model(inputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_image(model, prompt_pixels=100, temperature=0.8, top_k=40):
    """Generate an image from a partial prompt."""
    model.eval()
    
    # Start with zeros or random
    prompt = torch.zeros(1, prompt_pixels, dtype=torch.long, device=device)
    
    # Generate remaining pixels
    full_seq = model.generate(
        prompt, 
        max_new_tokens=SEQ_LEN - prompt_pixels,
        temperature=temperature,
        top_k=top_k
    )
    
    # Reshape to image
    img = full_seq[0, :SEQ_LEN].cpu().numpy().reshape(IMG_SIZE, IMG_SIZE)
    return img


@torch.no_grad()
def complete_image(model, partial_img, temperature=0.8, top_k=40):
    """Complete a partial image (e.g., given top half, generate bottom half)."""
    model.eval()
    
    prompt = torch.tensor(partial_img.flatten(), dtype=torch.long, device=device).unsqueeze(0)
    remaining = SEQ_LEN - prompt.size(1)
    
    full_seq = model.generate(
        prompt,
        max_new_tokens=remaining,
        temperature=temperature,
        top_k=top_k
    )
    
    img = full_seq[0, :SEQ_LEN].cpu().numpy().reshape(IMG_SIZE, IMG_SIZE)
    return img


def visualize_generations(model, num_samples=10, save_path="outputs/generated_digits.png"):
    """Generate and visualize random images."""
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 5))
    
    for i, ax in enumerate(axes.flat):
        img = generate_image(model, prompt_pixels=50, temperature=0.9)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
    
    plt.suptitle("BDH-GPU Generated Digits")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved generations to {save_path}")


# -----------------------------------------------------------------------------
# Sparse Activation Analysis
# -----------------------------------------------------------------------------
@torch.no_grad()
def get_sparse_activations(model, inputs):
    """Extract sparse activations from the first layer."""
    model.eval()
    B, T = inputs.size()
    
    v_ast = model.ln(model.wte(inputs).unsqueeze(1))
    x = F.relu(v_ast @ model.decoder_x)  # B, H, T, N//H
    
    # Flatten heads: B, T, N
    x_flat = x.view(B, T, -1)
    
    # Mean over sequence: B, N
    return x_flat.mean(dim=1)


@torch.no_grad()
def collect_digit_activations(model, test_loader, max_samples=200):
    """Collect activation patterns for each digit class."""
    model.eval()
    digit_activations = {i: [] for i in range(10)}
    samples_per_digit = {i: 0 for i in range(10)}
    
    for inputs, targets, labels in test_loader:
        inputs = inputs.to(device)
        acts = get_sparse_activations(model, inputs)
        
        for i, label in enumerate(labels):
            d = label.item()
            if samples_per_digit[d] < max_samples:
                digit_activations[d].append(acts[i].cpu())
                samples_per_digit[d] += 1
        
        if all(s >= max_samples for s in samples_per_digit.values()):
            break
    
    # Average activations per digit
    for d in range(10):
        if digit_activations[d]:
            digit_activations[d] = torch.stack(digit_activations[d]).mean(0)
        else:
            digit_activations[d] = torch.zeros(model.config.n_neurons)
    
    return digit_activations


def visualize_digit_activations(activations, save_path="outputs/digit_activations.png"):
    """Create a heatmap of neuron activations per digit."""
    matrix = torch.stack([activations[d] for d in range(10)]).numpy()
    
    # Show top 300 most variable neurons
    variance = matrix.var(axis=0)
    top_neurons = variance.argsort()[-300:]
    matrix_subset = matrix[:, top_neurons]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(matrix_subset, aspect="auto", cmap="viridis")
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Digit {d}" for d in range(10)])
    ax.set_xlabel("Top 300 Most Discriminative Neurons")
    ax.set_title("BDH-GPU: Sparse Activations per Digit Class")
    plt.colorbar(im, label="Activation Strength")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved activation heatmap to {save_path}")


def compute_sparsity(model, test_loader):
    """Compute activation sparsity statistics."""
    model.eval()
    sparsities = []
    
    with torch.no_grad():
        for inputs, targets, labels in test_loader:
            inputs = inputs.to(device)
            acts = get_sparse_activations(model, inputs)
            sparsity = (acts == 0).float().mean().item()
            sparsities.append(sparsity)
            if len(sparsities) > 50:
                break
    
    avg_sparsity = np.mean(sparsities)
    print(f"Average activation sparsity: {avg_sparsity:.2%}")
    return avg_sparsity


# -----------------------------------------------------------------------------
# Knowledge Erasure
# -----------------------------------------------------------------------------
def find_digit_neurons(activations, target_digit, threshold=2.0):
    """Find neurons specific to a digit using z-score analysis."""
    target = activations[target_digit]
    others = torch.stack([activations[d] for d in range(10) if d != target_digit])
    others_mean = others.mean(0)
    others_std = others.std(0) + 1e-6
    
    z_scores = (target - others_mean) / others_std
    digit_neurons = (z_scores > threshold).nonzero().flatten()
    
    return digit_neurons, z_scores


def ablate_neurons(model, neuron_indices):
    """Zero out specific neurons in the model."""
    model_copy = copy.deepcopy(model)
    N_per_head = model.config.n_neurons // model.config.n_head
    
    with torch.no_grad():
        for n in neuron_indices:
            n = n.item() if hasattr(n, 'item') else n
            h = n // N_per_head
            local_n = n % N_per_head
            
            # Zero projections to/from this neuron
            model_copy.decoder_x.data[h, :, local_n] = 0
            model_copy.decoder_y.data[h, :, local_n] = 0
            model_copy.encoder.data[n, :] = 0
    
    return model_copy


def test_knowledge_erasure(model, erased_model, test_data, digit_to_erase, 
                           save_path="outputs/erasure_test.png"):
    """Visualize the effect of knowledge erasure."""
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    
    for d in range(10):
        # Find an example of this digit
        for img, label in test_data:
            if label == d:
                real_img = (img.squeeze().numpy() * 255).astype(np.uint8)
                break
        
        # Use top half as prompt
        top_half = real_img[:IMG_SIZE // 2, :].flatten()
        prompt = torch.tensor([top_half], dtype=torch.long, device=device)
        remaining = SEQ_LEN - len(top_half)
        
        # Complete with original model
        with torch.no_grad():
            full_orig = model.generate(prompt, max_new_tokens=remaining, temperature=0.7, top_k=20)
        img_orig = full_orig[0, :SEQ_LEN].cpu().numpy().reshape(IMG_SIZE, IMG_SIZE)
        
        # Complete with erased model
        with torch.no_grad():
            full_erased = erased_model.generate(prompt, max_new_tokens=remaining, temperature=0.7, top_k=20)
        img_erased = full_erased[0, :SEQ_LEN].cpu().numpy().reshape(IMG_SIZE, IMG_SIZE)
        
        # Display
        axes[0, d].imshow(real_img, cmap="gray", vmin=0, vmax=255)
        axes[0, d].set_title(f"{d}", fontsize=10)
        axes[0, d].axis("off")
        
        axes[1, d].imshow(img_orig, cmap="gray", vmin=0, vmax=255)
        axes[1, d].axis("off")
        
        axes[2, d].imshow(img_erased, cmap="gray", vmin=0, vmax=255)
        axes[2, d].axis("off")
        
        if d == digit_to_erase:
            for row in range(3):
                for spine in axes[row, d].spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(3)
                    spine.set_visible(True)
    
    axes[0, 0].set_ylabel("Real", fontsize=12)
    axes[1, 0].set_ylabel("Original", fontsize=12)
    axes[2, 0].set_ylabel("Erased", fontsize=12)
    
    plt.suptitle(f"Knowledge Erasure Test: Digit {digit_to_erase} Erased (red box)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved erasure test to {save_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("BDH-GPU Image Training - MNIST Experiments")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading MNIST dataset...")
    train_loader, test_loader, test_data = get_dataloaders()
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    
    # Create model
    print("\n[2/6] Creating BDH-GPU model...")
    model = bdh.BDH_GPU(config).to(device)
    model = torch.compile(model)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print(f"  Neurons: {config.n_neurons:,}")
    print(f"  Layers: {config.n_layer}")
    
    # Train
    print("\n[3/6] Training...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, epoch)
        test_loss = evaluate(model, test_loader)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "outputs/bdh_gpu_mnist.pt")
    print("  Model saved to outputs/bdh_gpu_mnist.pt")
    
    # Generate samples
    print("\n[4/6] Generating sample images...")
    # Need to use uncompiled model for generation
    model_uncompiled = bdh.BDH_GPU(config).to(device)
    model_uncompiled.load_state_dict(torch.load("outputs/bdh_gpu_mnist.pt", weights_only=True))
    visualize_generations(model_uncompiled)
    
    # Analyze activations
    print("\n[5/6] Analyzing sparse activations...")
    compute_sparsity(model_uncompiled, test_loader)
    activations = collect_digit_activations(model_uncompiled, test_loader)
    visualize_digit_activations(activations)
    
    # Knowledge erasure experiment
    print("\n[6/6] Knowledge erasure experiment...")
    digit_to_erase = 7
    neurons, z_scores = find_digit_neurons(activations, digit_to_erase, threshold=2.0)
    print(f"  Found {len(neurons)} neurons specific to digit {digit_to_erase}")
    
    erased_model = ablate_neurons(model_uncompiled, neurons)
    test_knowledge_erasure(model_uncompiled, erased_model, test_data, digit_to_erase)
    
    print("\n" + "=" * 60)
    print("Done! Check the 'outputs/' directory for results:")
    print("  - generated_digits.png")
    print("  - digit_activations.png")
    print("  - erasure_test.png")
    print("  - bdh_gpu_mnist.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()

