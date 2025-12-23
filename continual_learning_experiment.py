# Copyright 2025 Pathway Technology, Inc.
# BDH Continual Learning Experiment
# Demonstrates sequential task learning with minimal catastrophic forgetting

import copy
import os
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

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
    n_embd=256,
    n_head=4,
    n_neurons=16384,  # More neurons for better task separation
    dropout=0.05,
    vocab_size=256,
    rope_theta=2**16,
)

BATCH_SIZE = 64
EPOCHS_PER_TASK = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
IMG_SIZE = 28
SEQ_LEN = IMG_SIZE * IMG_SIZE

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------------------------------------------------------
# Data Loading with Digit Filtering
# -----------------------------------------------------------------------------
def get_digit_subset(dataset, digits):
    """Filter dataset to only include specific digits."""
    indices = [i for i, (_, label) in enumerate(dataset) if label in digits]
    return Subset(dataset, indices)


def collate_fn(batch):
    """Convert images to pixel sequences for autoregressive training."""
    images, labels = zip(*batch)
    images = torch.stack(images)
    pixels = (images * 255).byte().view(len(images), -1)
    inputs = pixels[:, :-1].long()
    targets = pixels[:, 1:].long()
    return inputs, targets, torch.tensor(labels)


def get_task_loaders(task_digits):
    """Get train/test loaders for specific digits."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_full = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_full = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    train_subset = get_digit_subset(train_full, task_digits)
    test_subset = get_digit_subset(test_full, task_digits)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# Training & Evaluation
# -----------------------------------------------------------------------------
def train_epoch(model, loader, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for inputs, targets, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with ctx:
            _, loss = model(inputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, loader):
    """Evaluate on a data loader."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with ctx:
                _, loss = model(inputs, targets)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


# -----------------------------------------------------------------------------
# Neuron Analysis
# -----------------------------------------------------------------------------
@torch.no_grad()
def get_sparse_activations(model, inputs):
    """Extract sparse activations from the first layer."""
    model.eval()
    v_ast = model.ln(model.wte(inputs).unsqueeze(1))
    x = F.relu(v_ast @ model.decoder_x)  # B, H, T, N//H
    # Flatten to B, T, N
    x_flat = x.transpose(1, 2).reshape(inputs.size(0), inputs.size(1), -1)
    # Mean over batch and sequence: N
    return x_flat.mean(dim=(0, 1))


@torch.no_grad()
def get_task_neurons(model, loader, threshold=0.1):
    """Find which neurons are active for a given task."""
    model.eval()
    all_activations = []

    for inputs, _, _ in loader:
        inputs = inputs.to(device)
        acts = get_sparse_activations(model, inputs)
        all_activations.append(acts.cpu())
        if len(all_activations) > 20:
            break

    avg_activation = torch.stack(all_activations).mean(0)
    active_neurons = (avg_activation > threshold).nonzero().flatten()
    return active_neurons, avg_activation


def compute_neuron_overlap(neurons_a, neurons_b):
    """Compute Jaccard overlap between two neuron sets."""
    set_a = set(neurons_a.tolist())
    set_b = set(neurons_b.tolist())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0


@torch.no_grad()
def collect_per_digit_activations(model, loader, max_per_digit=100):
    """Collect average activations for each digit class."""
    model.eval()
    digit_acts = {d: [] for d in range(10)}
    counts = {d: 0 for d in range(10)}

    for inputs, _, labels in loader:
        inputs = inputs.to(device)
        v_ast = model.ln(model.wte(inputs).unsqueeze(1))
        x = F.relu(v_ast @ model.decoder_x)
        x_flat = x.transpose(1, 2).reshape(inputs.size(0), inputs.size(1), -1)
        x_mean = x_flat.mean(dim=1)  # B, N

        for i, label in enumerate(labels):
            d = label.item()
            if counts[d] < max_per_digit:
                digit_acts[d].append(x_mean[i].cpu())
                counts[d] += 1

        if all(c >= max_per_digit for c in counts.values()):
            break

    # Average per digit
    for d in range(10):
        if digit_acts[d]:
            digit_acts[d] = torch.stack(digit_acts[d]).mean(0)
        else:
            digit_acts[d] = torch.zeros(config.n_neurons)

    return digit_acts


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_results(results, neurons_a, neurons_b, act_a, act_b, save_path):
    """Visualize the experiment results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Loss curves
    ax = axes[0]
    n_epochs_a = len(results["phase1_test_a"])
    n_epochs_b = len(results["phase2_test_a"])
    epochs_a = range(1, n_epochs_a + 1)
    epochs_b = range(n_epochs_a + 1, n_epochs_a + n_epochs_b + 1)

    ax.plot(epochs_a, results["phase1_test_a"], "b-o", label="Task A perf", linewidth=2)
    ax.plot(epochs_b, results["phase2_test_a"], "b--o", linewidth=2)
    ax.plot(epochs_b, results["phase2_test_b"], "r-s", label="Task B perf", linewidth=2)
    ax.axvline(x=n_epochs_a + 0.5, color="gray", linestyle=":", label="Task switch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Loss")
    ax.set_title("Continual Learning: Loss Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Neuron activation comparison (subset for visibility)
    ax = axes[1]
    # Show first 500 neurons for clarity
    n_show = min(500, len(act_a))
    x_range = range(n_show)
    ax.bar(x_range, act_a[:n_show].numpy(), alpha=0.6, label="Task A", width=1, color="blue")
    ax.bar(x_range, act_b[:n_show].numpy(), alpha=0.6, label="Task B", width=1, color="red")
    ax.set_xlabel("Neuron Index (first 500)")
    ax.set_ylabel("Mean Activation")
    ax.set_title("Neuron Activation Patterns")
    ax.legend()

    # Plot 3: Neuron overlap
    ax = axes[2]
    set_a = set(neurons_a.tolist())
    set_b = set(neurons_b.tolist())
    only_a = len(set_a - set_b)
    only_b = len(set_b - set_a)
    both = len(set_a & set_b)
    total = only_a + only_b + both

    bars = ax.bar(
        ["Task A\nOnly", "Shared", "Task B\nOnly"],
        [only_a, both, only_b],
        color=["#3498db", "#9b59b6", "#e74c3c"],
        alpha=0.8,
    )
    ax.set_ylabel("Number of Neurons")
    overlap_pct = both / total * 100 if total > 0 else 0
    ax.set_title(f"Neuron Specialization\n(Overlap: {overlap_pct:.1f}%)")

    # Add value labels on bars
    for bar, val in zip(bars, [only_a, both, only_b]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(val),
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved results to {save_path}")


def plot_digit_neuron_heatmap(digit_acts_before, digit_acts_after, save_path):
    """Compare digit-neuron activations before and after Task B training."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    # Stack activations: 10 x N
    matrix_before = torch.stack([digit_acts_before[d] for d in range(10)]).numpy()
    matrix_after = torch.stack([digit_acts_after[d] for d in range(10)]).numpy()

    # Show top 200 most variable neurons
    variance = matrix_after.var(axis=0)
    top_neurons = variance.argsort()[-200:]

    for ax, matrix, title in [
        (axes[0], matrix_before[:, top_neurons], "After Task A Training (digits 0-4 only)"),
        (axes[1], matrix_after[:, top_neurons], "After Task B Training (all digits)"),
    ]:
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"{d}" for d in range(10)])
        ax.set_ylabel("Digit")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Activation")

    axes[1].set_xlabel("Top 200 Most Variable Neurons")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved digit heatmap to {save_path}")


# -----------------------------------------------------------------------------
# Main Experiment
# -----------------------------------------------------------------------------
def run_experiment():
    print("=" * 70)
    print("BDH Continual Learning Experiment")
    print("Sequential Task Learning: Digits 0-4, then 5-9")
    print("=" * 70)

    # Define tasks
    TASK_A_DIGITS = [0, 1, 2, 3, 4]
    TASK_B_DIGITS = [5, 6, 7, 8, 9]
    ALL_DIGITS = list(range(10))

    # Load data
    print("\n[1/6] Loading data...")
    train_a, test_a = get_task_loaders(TASK_A_DIGITS)
    train_b, test_b = get_task_loaders(TASK_B_DIGITS)
    _, test_all = get_task_loaders(ALL_DIGITS)
    print(f"  Task A (digits 0-4): {len(train_a.dataset)} train samples")
    print(f"  Task B (digits 5-9): {len(train_b.dataset)} train samples")

    # Create model
    print("\n[2/6] Creating BDH-GPU model...")
    model = bdh.BDH_GPU(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print(f"  Neurons: {config.n_neurons:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    results = {
        "phase1_train": [],
        "phase1_test_a": [],
        "phase2_train": [],
        "phase2_test_a": [],
        "phase2_test_b": [],
    }

    # Phase 1: Train on Task A (digits 0-4)
    print("\n[3/6] Phase 1: Training on Task A (digits 0-4)...")
    for epoch in range(EPOCHS_PER_TASK):
        train_loss = train_epoch(model, train_a, optimizer)
        test_a_loss = evaluate(model, test_a)
        print(f"  Epoch {epoch+1}/{EPOCHS_PER_TASK}: Train={train_loss:.4f}, Test_A={test_a_loss:.4f}")
        results["phase1_train"].append(train_loss)
        results["phase1_test_a"].append(test_a_loss)

    # Analyze Task A neurons
    neurons_a, activations_a = get_task_neurons(model, test_a)
    print(f"\n  Task A activates {len(neurons_a)} neurons (threshold > 0.1)")

    # Collect per-digit activations after Task A
    digit_acts_after_a = collect_per_digit_activations(model, test_all)

    # Save model state after Task A for comparison
    model_after_a = copy.deepcopy(model)
    baseline_loss_a = results["phase1_test_a"][-1]

    # Phase 2: Train on Task B (digits 5-9) WITHOUT seeing Task A
    print("\n[4/6] Phase 2: Training on Task B (digits 5-9)...")
    print("  (Not seeing Task A data anymore)")
    for epoch in range(EPOCHS_PER_TASK):
        train_loss = train_epoch(model, train_b, optimizer)
        test_a_loss = evaluate(model, test_a)  # Monitor forgetting
        test_b_loss = evaluate(model, test_b)
        print(
            f"  Epoch {epoch+1}/{EPOCHS_PER_TASK}: Train={train_loss:.4f}, "
            f"Test_A={test_a_loss:.4f}, Test_B={test_b_loss:.4f}"
        )
        results["phase2_train"].append(train_loss)
        results["phase2_test_a"].append(test_a_loss)
        results["phase2_test_b"].append(test_b_loss)

    # Analyze Task B neurons
    neurons_b, activations_b = get_task_neurons(model, test_b)
    print(f"\n  Task B activates {len(neurons_b)} neurons (threshold > 0.1)")

    # Compute overlap
    overlap = compute_neuron_overlap(neurons_a, neurons_b)
    print(f"  Neuron overlap (Jaccard): {overlap:.2%}")

    # Collect per-digit activations after Task B
    digit_acts_after_b = collect_per_digit_activations(model, test_all)

    # Phase 3: Final evaluation
    print("\n[5/6] Final Evaluation...")
    final_loss_a = evaluate(model, test_a)
    final_loss_b = evaluate(model, test_b)
    final_loss_all = evaluate(model, test_all)

    forgetting = final_loss_a - baseline_loss_a
    print(f"\n  Results Summary:")
    print(f"  ─────────────────────────────────────────")
    print(f"  Task A loss (after A): {baseline_loss_a:.4f}")
    print(f"  Task A loss (after B): {final_loss_a:.4f}")
    print(f"  Task B loss (after B): {final_loss_b:.4f}")
    print(f"  All digits loss:       {final_loss_all:.4f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  FORGETTING (Δ Task A): {forgetting:+.4f}")

    if forgetting < 0.2:
        print("  ✓ Minimal forgetting! Continual learning successful.")
    elif forgetting < 0.5:
        print("  ~ Moderate forgetting. Some knowledge retained.")
    else:
        print("  ✗ Significant forgetting detected.")

    # Visualize results
    print("\n[6/6] Generating visualizations...")
    plot_results(
        results,
        neurons_a,
        neurons_b,
        activations_a,
        activations_b,
        "outputs/continual_learning_results.png",
    )
    plot_digit_neuron_heatmap(
        digit_acts_after_a,
        digit_acts_after_b,
        "outputs/continual_learning_heatmap.png",
    )

    # Save models
    torch.save(model.state_dict(), "outputs/bdh_continual_final.pt")
    torch.save(model_after_a.state_dict(), "outputs/bdh_after_task_a.pt")
    print("  Saved models to outputs/")

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("Check outputs/ for:")
    print("  - continual_learning_results.png (loss curves & neuron analysis)")
    print("  - continual_learning_heatmap.png (digit-neuron activations)")
    print("  - bdh_continual_final.pt (final model)")
    print("  - bdh_after_task_a.pt (model after Task A only)")
    print("=" * 70)

    return model, results


if __name__ == "__main__":
    model, results = run_experiment()

