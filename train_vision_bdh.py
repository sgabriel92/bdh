# Copyright 2025 Pathway Technology, Inc.
# Vision-BDH Training on CIFAR-10
# Includes visualization of learned spatial features

import os
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vision_bdh import VisionBDH, cifar10_config

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
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"Using device: {device} with dtype {dtype}")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
config = cifar10_config()

BATCH_SIZE = 128
MAX_EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
LOG_FREQ = 100

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def get_dataloaders():
    """Load CIFAR-10 with augmentation."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    train_data = datasets.CIFAR10(
        "./data", train=True, download=True, transform=train_transform
    )
    test_data = datasets.CIFAR10(
        "./data", train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    return train_loader, test_loader, test_data


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_epoch(model, train_loader, optimizer, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with ctx:
            logits, loss = model(images, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % LOG_FREQ == 0:
            acc = 100 * correct / total
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}, Acc: {acc:.1f}%")
    
    return total_loss / len(train_loader), 100 * correct / total


def evaluate(model, test_loader):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with ctx:
                logits, loss = model(images, labels)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(test_loader), 100 * correct / total


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
@torch.no_grad()
def collect_class_activations(model, test_loader, max_per_class=100):
    """Collect sparse activations for each class."""
    model.eval()
    class_acts = {c: [] for c in range(10)}
    counts = {c: 0 for c in range(10)}
    
    for images, labels in test_loader:
        images = images.to(device)
        sparse = model.get_sparse_activations(images)  # B, T, N
        
        # Average over patches (excluding CLS): B, N
        patch_mean = sparse[:, 1:, :].mean(dim=1)
        
        for i, label in enumerate(labels):
            c = label.item()
            if counts[c] < max_per_class:
                class_acts[c].append(patch_mean[i].cpu())
                counts[c] += 1
        
        if all(cnt >= max_per_class for cnt in counts.values()):
            break
    
    # Average per class
    for c in range(10):
        if class_acts[c]:
            class_acts[c] = torch.stack(class_acts[c]).mean(0)
        else:
            class_acts[c] = torch.zeros(config.n_neurons)
    
    return class_acts


def visualize_class_activations(class_acts, save_path):
    """Heatmap of neuron activations per class."""
    matrix = torch.stack([class_acts[c] for c in range(10)]).numpy()
    
    # Top 200 most variable neurons
    variance = matrix.var(axis=0)
    top_neurons = variance.argsort()[-200:]
    matrix_subset = matrix[:, top_neurons]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix_subset, aspect="auto", cmap="viridis")
    ax.set_yticks(range(10))
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("Top 200 Most Discriminative Neurons")
    ax.set_title("Vision-BDH: Sparse Activations per Class")
    plt.colorbar(im, label="Activation Strength")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


@torch.no_grad()
def visualize_patch_activations(model, test_loader, save_path):
    """Show which patches activate which neurons (2D spatial structure!)."""
    model.eval()
    
    # Get a batch
    images, labels = next(iter(test_loader))
    images = images[:8].to(device)
    labels = labels[:8]
    
    # Get sparse activations: B, T, N (T includes CLS token)
    sparse = model.get_sparse_activations(images)
    
    # Remove CLS token, reshape to grid: B, grid_h, grid_w, N
    patch_sparse = sparse[:, 1:, :]  # B, n_patches, N
    grid_size = config.grid_size
    patch_sparse = patch_sparse.reshape(-1, grid_size, grid_size, config.n_neurons)
    
    # Find most active neurons per image
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    
    for i in range(8):
        # Original image (denormalize)
        img = images[i].cpu()
        img = img * torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        img = img + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        img = img.permute(1, 2, 0).numpy().clip(0, 1)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{CIFAR10_CLASSES[labels[i]]}", fontsize=9)
        axes[0, i].axis("off")
        
        # Total sparse activation per patch
        total_act = patch_sparse[i].sum(dim=-1).cpu().numpy()
        axes[1, i].imshow(total_act, cmap="hot")
        axes[1, i].set_title("Total Activation", fontsize=8)
        axes[1, i].axis("off")
        
        # Number of active neurons per patch
        n_active = (patch_sparse[i] > 0).sum(dim=-1).cpu().numpy()
        axes[2, i].imshow(n_active, cmap="viridis")
        axes[2, i].set_title("# Active Neurons", fontsize=8)
        axes[2, i].axis("off")
        
        # Top neuron activation pattern
        top_neuron = patch_sparse[i].sum(dim=(0, 1)).argmax().item()
        top_pattern = patch_sparse[i, :, :, top_neuron].cpu().numpy()
        axes[3, i].imshow(top_pattern, cmap="plasma")
        axes[3, i].set_title(f"Neuron {top_neuron}", fontsize=8)
        axes[3, i].axis("off")
    
    axes[0, 0].set_ylabel("Image", fontsize=10)
    axes[1, 0].set_ylabel("Σ Activations", fontsize=10)
    axes[2, 0].set_ylabel("# Neurons", fontsize=10)
    axes[3, 0].set_ylabel("Top Neuron", fontsize=10)
    
    plt.suptitle("Vision-BDH: Spatial Patch Activations", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


@torch.no_grad()
def cluster_and_visualize_neurons(model, test_loader, n_clusters=8, save_path=None):
    """Cluster neurons by their spatial activation patterns."""
    model.eval()
    
    # Collect activations
    all_sparse = []
    all_labels = []
    
    for images, labels in test_loader:
        images = images.to(device)
        sparse = model.get_sparse_activations(images)  # B, T, N
        patch_sparse = sparse[:, 1:, :]  # Remove CLS
        
        all_sparse.append(patch_sparse.cpu())
        all_labels.extend(labels.tolist())
        
        if len(all_sparse) > 20:
            break
    
    all_sparse = torch.cat(all_sparse, dim=0)  # M, n_patches, N
    M = all_sparse.size(0)
    
    # Average over samples: n_patches, N
    avg_sparse = all_sparse.mean(dim=0).numpy()
    
    # Cluster neurons by their spatial pattern
    # Each neuron has a (grid_size × grid_size) activation pattern
    neuron_patterns = avg_sparse.T  # N, n_patches
    
    # Only cluster active neurons
    active_mask = neuron_patterns.sum(axis=1) > 0.01
    active_patterns = neuron_patterns[active_mask]
    
    if len(active_patterns) < n_clusters:
        print(f"  Only {len(active_patterns)} active neurons, skipping clustering")
        return
    
    print(f"  Clustering {len(active_patterns)} active neurons...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(active_patterns)
    
    # Visualize cluster centroids as spatial maps
    grid_size = config.grid_size
    fig, axes = plt.subplots(2, n_clusters // 2, figsize=(3 * n_clusters // 2, 6))
    axes = axes.flat
    
    for c in range(n_clusters):
        centroid = kmeans.cluster_centers_[c].reshape(grid_size, grid_size)
        centroid = (centroid - centroid.min()) / (centroid.max() - centroid.min() + 1e-8)
        
        n_in_cluster = (cluster_labels == c).sum()
        
        axes[c].imshow(centroid, cmap="hot")
        axes[c].set_title(f"Cluster {c}\n({n_in_cluster} neurons)", fontsize=10)
        axes[c].axis("off")
    
    plt.suptitle("Discovered Neuron Clusters (Spatial Patterns)", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")
    else:
        plt.show()


def visualize_training_curves(train_losses, train_accs, test_losses, test_accs, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train')
    axes[0].plot(epochs, test_losses, 'r-', label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, train_accs, 'b-', label='Train')
    axes[1].plot(epochs, test_accs, 'r-', label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Vision-BDH Training on CIFAR-10', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Vision-BDH Training on CIFAR-10")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading CIFAR-10...")
    train_loader, test_loader, test_data = get_dataloaders()
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    print(f"  Patches: {config.n_patches} ({config.grid_size}×{config.grid_size})")
    
    # Create model
    print("\n[2/5] Creating Vision-BDH model...")
    model = VisionBDH(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Neurons: {config.n_neurons:,}")
    print(f"  Layers: {config.n_layer}")
    
    # Test sparsity
    with torch.no_grad():
        test_img = torch.randn(1, 3, 32, 32, device=device)
        sparse = model.get_sparse_activations(test_img)
        sparsity = (sparse == 0).float().mean().item()
        print(f"  Initial sparsity: {sparsity:.1%}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )
    
    # Training
    print("\n[3/5] Training...")
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0
    
    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch)
        test_loss, test_acc = evaluate(model, test_loader)
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.1f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "outputs/vision_bdh_best.pt")
    
    print(f"\n  Best Test Accuracy: {best_acc:.1f}%")
    
    # Save final model
    torch.save(model.state_dict(), "outputs/vision_bdh_final.pt")
    print("  Saved: outputs/vision_bdh_final.pt")
    
    # Visualizations
    print("\n[4/5] Creating visualizations...")
    
    visualize_training_curves(
        train_losses, train_accs, test_losses, test_accs,
        "outputs/vision_bdh_training.png"
    )
    
    visualize_patch_activations(
        model, test_loader, "outputs/vision_bdh_patches.png"
    )
    
    class_acts = collect_class_activations(model, test_loader)
    visualize_class_activations(class_acts, "outputs/vision_bdh_classes.png")
    
    cluster_and_visualize_neurons(
        model, test_loader, n_clusters=8,
        save_path="outputs/vision_bdh_clusters.png"
    )
    
    # Final sparsity
    print("\n[5/5] Final statistics...")
    with torch.no_grad():
        test_img = torch.randn(1, 3, 32, 32, device=device)
        sparse = model.get_sparse_activations(test_img)
        sparsity = (sparse == 0).float().mean().item()
        active = (sparse.sum(dim=(0, 1)) > 0).sum().item()
        print(f"  Final sparsity: {sparsity:.1%}")
        print(f"  Active neurons: {active}/{config.n_neurons}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\nOutputs:")
    print("  - vision_bdh_training.png  : Loss/accuracy curves")
    print("  - vision_bdh_patches.png   : Spatial patch activations")
    print("  - vision_bdh_classes.png   : Class-specific neurons")
    print("  - vision_bdh_clusters.png  : Discovered spatial patterns")
    print("  - vision_bdh_best.pt       : Best model checkpoint")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    model = main()

