# Copyright 2025 Pathway Technology, Inc.
# Single Image Neuron Analysis
# Train BDH to memorize ONE image, then analyze neuron-pixel relationships
# Uses automatic clustering to discover what the network learned (no predefined regions!)

import os
import sys
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

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
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"Using device: {device} with dtype {dtype}")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
IMG_SIZE = 64  # Resize image to 64x64
SEQ_LEN = IMG_SIZE * IMG_SIZE
N_CLUSTERS = 8  # Number of neuron clusters to discover

config = bdh.BDH_GPUConfig(
    n_layer=4,
    n_embd=128,
    n_head=4,
    n_neurons=4096,  # Enough neurons for spatial decomposition
    dropout=0.0,  # No dropout - we WANT to memorize
    vocab_size=256,
    rope_theta=2**16,
)

os.makedirs("outputs", exist_ok=True)


# -----------------------------------------------------------------------------
# Load and Prepare Single Image
# -----------------------------------------------------------------------------
def load_single_image(path, size=IMG_SIZE):
    """Load and preprocess a single image."""
    img = Image.open(path).convert("L")  # Grayscale
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.array(img, dtype=np.uint8)
    return pixels


def create_single_image_dataset(pixels, num_copies=500):
    """Create dataset from single image (repeated for batching)."""
    flat = pixels.flatten()
    inputs = torch.tensor(flat[:-1], dtype=torch.long).unsqueeze(0).repeat(num_copies, 1)
    targets = torch.tensor(flat[1:], dtype=torch.long).unsqueeze(0).repeat(num_copies, 1)
    return TensorDataset(inputs, targets)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_single_image(model, dataset, epochs=300, lr=1e-3):
    """Train until near-perfect memorization."""
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with ctx:
                _, loss = model(inputs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 25 == 0:
            print(f"  Epoch {epoch}: Loss = {avg_loss:.6f}")

        if avg_loss < 0.01:  # Near-perfect memorization
            print(f"  ✓ Memorization achieved at epoch {epoch}!")
            break

    return model


# -----------------------------------------------------------------------------
# Neuron-Pixel Analysis
# -----------------------------------------------------------------------------
@torch.no_grad()
def get_position_activations(model, pixels):
    """Get neuron activations at each pixel position."""
    model.eval()
    flat = torch.tensor(pixels.flatten()[:-1], dtype=torch.long, device=device).unsqueeze(0)

    # Get sparse activations from first layer pass
    v_ast = model.ln(model.wte(flat).unsqueeze(1))
    x = F.relu(v_ast @ model.decoder_x)  # 1, H, T, N//H

    # Reshape to T, N (position × neuron)
    T = flat.size(1)
    N = model.config.n_neurons
    x_flat = x.squeeze(0).transpose(0, 1).reshape(T, N)
    return x_flat.cpu().numpy()


@torch.no_grad()
def get_layer_activations(model, pixels):
    """Extract activations at each layer of the network."""
    model.eval()
    flat = torch.tensor(pixels.flatten()[:-1], dtype=torch.long, device=device).unsqueeze(0)

    B, T = flat.size()
    N = model.config.n_neurons
    L = model.config.n_layer

    layer_activations = []
    v_ast = model.ln(model.wte(flat).unsqueeze(1))  # B, 1, T, D

    for layer_idx in range(L):
        # Sparse projection
        x = F.relu(v_ast @ model.decoder_x)  # B, H, T, N//H

        # Linear attention
        a_ast = model.attn(Q=x, K=x, V=v_ast)

        # Gated output
        y = F.relu(model.ln(a_ast) @ model.decoder_y) * x
        y = y.transpose(1, 2).reshape(B, 1, T, N)

        # Store this layer's sparse activations
        layer_activations.append(
            {
                "sparse_x": x.squeeze(0).transpose(0, 1).reshape(T, N).cpu().numpy(),
                "sparse_y": y.squeeze(0).squeeze(0).cpu().numpy(),  # T, N
                "dense_v": v_ast.squeeze(0).squeeze(0).cpu().numpy(),  # T, D
            }
        )

        # Update for next layer
        v_ast = v_ast + model.ln(y @ model.encoder)

    return layer_activations


# -----------------------------------------------------------------------------
# Automatic Neuron Clustering (The Right Way!)
# -----------------------------------------------------------------------------
def cluster_neurons_by_activation(activations, n_clusters=N_CLUSTERS, min_activity=0.01):
    """
    Cluster neurons based on WHERE they activate in the image.
    This lets the NETWORK tell us what features it learned.
    
    Returns:
        cluster_labels: array of shape (N,) with cluster ID for each neuron
        cluster_info: dict with cluster statistics
    """
    T, N = activations.shape
    
    # Filter to only active neurons (non-zero somewhere)
    neuron_activity = activations.sum(axis=0)
    active_mask = neuron_activity > (neuron_activity.max() * min_activity)
    active_indices = np.where(active_mask)[0]
    
    print(f"  Clustering {len(active_indices)} active neurons (of {N} total)")
    
    # Get activation patterns for active neurons: N_active × T
    active_patterns = activations[:, active_mask].T
    
    # Normalize patterns (we care about WHERE, not how much)
    pattern_norms = np.linalg.norm(active_patterns, axis=1, keepdims=True) + 1e-8
    active_patterns_normalized = active_patterns / pattern_norms
    
    # Cluster neurons by their spatial activation patterns
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    active_labels = kmeans.fit_predict(active_patterns_normalized)
    
    # Map back to full neuron indices
    cluster_labels = np.full(N, -1)  # -1 for inactive neurons
    cluster_labels[active_mask] = active_labels
    
    # Compute cluster info
    cluster_info = {}
    for c in range(n_clusters):
        cluster_neurons = active_indices[active_labels == c]
        cluster_info[c] = {
            "neurons": cluster_neurons,
            "n_neurons": len(cluster_neurons),
            "mean_pattern": activations[:, cluster_neurons].mean(axis=1),
            "total_activity": activations[:, cluster_neurons].sum(),
        }
    
    return cluster_labels, cluster_info


def describe_cluster(cluster_info, img_size):
    """Generate a human-readable description of what each cluster focuses on."""
    descriptions = {}
    h, w = img_size, img_size - 1
    
    for c, info in cluster_info.items():
        pattern = info["mean_pattern"][:h * w].reshape(h, w)
        
        # Find centroid of activation
        total = pattern.sum() + 1e-8
        y_coords, x_coords = np.mgrid[:h, :w]
        center_y = (y_coords * pattern).sum() / total
        center_x = (x_coords * pattern).sum() / total
        
        # Determine rough location
        if center_y < h / 3:
            v_pos = "top"
        elif center_y < 2 * h / 3:
            v_pos = "middle"
        else:
            v_pos = "bottom"
            
        if center_x < w / 3:
            h_pos = "left"
        elif center_x < 2 * w / 3:
            h_pos = "center"
        else:
            h_pos = "right"
        
        # Measure spread
        variance = ((y_coords - center_y)**2 * pattern + (x_coords - center_x)**2 * pattern).sum() / total
        spread = "localized" if variance < (h * w / 8) else "distributed"
        
        descriptions[c] = f"{v_pos}-{h_pos} ({spread}, {info['n_neurons']} neurons)"
    
    return descriptions


# -----------------------------------------------------------------------------
# Ablation & Generation
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_with_ablation(model, prompt_pixels, ablate_neurons=None, total_len=SEQ_LEN):
    """Generate image with specific neurons ablated."""
    model.eval()

    if ablate_neurons is not None and len(ablate_neurons) > 0:
        # Temporarily zero out neurons
        original_encoder = model.encoder.data.clone()
        original_decoder_x = model.decoder_x.data.clone()
        original_decoder_y = model.decoder_y.data.clone()

        N_per_head = model.config.n_neurons // model.config.n_head
        for n in ablate_neurons:
            n = int(n)
            h = n // N_per_head
            local_n = n % N_per_head
            model.decoder_x.data[h, :, local_n] = 0
            model.decoder_y.data[h, :, local_n] = 0
            model.encoder.data[n, :] = 0

    prompt = torch.tensor(prompt_pixels, dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(
        prompt, max_new_tokens=total_len - len(prompt_pixels), temperature=0.1, top_k=5
    )

    if ablate_neurons is not None and len(ablate_neurons) > 0:
        # Restore weights
        model.encoder.data = original_encoder
        model.decoder_x.data = original_decoder_x
        model.decoder_y.data = original_decoder_y

    return generated[0, :total_len].cpu().numpy()


def neuron_weight_sweep(model, pixels, neuron_idx, values=np.linspace(0, 2, 5)):
    """Sweep a neuron's weight and see how the image changes."""
    results = []
    neuron_idx = int(neuron_idx)

    original_encoder = model.encoder.data[neuron_idx].clone()

    for scale in values:
        with torch.no_grad():
            model.encoder.data[neuron_idx] = original_encoder * scale

        prompt = pixels.flatten()[:100]  # First 100 pixels as prompt
        generated = generate_with_ablation(model, prompt, ablate_neurons=None)
        results.append(generated.reshape(IMG_SIZE, IMG_SIZE))

    # Restore
    with torch.no_grad():
        model.encoder.data[neuron_idx] = original_encoder

    return results, values


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------
def visualize_original_image(pixels, save_path):
    """Save the original image for reference."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(pixels, cmap="gray")
    ax.set_title(f"Original Image ({IMG_SIZE}×{IMG_SIZE})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_neuron_clusters(activations, cluster_labels, cluster_info, pixels, img_size, save_path):
    """
    Visualize what each automatically-discovered cluster focuses on.
    This is the key visualization - shows what the NETWORK learned.
    """
    n_clusters = len(cluster_info)
    cols = 4
    rows = (n_clusters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flat if n_clusters > 1 else [axes]
    
    h, w = img_size, img_size - 1
    
    # Get cluster descriptions
    descriptions = describe_cluster(cluster_info, img_size)
    
    for c in range(n_clusters):
        ax = axes[c]
        info = cluster_info[c]
        
        # Create spatial activation map for this cluster
        cluster_pattern = info["mean_pattern"][:h * w].reshape(h, w)
        cluster_pattern = np.pad(cluster_pattern, ((0, 0), (0, 1)), mode="edge")
        
        # Normalize
        cluster_pattern = (cluster_pattern - cluster_pattern.min()) / (cluster_pattern.max() - cluster_pattern.min() + 1e-8)
        
        # Overlay on original image
        img_norm = pixels / 255.0
        
        # Use different colors for different clusters
        colors = plt.cm.tab10(c / n_clusters)[:3]
        overlay = np.stack([
            np.clip(img_norm * 0.4 + cluster_pattern * colors[0], 0, 1),
            np.clip(img_norm * 0.4 + cluster_pattern * colors[1], 0, 1),
            np.clip(img_norm * 0.4 + cluster_pattern * colors[2], 0, 1),
        ], axis=-1)
        
        ax.imshow(overlay)
        ax.set_title(f"Cluster {c}\n{descriptions[c]}", fontsize=9)
        ax.axis("off")
    
    # Hide unused axes
    for ax in axes[n_clusters:]:
        ax.axis("off")
    
    plt.suptitle("Discovered Neuron Clusters (Network-Learned Features)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_cluster_ablation(original, ablated_images, cluster_info, save_path):
    """Show what happens when each cluster is ablated."""
    n = len(ablated_images) + 1
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = axes.flat
    
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original\nReconstruction", fontsize=10)
    axes[0].axis("off")
    
    for i, (img, c) in enumerate(zip(ablated_images, cluster_info.keys())):
        ax = axes[i + 1]
        
        # Show difference from original
        diff = np.abs(original.astype(float) - img.astype(float))
        
        # Overlay: original in gray, difference in red
        img_norm = img / 255.0
        diff_norm = diff / (diff.max() + 1e-8)
        
        overlay = np.stack([
            np.clip(img_norm + diff_norm * 0.5, 0, 1),
            img_norm * 0.8,
            img_norm * 0.8,
        ], axis=-1)
        
        ax.imshow(overlay)
        n_neurons = cluster_info[c]["n_neurons"]
        ax.set_title(f"Ablate Cluster {c}\n({n_neurons} neurons)", fontsize=10)
        ax.axis("off")
    
    # Hide unused
    for ax in axes[len(ablated_images) + 1:]:
        ax.axis("off")
    
    plt.suptitle("Cluster Ablation (Red = Changed Regions)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_neuron_spatial_maps(activations, img_size, top_n=20, save_path=None):
    """Show which pixels each top neuron responds to."""
    neuron_total_act = activations.sum(axis=0)
    top_neurons = np.argsort(neuron_total_act)[-top_n:]

    rows = 4
    cols = top_n // rows
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))

    for idx, ax in enumerate(axes.flat):
        if idx < len(top_neurons):
            n = top_neurons[-(idx + 1)]  # Most active first
            spatial_map = activations[:, n]
            h, w = img_size, img_size - 1
            spatial_map = spatial_map[: h * w].reshape(h, w)
            spatial_map = np.pad(spatial_map, ((0, 0), (0, 1)), mode="edge")
            ax.imshow(spatial_map, cmap="hot")
            ax.set_title(f"Neuron {n}", fontsize=9)
        ax.axis("off")

    plt.suptitle("Top Neurons: Spatial Activation Maps", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_layer_activations(layer_acts, img_size, save_path):
    """Visualize activations at each layer as images."""
    n_layers = len(layer_acts)
    fig, axes = plt.subplots(n_layers, 4, figsize=(14, 3.5 * n_layers))

    for layer_idx, acts in enumerate(layer_acts):
        h, w = img_size, img_size - 1

        # 1. Dense representation - visualize magnitude
        dense = acts["dense_v"]
        dense_magnitude = np.linalg.norm(dense, axis=1)
        dense_img = dense_magnitude[: h * w].reshape(h, w)
        dense_img = np.pad(dense_img, ((0, 0), (0, 1)), mode="edge")

        axes[layer_idx, 0].imshow(dense_img, cmap="viridis")
        axes[layer_idx, 0].set_title(f"Layer {layer_idx}: Dense |v*|")
        axes[layer_idx, 0].axis("off")

        # 2. Sparse X activations - sum across neurons
        sparse_x = acts["sparse_x"]
        sparse_sum = sparse_x.sum(axis=1)
        sparse_img = sparse_sum[: h * w].reshape(h, w)
        sparse_img = np.pad(sparse_img, ((0, 0), (0, 1)), mode="edge")

        axes[layer_idx, 1].imshow(sparse_img, cmap="hot")
        axes[layer_idx, 1].set_title(f"Layer {layer_idx}: Sparse Sum")
        axes[layer_idx, 1].axis("off")

        # 3. Sparsity map - how many neurons active at each position
        sparsity = (sparse_x > 0).sum(axis=1)
        sparsity_img = sparsity[: h * w].reshape(h, w)
        sparsity_img = np.pad(sparsity_img, ((0, 0), (0, 1)), mode="edge")

        axes[layer_idx, 2].imshow(sparsity_img, cmap="plasma")
        axes[layer_idx, 2].set_title(f"Layer {layer_idx}: # Active")
        axes[layer_idx, 2].axis("off")

        # 4. Top-k neuron activation pattern
        topk = 10
        top_neurons = sparse_x.sum(axis=0).argsort()[-topk:]
        top_acts = sparse_x[:, top_neurons].mean(axis=1)
        top_img = top_acts[: h * w].reshape(h, w)
        top_img = np.pad(top_img, ((0, 0), (0, 1)), mode="edge")

        axes[layer_idx, 3].imshow(top_img, cmap="magma")
        axes[layer_idx, 3].set_title(f"Layer {layer_idx}: Top-{topk}")
        axes[layer_idx, 3].axis("off")

    plt.suptitle("Layer-by-Layer Internal Activations", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_learned_filters(model, save_path):
    """Visualize the learned decoder weights as 'filters'."""
    decoder_x = model.decoder_x.data.cpu().numpy()
    H, D, N_per_H = decoder_x.shape

    decoder_flat = decoder_x.transpose(0, 2, 1).reshape(H * N_per_H, D)

    variance = decoder_flat.var(axis=1)
    top_neurons = variance.argsort()[-64:]

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))

    for idx, ax in enumerate(axes.flat):
        n = top_neurons[-(idx + 1)]
        weights = decoder_flat[n]

        side = int(np.ceil(np.sqrt(D)))
        padded = np.zeros(side * side)
        padded[:D] = weights
        filter_img = padded.reshape(side, side)

        vmax = np.abs(filter_img).max()
        ax.imshow(filter_img, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_title(f"N{n}", fontsize=7)
        ax.axis("off")

    plt.suptitle("Learned Decoder Filters (Top 64 by Variance)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_neuron_receptive_fields(activations, pixels, img_size, save_path):
    """For each top neuron, show what image regions it responds to."""
    T, N = activations.shape

    neuron_max = activations.max(axis=0)
    neuron_std = activations.std(axis=0) + 1e-6
    localization = neuron_max / neuron_std

    top_localized = localization.argsort()[-25:]

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    h, w = img_size, img_size - 1

    for idx, ax in enumerate(axes.flat):
        n = top_localized[-(idx + 1)]

        act_map = activations[:, n]
        act_map = act_map[: h * w].reshape(h, w)
        act_map = np.pad(act_map, ((0, 0), (0, 1)), mode="edge")

        act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)

        img_norm = pixels / 255.0
        overlay = np.stack(
            [
                np.clip(img_norm * 0.6 + act_map * 0.8, 0, 1),
                img_norm * 0.6,
                img_norm * 0.6,
            ],
            axis=-1,
        )

        ax.imshow(overlay)
        ax.set_title(f"N{n}", fontsize=9)
        ax.axis("off")

    plt.suptitle("Neuron Receptive Fields (Red = High Activation)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_neuron_sweep(sweep_results, values, neuron_idx, save_path):
    """Show how varying one neuron's weight changes the image."""
    n = len(values)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))

    for i, (img, val) in enumerate(zip(sweep_results, values)):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"×{val:.1f}")
        axes[i].axis("off")

    plt.suptitle(f"Neuron {neuron_idx} Weight Sweep", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_cluster_pca(activations, cluster_labels, save_path):
    """Visualize neuron clusters in 2D using PCA."""
    T, N = activations.shape
    
    # Only use active neurons
    active_mask = cluster_labels >= 0
    active_patterns = activations[:, active_mask].T  # N_active × T
    active_labels = cluster_labels[active_mask]
    
    # PCA to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(active_patterns)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], 
        c=active_labels, cmap="tab10", 
        alpha=0.6, s=20
    )
    
    # Add cluster centers
    for c in range(N_CLUSTERS):
        mask = active_labels == c
        if mask.sum() > 0:
            center = coords[mask].mean(axis=0)
            ax.annotate(
                f"C{c}", center, fontsize=12, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Neuron Clusters in PCA Space\n(Each point = one neuron)")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(image_path):
    print("=" * 70)
    print("BDH Single Image Neuron Analysis")
    print("Using Automatic Clustering (Network-Discovered Features)")
    print("=" * 70)

    # Load image
    print("\n[1/8] Loading image...")
    pixels = load_single_image(image_path)
    print(f"  Image size: {pixels.shape}")
    visualize_original_image(pixels, "outputs/single_original.png")
    print("  Saved: outputs/single_original.png")

    # Create dataset
    dataset = create_single_image_dataset(pixels)

    # Create and train model
    print("\n[2/8] Training to memorize image...")
    model = bdh.BDH_GPU(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    print(f"  Neurons: {config.n_neurons}")
    model = train_single_image(model, dataset, epochs=300)

    # Test reconstruction
    print("\n[3/8] Testing reconstruction...")
    prompt = pixels.flatten()[:200]
    reconstructed = generate_with_ablation(model, prompt)
    recon_img = reconstructed.reshape(IMG_SIZE, IMG_SIZE)
    mse = np.mean((pixels.astype(float) - recon_img.astype(float)) ** 2)
    print(f"  Reconstruction MSE: {mse:.2f}")

    # Get activations
    print("\n[4/8] Analyzing neuron-pixel relationships...")
    activations = get_position_activations(model, pixels)
    print(f"  Activation shape: {activations.shape} (positions × neurons)")

    # Sparsity stats
    sparsity = (activations == 0).mean()
    active_neurons = (activations.sum(axis=0) > 0).sum()
    print(f"  Activation sparsity: {sparsity:.1%}")
    print(f"  Active neurons: {active_neurons}/{config.n_neurons}")

    # AUTOMATIC CLUSTERING - Let the network tell us what it learned!
    print("\n[5/8] Discovering neuron clusters (network-learned features)...")
    cluster_labels, cluster_info = cluster_neurons_by_activation(activations, n_clusters=N_CLUSTERS)
    
    descriptions = describe_cluster(cluster_info, IMG_SIZE)
    print("\n  Discovered clusters:")
    for c, desc in descriptions.items():
        print(f"    Cluster {c}: {desc}")

    # Visualize clusters
    print("\n[6/8] Creating visualizations...")
    visualize_neuron_clusters(
        activations, cluster_labels, cluster_info, pixels, IMG_SIZE,
        "outputs/single_clusters.png"
    )
    print("  Saved: outputs/single_clusters.png (MAIN RESULT)")

    visualize_cluster_pca(activations, cluster_labels, "outputs/single_cluster_pca.png")
    print("  Saved: outputs/single_cluster_pca.png")

    visualize_neuron_spatial_maps(
        activations, IMG_SIZE, top_n=20, save_path="outputs/single_neuron_maps.png"
    )
    print("  Saved: outputs/single_neuron_maps.png")

    layer_acts = get_layer_activations(model, pixels)
    visualize_layer_activations(layer_acts, IMG_SIZE, "outputs/single_layers.png")
    print("  Saved: outputs/single_layers.png")

    visualize_learned_filters(model, "outputs/single_filters.png")
    print("  Saved: outputs/single_filters.png")

    visualize_neuron_receptive_fields(
        activations, pixels, IMG_SIZE, "outputs/single_receptive_fields.png"
    )
    print("  Saved: outputs/single_receptive_fields.png")

    # CLUSTER ABLATION - Delete each cluster and see what breaks
    print("\n[7/8] Running cluster ablation experiments...")
    original = generate_with_ablation(model, prompt).reshape(IMG_SIZE, IMG_SIZE)

    ablated_images = []
    for c in range(N_CLUSTERS):
        neurons_to_ablate = cluster_info[c]["neurons"]
        print(f"  Ablating Cluster {c} ({len(neurons_to_ablate)} neurons)...")
        ablated = generate_with_ablation(
            model, prompt, ablate_neurons=neurons_to_ablate
        ).reshape(IMG_SIZE, IMG_SIZE)
        ablated_images.append(ablated)

    visualize_cluster_ablation(
        original, ablated_images, cluster_info, "outputs/single_ablation.png"
    )
    print("  Saved: outputs/single_ablation.png")

    # Weight sweep on most active neuron
    print("\n[8/8] Running neuron weight sweep...")
    top_neuron = int(np.argmax(activations.sum(axis=0)))
    print(f"  Sweeping most active neuron: {top_neuron}")
    sweep_results, values = neuron_weight_sweep(
        model, pixels, top_neuron, values=np.linspace(0, 3, 7)
    )
    visualize_neuron_sweep(
        sweep_results, values, top_neuron, "outputs/single_sweep.png"
    )
    print("  Saved: outputs/single_sweep.png")

    # Save model
    torch.save(model.state_dict(), "outputs/single_image_model.pt")
    print("\n  Saved model: outputs/single_image_model.pt")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nGenerated outputs:")
    print("  - single_original.png       : Input image")
    print("  - single_clusters.png       : ★ DISCOVERED FEATURES (main result)")
    print("  - single_cluster_pca.png    : Neuron clusters in 2D")
    print("  - single_neuron_maps.png    : Top neuron activation patterns")
    print("  - single_layers.png         : Layer-by-layer activations")
    print("  - single_filters.png        : Learned decoder filters")
    print("  - single_receptive_fields.png : Individual neuron focus areas")
    print("  - single_ablation.png       : What breaks when clusters deleted")
    print("  - single_sweep.png          : Neuron weight morphing")
    print("=" * 70)

    return model, activations, cluster_labels, cluster_info


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python single_image_analysis.py <image_path>")
        print("Example: python single_image_analysis.py skull.png")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    main(image_path)
