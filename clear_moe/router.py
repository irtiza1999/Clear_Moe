"""
CLEAR-MoE Router — Expert routing architecture and training.

Router types:
- Linear: Simple linear projection hidden_dim -> num_experts
- MLP: Two-layer MLP with a small hidden dimension

Training:
- Supervised from KMeans cluster assignments on calibration data
- Optional entropy regularization for load balance
- Confidence-adaptive routing: top-1 by default, escalate to top-2 for uncertain tokens

The router is the only learned component in CLEAR-MoE (besides optional
per-expert output scales). Everything else is extracted from the pretrained
dense backbone without retraining.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LinearRouter(nn.Module):
    """
    Simple linear router: projects hidden states to expert logits.

    forward(x) -> (expert_indices, expert_weights, logits)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(hidden_dim, num_experts, bias=True)

        # Initialize with small weights
        nn.init.kaiming_uniform_(self.gate.weight, a=0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route input tokens to experts.

        Args:
            x: (batch, seq_len, hidden_dim) or (batch, hidden_dim) input

        Returns:
            expert_indices: (batch, [seq_len,] top_k) selected expert IDs
            expert_weights: (batch, [seq_len,] top_k) softmax routing weights
            logits: (batch, [seq_len,] num_experts) raw gate logits
        """
        logits = self.gate(x)  # (..., num_experts)
        probs = F.softmax(logits, dim=-1)

        # Select top-k experts
        weights, indices = torch.topk(probs, self.top_k, dim=-1)

        # Renormalize weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return indices, weights, logits


class MLPRouter(nn.Module):
    """
    Two-layer MLP router for more expressive routing decisions.

    Architecture: hidden_dim -> router_hidden -> num_experts
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        router_hidden: int = 64,
        top_k: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_experts),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route input tokens to experts."""
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return indices, weights, logits


class AdaptiveRouter(nn.Module):
    """
    Confidence-adaptive router: uses top-1 by default, escalates
    to top-2 when the router is uncertain.

    Uncertainty is measured by the gap between the top-2 probabilities.
    If gap < confidence_threshold, use top-2 routing for that token.
    This extends D2DMoE's dynamic-k idea to extracted vision experts.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        confidence_threshold: float = 0.6,
        router_type: str = "linear",
        router_hidden: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.confidence_threshold = confidence_threshold

        if router_type == "linear":
            self.gate = nn.Linear(hidden_dim, num_experts)
        else:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, num_experts),
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adaptive routing: top-1 for confident tokens, top-2 for uncertain.

        Returns:
            expert_indices: (..., 2) — always returns top-2 indices
            expert_weights: (..., 2) — weight of second expert is 0 for confident tokens
            logits: (..., num_experts) raw logits
        """
        logits = self.gate(x) if isinstance(self.gate, nn.Linear) else self.gate(x)
        probs = F.softmax(logits, dim=-1)

        # Get top-2 experts
        top2_weights, top2_indices = torch.topk(probs, 2, dim=-1)

        # Measure confidence: gap between top-1 and top-2 probabilities
        confidence = top2_weights[..., 0] - top2_weights[..., 1]

        # Mask: use top-2 only when uncertain (confidence < threshold)
        uncertain = (confidence < self.confidence_threshold).unsqueeze(-1)

        # Zero out the second expert weight for confident tokens
        mask = torch.ones_like(top2_weights)
        mask[..., 1] = uncertain.squeeze(-1).float()
        top2_weights = top2_weights * mask

        # Renormalize
        top2_weights = top2_weights / (top2_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return top2_indices, top2_weights, logits


def create_router(
    config: Dict,
    hidden_dim: int,
    num_experts: int,
) -> nn.Module:
    """
    Factory function to create a router from config.

    Args:
        config: Config dict with router section
        hidden_dim: Input dimension
        num_experts: Number of experts

    Returns:
        Router module
    """
    router_cfg = config.get("router", {})
    router_type = router_cfg.get("type", "linear")
    adaptive = router_cfg.get("adaptive_routing", False)
    confidence = router_cfg.get("confidence_threshold", 0.6)
    router_hidden = router_cfg.get("hidden_dim", 64)

    if adaptive:
        return AdaptiveRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            confidence_threshold=confidence,
            router_type=router_type,
            router_hidden=router_hidden,
        )
    elif router_type == "linear":
        return LinearRouter(hidden_dim, num_experts)
    elif router_type == "mlp":
        return MLPRouter(hidden_dim, num_experts, router_hidden=router_hidden)
    elif router_type == "adaptive":
        return AdaptiveRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            confidence_threshold=confidence,
            router_type="linear",
            router_hidden=router_hidden,
        )
    else:
        raise ValueError(f"Unknown router type: {router_type}")


def fit_router(
    router: nn.Module,
    activations: torch.Tensor,
    cluster_labels: np.ndarray,
    config: Dict,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    """
    Train a router to predict expert assignments from hidden states.

    Uses the cluster labels from KMeans as supervision targets.
    This is a lightweight training step — only the router parameters
    are learned, everything else stays frozen.

    Args:
        router: Router module to train
        activations: (N, hidden_dim) calibration activations
        cluster_labels: (N,) target expert assignments from KMeans
        config: Config dict with training hyperparameters
        device: Torch device

    Returns:
        Trained router and training history dict
    """
    router_cfg = config.get("router", {})
    epochs = router_cfg.get("epochs", 5)
    lr = router_cfg.get("lr", 1e-3)
    weight_decay = router_cfg.get("weight_decay", 0.01)
    balance_coeff = router_cfg.get("balance_coeff", 0.01)
    batch_size = min(256, activations.shape[0])

    # Prepare dataset
    labels_tensor = torch.tensor(cluster_labels, dtype=torch.long)
    dataset = TensorDataset(activations, labels_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader)
    )

    router.to(device)
    router.train()

    history = {"loss": [], "accuracy": [], "entropy": []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_entropy = 0.0

        for acts, labels in loader:
            acts = acts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward
            indices, weights, logits = router(acts)

            # Classification loss
            ce_loss = F.cross_entropy(logits, labels)

            # Balance regularization (entropy of mean routing probabilities)
            probs = F.softmax(logits, dim=-1)
            mean_probs = probs.mean(dim=0)
            entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
            # We want to maximize entropy (encourage balanced load)
            balance_loss = -balance_coeff * entropy

            loss = ce_loss + balance_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track metrics
            epoch_loss += ce_loss.item() * acts.shape[0]
            predicted = logits.argmax(dim=-1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += acts.shape[0]
            epoch_entropy += entropy.item()

        avg_loss = epoch_loss / epoch_total
        accuracy = epoch_correct / epoch_total
        avg_entropy = epoch_entropy / len(loader)

        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        history["entropy"].append(avg_entropy)

        logger.info(
            f"  Epoch {epoch+1}/{epochs}: "
            f"loss={avg_loss:.4f}, acc={accuracy:.4f}, entropy={avg_entropy:.4f}"
        )

    router.eval()
    return router, history


def fit_all_routers(
    model: nn.Module,
    expertized_layers: Dict,
    activations: Dict[str, torch.Tensor],
    config: Dict,
    device: torch.device,
) -> Dict[str, nn.Module]:
    """
    Fit routers for all expertized layers.

    Args:
        model: Pretrained model
        expertized_layers: Dict of ExpertizedLayer results
        activations: Dict of calibration activations per layer
        config: Config dict
        device: Torch device

    Returns:
        Dict mapping layer name to trained router module
    """
    routers = {}

    for name, exp_layer in expertized_layers.items():
        logger.info(f"Fitting router for {name}")

        router = create_router(
            config,
            hidden_dim=exp_layer.hidden_dim,
            num_experts=exp_layer.num_experts,
        )

        acts = activations.get(name)
        if acts is None:
            logger.warning(f"No activations for {name}, skipping router fitting")
            continue

        router, history = fit_router(
            router=router,
            activations=acts,
            cluster_labels=exp_layer.cluster_labels,
            config=config,
            device=device,
        )

        routers[name] = router
        logger.info(
            f"  Final accuracy: {history['accuracy'][-1]:.4f}, "
            f"entropy: {history['entropy'][-1]:.4f}"
        )

    return routers


def compute_routing_stats(
    router: nn.Module,
    activations: torch.Tensor,
    device: torch.device,
) -> Dict:
    """
    Compute routing statistics for analysis.

    Returns:
        Dict with expert load distribution, entropy, confidence stats
    """
    router.eval()
    router.to(device)

    with torch.no_grad():
        # Process in chunks for memory
        chunk_size = 1024
        all_indices = []
        all_weights = []
        all_entropies = []

        for i in range(0, activations.shape[0], chunk_size):
            chunk = activations[i:i+chunk_size].to(device)
            indices, weights, logits = router(chunk)

            all_indices.append(indices.cpu())
            all_weights.append(weights.cpu())

            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            all_entropies.append(entropy.cpu())

        indices = torch.cat(all_indices, dim=0)
        weights = torch.cat(all_weights, dim=0)
        entropies = torch.cat(all_entropies, dim=0)

    # Expert load distribution (for top-1 expert)
    if indices.dim() > 1:
        top1_indices = indices[:, 0] if indices.shape[-1] > 1 else indices.squeeze(-1)
    else:
        top1_indices = indices

    num_experts = router.num_experts if hasattr(router, "num_experts") else int(top1_indices.max()) + 1
    load_counts = torch.zeros(num_experts)
    for e in range(num_experts):
        load_counts[e] = (top1_indices == e).sum().item()

    load_fractions = load_counts / load_counts.sum()

    stats = {
        "expert_load_counts": load_counts.tolist(),
        "expert_load_fractions": load_fractions.tolist(),
        "load_balance_std": load_fractions.std().item(),
        "mean_entropy": entropies.mean().item(),
        "std_entropy": entropies.std().item(),
        "mean_top1_weight": weights[:, 0].mean().item() if weights.dim() > 1 else weights.mean().item(),
    }

    return stats
