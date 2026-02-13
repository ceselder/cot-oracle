"""
Train linear and MLP probes to predict nudge-following from activations.

This is the initial validation experiment:
- Can we distinguish "followed nudge" from "ignored nudge" in activations?
- If linear probe fails but MLP succeeds, the signal exists but is nonlinear
- If both fail, activations may not encode causal influence distinctly
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm


class NudgeDataset(Dataset):
    """Dataset of trajectory activations with nudge-following labels."""

    def __init__(
        self,
        examples: list[dict],
        activation_type: str = "final",  # "final", "mean_trajectory", "all_trajectory"
        nudge_types: list[str] | None = None,  # Filter to specific nudge types
    ):
        self.examples = examples
        self.activation_type = activation_type

        # Filter by nudge type if specified
        if nudge_types:
            self.examples = [e for e in self.examples if e["nudge_type"] in nudge_types]

        # Filter to only examples where we have a clear signal
        # (exclude controls for binary classification)
        self.examples = [e for e in self.examples if e["nudge_type"] != "none"]

        self._load_activations()

    def _load_activations(self):
        """Load all activations into memory."""
        self.activations = []
        self.labels = []

        for ex in tqdm(self.examples, desc="Loading activations"):
            try:
                data = torch.load(ex["trajectory_path"])

                if self.activation_type == "final":
                    act = data["final_activation"]
                elif self.activation_type == "mean_trajectory":
                    # Mean over all sentence positions
                    act = torch.stack(data["sentence_activations"]).mean(dim=0)
                elif self.activation_type == "first_last_diff":
                    # Difference between last and first sentence
                    acts = data["sentence_activations"]
                    act = acts[-1] - acts[0] if len(acts) > 1 else acts[0]
                else:
                    raise ValueError(f"Unknown activation_type: {self.activation_type}")

                self.activations.append(act)
                self.labels.append(int(ex["followed_nudge"]))

            except Exception as e:
                print(f"Error loading {ex['trajectory_path']}: {e}")

        print(f"Loaded {len(self.activations)} examples")
        print(f"Class balance: {sum(self.labels)}/{len(self.labels)} followed nudge")

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]


class LinearProbe(nn.Module):
    """Simple linear probe."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLPProbe(nn.Module):
    """MLP probe with one hidden layer."""

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_probe(
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
) -> dict:
    """Train a probe and return metrics."""

    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(n_epochs):
        # Training
        probe.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for acts, labels in train_loader:
            acts = acts.to(device).float()
            labels = labels.to(device).float()

            optimizer.zero_grad()
            logits = probe(acts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend((logits > 0).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        # Validation
        probe.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for acts, labels in val_loader:
                acts = acts.to(device).float()
                logits = probe(acts)
                val_preds.extend((logits > 0).cpu().numpy())
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    # Restore best model
    if best_state:
        probe.load_state_dict(best_state)

    return {"best_val_acc": best_val_acc, "history": history}


def run_probe_experiment(
    data_dir: Path = Path("data/collected"),
    activation_types: list[str] = ["final", "mean_trajectory", "first_last_diff"],
    nudge_types: list[str] | None = None,
    device: str = "cuda",
):
    """Run full probe experiment with different configurations."""

    # Load collected examples
    with open(data_dir / "collected_examples.json") as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} examples")

    results = {}

    for act_type in activation_types:
        print(f"\n{'='*50}")
        print(f"Activation type: {act_type}")
        print('='*50)

        # Create dataset
        dataset = NudgeDataset(examples, activation_type=act_type, nudge_types=nudge_types)

        if len(dataset) < 20:
            print(f"Not enough examples for {act_type}, skipping")
            continue

        # Split
        train_idx, val_idx = train_test_split(
            range(len(dataset)), test_size=0.2, random_state=42,
            stratify=[dataset.labels[i] for i in range(len(dataset))]
        )

        train_data = [(dataset.activations[i], dataset.labels[i]) for i in train_idx]
        val_data = [(dataset.activations[i], dataset.labels[i]) for i in val_idx]

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16)

        d_model = dataset.activations[0].shape[0]
        print(f"d_model: {d_model}")

        # Train linear probe
        print("\n--- Linear Probe ---")
        linear_probe = LinearProbe(d_model)
        linear_results = train_probe(linear_probe, train_loader, val_loader, device=device)
        print(f"Best val accuracy: {linear_results['best_val_acc']:.3f}")

        # Train MLP probe
        print("\n--- MLP Probe ---")
        mlp_probe = MLPProbe(d_model)
        mlp_results = train_probe(mlp_probe, train_loader, val_loader, device=device)
        print(f"Best val accuracy: {mlp_results['best_val_acc']:.3f}")

        results[act_type] = {
            "linear_acc": linear_results["best_val_acc"],
            "mlp_acc": mlp_results["best_val_acc"],
            "n_examples": len(dataset),
            "class_balance": sum(dataset.labels) / len(dataset.labels),
        }

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for act_type, res in results.items():
        print(f"\n{act_type}:")
        print(f"  Linear: {res['linear_acc']:.3f}")
        print(f"  MLP:    {res['mlp_acc']:.3f}")
        print(f"  (n={res['n_examples']}, {res['class_balance']:.1%} positive)")

    # Save results
    with open(data_dir / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def analyze_probe_weights(
    probe: LinearProbe,
    data_dir: Path,
    top_k: int = 20,
):
    """Analyze which dimensions the linear probe uses most."""

    weights = probe.linear.weight.data[0].cpu().numpy()

    # Top positive and negative weights
    top_pos = np.argsort(weights)[-top_k:][::-1]
    top_neg = np.argsort(weights)[:top_k]

    print(f"\nTop {top_k} positive weight dimensions (predict 'followed nudge'):")
    for i in top_pos:
        print(f"  dim {i}: {weights[i]:.4f}")

    print(f"\nTop {top_k} negative weight dimensions (predict 'ignored nudge'):")
    for i in top_neg:
        print(f"  dim {i}: {weights[i]:.4f}")

    return {"top_pos": top_pos.tolist(), "top_neg": top_neg.tolist(), "weights": weights}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/collected")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nudge_types", nargs="+", default=None,
                       help="Filter to specific nudge types (authority, hint, sycophancy)")
    args = parser.parse_args()

    results = run_probe_experiment(
        data_dir=Path(args.data_dir),
        nudge_types=args.nudge_types,
        device=args.device,
    )
