"""Dummy training script and test harness for nn-htop."""

import argparse
import sys
import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class ThreeLayerMLP(nn.Module):
    """Simple 3-layer MLP for testing nn-htop."""

    def __init__(self, in_features: int = 16, hidden_1: int = 32, hidden_2: int = 16, out_features: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_2, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def run_training_loop(
    model: nn.Module,
    monitor,
    stop_event: Optional[threading.Event] = None,
    max_steps: Optional[int] = None,
    step_delay: float = 0.05,
    batch_size: int = 32,
    in_features: int = 16,
    num_classes: int = 4,
):
    """Runs a continuous or bounded training loop on synthetic data."""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    step = 0
    while True:
        if stop_event and stop_event.is_set():
            break
        if max_steps is not None and step >= max_steps:
            break

        # Generate synthetic batch
        inputs = torch.randn(batch_size, in_features)
        targets = torch.randint(0, num_classes, (batch_size,))

        # Training step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Update monitor step info
        if hasattr(monitor, "record_step"):
            monitor.record_step(loss=loss.item())

        step += 1
        if step_delay > 0:
            time.sleep(step_delay)


def main():
    parser = argparse.ArgumentParser(description="nn-htop dummy training harness")
    parser.add_argument("--no-tui", action="store_true", help="Run training without TUI for CLI verification")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to run (default: infinite)")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay per step in seconds (default: 0.05)")
    args = parser.parse_args()

    from nn_htop.monitor import Monitor

    model = ThreeLayerMLP()
    monitor = Monitor(model)

    print(f"Created ThreeLayerMLP model with {len(monitor.get_layer_names())} monitored layers:")
    for name, s in monitor.get_stats().items():
        print(f"  - {name} ({s.layer_type})")

    if args.no_tui:
        print(f"\nRunning training for {args.steps or 20} steps without TUI...")
        run_training_loop(model, monitor, max_steps=args.steps or 20, step_delay=args.delay)
        print("\nFinal Monitored Stats:")
        for name, s in monitor.get_stats().items():
            print(f"  {name:10} | Type: {s.layer_type:10} | Mean Act: {s.mean_activation:10.5f} | Grad Norm: {s.grad_norm:10.5f}")
        monitor.remove_hooks()
        print("Training finished successfully.")
    else:
        from nn_htop.ui import launch_htop
        print("Launching nn-htop TUI...")
        launch_htop(monitor=monitor, step_delay=args.delay)


if __name__ == "__main__":
    main()
