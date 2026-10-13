"""Tests for nn_htop Monitor core logic (PyTorch Hooks)."""

import threading
import time
import pytest
import torch
import torch.nn as nn
from nn_htop.monitor import Monitor, LayerStats

pytestmark = pytest.mark.filterwarnings("ignore:Full backward hook is firing.*")


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_monitor_registration():
    """Verify monitor discovers leaf layers and attaches hooks."""
    model = SimpleMLP()
    monitor = Monitor(model)

    layer_names = monitor.get_layer_names()
    assert "fc1" in layer_names
    assert "relu" in layer_names
    assert "fc2" in layer_names

    stats = monitor.get_stats()
    assert len(stats) == 3
    assert stats["fc1"].layer_type == "Linear"
    assert stats["relu"].layer_type == "ReLU"
    assert stats["fc2"].layer_type == "Linear"
    # Initial stats should be 0.0 before any forward/backward pass
    assert stats["fc1"].mean_activation == 0.0
    assert stats["fc1"].grad_norm == 0.0

    monitor.remove_hooks()


def test_forward_hook_records_mean_activation():
    """Verify forward hook calculates and stores mean activation accurately."""
    model = SimpleMLP()
    monitor = Monitor(model)

    # Use fixed input for predictable calculation
    x = torch.ones(2, 4)
    # Run forward pass
    output = model(x)

    stats = monitor.get_stats()
    assert stats["fc1"].mean_activation != 0.0
    assert isinstance(stats["fc1"].mean_activation, float)
    assert stats["relu"].mean_activation >= 0.0  # ReLU outputs are non-negative
    assert isinstance(stats["fc2"].mean_activation, float)

    # Verify that the value matches output tensor mean
    with torch.no_grad():
        fc1_out = model.fc1(x)
        expected_fc1_mean = fc1_out.mean().item()
        assert pytest.approx(stats["fc1"].mean_activation, rel=1e-5) == expected_fc1_mean

    monitor.remove_hooks()


def test_backward_hook_records_gradient_norm():
    """Verify backward hook calculates and stores gradient norm accurately."""
    model = SimpleMLP()
    monitor = Monitor(model)

    x = torch.randn(2, 4)
    output = model(x)
    loss = output.sum()
    loss.backward()

    stats = monitor.get_stats()
    assert stats["fc1"].grad_norm > 0.0
    assert stats["fc2"].grad_norm > 0.0
    assert isinstance(stats["fc1"].grad_norm, float)
    assert isinstance(stats["fc2"].grad_norm, float)

    monitor.remove_hooks()


def test_multiple_training_steps_updates_stats():
    """Verify metrics update continuously across training steps."""
    model = SimpleMLP()
    monitor = Monitor(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    initial_step = monitor.step_count

    for i in range(5):
        optimizer.zero_grad()
        x = torch.randn(2, 4)
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    assert monitor.step_count == initial_step + 5
    stats = monitor.get_stats()
    assert stats["fc1"].grad_norm > 0.0

    monitor.remove_hooks()


def test_thread_safe_stats_access():
    """Verify get_stats can be called concurrently during training without race conditions."""
    model = SimpleMLP()
    monitor = Monitor(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    stop_event = threading.Event()
    read_errors = []

    def reader():
        while not stop_event.is_set():
            try:
                stats = monitor.get_stats()
                for name, layer_stat in stats.items():
                    assert isinstance(layer_stat.mean_activation, float)
                    assert isinstance(layer_stat.grad_norm, float)
                time.sleep(0.001)
            except Exception as e:
                read_errors.append(e)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    # Run 50 training steps
    for _ in range(50):
        optimizer.zero_grad()
        x = torch.randn(4, 4)
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    stop_event.set()
    reader_thread.join(timeout=2.0)

    assert len(read_errors) == 0
    monitor.remove_hooks()


def test_custom_module_or_sequential():
    """Verify monitor handles nn.Sequential models with Conv2d and Linear."""
    seq_model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 2),
    )
    monitor = Monitor(seq_model)
    stats = monitor.get_stats()
    assert len(stats) >= 3  # Conv2d, ReLU, Linear (Flatten is leaf too)

    x = torch.randn(2, 1, 6, 6)
    out = seq_model(x)
    loss = out.sum()
    loss.backward()

    updated_stats = monitor.get_stats()
    for name, s in updated_stats.items():
        assert isinstance(s.mean_activation, float)
        assert isinstance(s.grad_norm, float)

    monitor.remove_hooks()


def test_inplace_activation_layers():
    """Verify monitor gracefully handles layers configured with inplace=True."""
    inplace_model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(inplace=True),
        nn.Linear(8, 2),
    )
    monitor = Monitor(inplace_model)

    x = torch.randn(2, 4)
    out = inplace_model(x)
    loss = out.sum()
    loss.backward()

    stats = monitor.get_stats()
    # Check that linear layers and ReLU are present and don't cause crashes
    assert len(stats) >= 3
    for name, s in stats.items():
        assert isinstance(s.mean_activation, float)
        assert isinstance(s.grad_norm, float)

    monitor.remove_hooks()


def test_context_manager_cleanup():
    """Verify __enter__ and __exit__ cleanly remove hooks upon context exit."""
    model = SimpleMLP()
    with Monitor(model) as monitor:
        assert len(monitor._hook_handles) > 0
        x = torch.randn(2, 4)
        out = model(x)
        loss = out.sum()
        loss.backward()
        stats = monitor.get_stats()
        assert stats["fc1"].mean_activation != 0.0

    # After exit, hook handles must be strictly removed
    assert len(monitor._hook_handles) == 0

    # Running another forward pass should not alter monitor stats
    prev_step = monitor.step_count
    _ = model(torch.randn(2, 4))
    assert monitor.step_count == prev_step


def test_context_manager_cleanup_on_exception():
    """Verify hooks are removed even if an exception occurs inside the with block."""
    model = SimpleMLP()
    monitor_ref = None
    try:
        with Monitor(model) as monitor:
            monitor_ref = monitor
            assert len(monitor._hook_handles) > 0
            raise ValueError("Training interrupted")
    except ValueError:
        pass

    assert monitor_ref is not None
    assert len(monitor_ref._hook_handles) == 0
