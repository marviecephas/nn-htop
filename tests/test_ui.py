import asyncio
import pytest
import torch
import torch.nn as nn
from textual.widgets import DataTable
from nn_htop.monitor import Monitor
from nn_htop.ui import NNHtopApp

pytestmark = pytest.mark.filterwarnings("ignore:Full backward hook is firing.*")


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_ui_table_structure_and_columns():
    """Verify DataTable has the required columns and populates layer rows."""
    async def _run():
        model = SimpleModel()
        monitor = Monitor(model)
        app = NNHtopApp(monitor=monitor)

        async with app.run_test() as pilot:
            table = app.query_one(DataTable)
            assert table is not None

            # Check required columns
            col_labels = [str(col.label) for col in table.columns.values()]
            assert "Layer Name" in col_labels
            assert "Type" in col_labels
            assert "Mean Activation" in col_labels
            assert "Gradient Norm" in col_labels

            # Check that all layers are in the table
            layer_names = monitor.get_layer_names()
            assert len(table.rows) == len(layer_names)
            for name in layer_names:
                assert name in table.rows
        monitor.remove_hooks()

    asyncio.run(_run())


def test_ui_updates_when_monitor_changes():
    """Verify table cells update when forward and backward passes execute."""
    async def _run():
        model = SimpleModel()
        monitor = Monitor(model)
        app = NNHtopApp(monitor=monitor)

        async with app.run_test() as pilot:
            table = app.query_one(DataTable)

            # Initially values are zero / uninitialized
            # Now perform a forward and backward step
            x = torch.randn(2, 4)
            out = model(x)
            loss = out.sum()
            loss.backward()

            # Update the UI
            app.update_table()

            # Check cell values for fc1
            stats = monitor.get_stats()
            fc1_stat = stats["fc1"]

            # Mean activation and grad norm should be non-zero
            assert fc1_stat.mean_activation != 0.0
            assert fc1_stat.grad_norm != 0.0

            # Verify the table displays updated values
            row_fc1 = table.get_row("fc1")
            # Row values: Layer Name, Type, Mean Activation, Gradient Norm, Health, Activity
            assert "fc1" in str(row_fc1[0])
            assert "Linear" in str(row_fc1[1])
            assert f"{fc1_stat.mean_activation:+.5f}" in str(row_fc1[2]) or f"{fc1_stat.mean_activation:.5f}" in str(row_fc1[2])
            assert f"{fc1_stat.grad_norm:.5f}" in str(row_fc1[3])
        monitor.remove_hooks()

    asyncio.run(_run())


def test_ui_decoupled_state_and_5fps_interval():
    """Verify UI defaults to 0.2s (5 FPS) refresh interval and pulls from state dict."""
    model = SimpleModel()
    monitor = Monitor(model)
    app = NNHtopApp(monitor=monitor)

    assert app.refresh_interval == 0.2

    # Verify PyTorch model has no reference to Textual App or DataTable
    assert not hasattr(model, "table")
    assert not hasattr(model, "app")
    assert not hasattr(monitor, "table")

    # PyTorch hooks write strictly to monitor state dict
    x = torch.randn(2, 4)
    out = model(x)
    loss = out.sum()
    loss.backward()

    state_dict = monitor.get_stats()
    assert "fc1" in state_dict
    assert state_dict["fc1"].mean_activation != 0.0

    monitor.remove_hooks()
