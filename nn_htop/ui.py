"""Textual-based TUI for real-time PyTorch neural network layer telemetry."""

import inspect
import math
import threading
import time
from typing import Any, Callable, Dict, Optional, Union
import torch
import torch.nn as nn
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static
from rich.text import Text

from nn_htop.monitor import Monitor, LayerStats


def _make_bar(value: float, max_expected: float = 1.0, width: int = 12) -> str:
    """Generates an ASCII meter bar for visualizing metric magnitude."""
    if math.isnan(value) or math.isinf(value):
        return "[red]ERROR[/red]"
    ratio = min(max(abs(value) / max(max_expected, 1e-6), 0.0), 1.0)
    filled = int(round(ratio * width))
    bar = "■" * filled + "·" * (width - filled)
    if ratio > 0.8:
        return f"[red]{bar}[/red]"
    elif ratio > 0.3:
        return f"[green]{bar}[/green]"
    else:
        return f"[cyan]{bar}[/cyan]"


def _get_health_badge(mean_act: float, grad_norm: float, layer_type: str) -> Text:
    """Returns a color-coded health diagnostic badge for a layer."""
    if math.isnan(mean_act) or math.isnan(grad_norm):
        return Text("NaN DETECTED", style="bold red")
    if math.isinf(mean_act) or math.isinf(grad_norm) or grad_norm > 1000.0:
        return Text("EXPLODING", style="bold red")
    if grad_norm == 0.0:
        return Text("ZERO GRAD", style="bold yellow")
    if layer_type in ("ReLU", "LeakyReLU") and abs(mean_act) < 1e-7:
        return Text("DEAD RELU", style="bold yellow")
    if grad_norm < 1e-5:
        return Text("VANISHING", style="yellow")
    return Text("LEARNING", style="bold green")


class StatsHeader(Static):
    """Header banner displaying global training state, step count, and loss."""

    def update_metrics(self, step: int, loss: float, total_layers: int, learning_layers: int, paused: bool):
        status_text = "[bold yellow]❚❚ PAUSED[/bold yellow]" if paused else "[bold green]● RUNNING[/bold green]"
        loss_text = f"{loss:.4f}" if loss > 0 else "N/A"

        self.update(
            f" [bold cyan]nn-htop[/bold cyan] │ Status: {status_text} │ "
            f"Step: [bold white]{step}[/bold white] │ "
            f"Loss: [bold magenta]{loss_text}[/bold magenta] │ "
            f"Layers: [bold green]{learning_layers}/{total_layers} learning[/bold green]"
        )


class NNHtopApp(App):
    """Textual application monitoring neural network layers in real time."""

    CSS = """
    Screen {
        background: #111418;
        color: #e0e6ed;
    }

    Header {
        background: #1b2129;
        color: #00d2ff;
        text-style: bold;
    }

    #stats-banner {
        height: 3;
        background: #181e26;
        border: solid #2c3848;
        padding: 0 1;
        content-align: left middle;
    }

    #table-container {
        height: 1fr;
        border: solid #252e3b;
        margin: 0 0;
    }

    DataTable {
        height: 1fr;
        background: #111418;
    }

    Footer {
        background: #1b2129;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("space", "toggle_pause", "Pause/Resume"),
        ("p", "toggle_pause", "Pause/Resume"),
        ("r", "reset_stats", "Reset Step"),
    ]

    def __init__(
        self,
        monitor: Monitor,
        refresh_interval: float = 0.2,
        pause_event: Optional[threading.Event] = None,
        stop_event: Optional[threading.Event] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.monitor = monitor
        # 5 FPS (200ms) interval to ensure smooth rendering on massive models
        self.refresh_interval = refresh_interval
        self.pause_event = pause_event or threading.Event()
        self.stop_event = stop_event
        self.is_paused = False
        self._columns_added = False
        self._row_keys = set()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield StatsHeader(id="stats-banner")
        with Container(id="table-container"):
            yield DataTable(id="metrics-table", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"

        # Initialize columns
        table.add_column("Layer Name", key="layer_name")
        table.add_column("Type", key="type")
        table.add_column("Mean Activation", key="mean_act")
        table.add_column("Gradient Norm", key="grad_norm")
        table.add_column("Health", key="health")
        table.add_column("Grad Activity", key="activity")
        self._columns_added = True

        # Pre-populate rows for registered layers
        stats = self.monitor.get_stats()
        for name in self.monitor.get_layer_names():
            s = stats.get(name)
            layer_type = s.layer_type if s else "Module"
            table.add_row(
                Text(name, style="bold cyan"),
                Text(layer_type, style="white"),
                "0.00000",
                "0.00000",
                Text("INITIALIZING", style="dim"),
                "············",
                key=name,
            )
            self._row_keys.add(name)

        # UI pulls from state dictionary at 5 FPS (0.2s) without blocking PyTorch thread
        self.set_interval(0.2, self.update_table)

    def update_table(self) -> None:
        """Dynamically updates the DataTable with the latest telemetry from Monitor."""
        try:
            table = self.query_one(DataTable)
            stats = self.monitor.get_stats()
            layer_names = self.monitor.get_layer_names()

            total_layers = len(layer_names)
            learning_layers = 0

            for name in layer_names:
                if name not in stats:
                    continue
                s = stats[name]

                # If layer is not yet in table rows, insert it
                if name not in self._row_keys:
                    table.add_row(
                        Text(name, style="bold cyan"),
                        Text(s.layer_type, style="white"),
                        f"{s.mean_activation:+.5f}",
                        f"{s.grad_norm:.5f}",
                        _get_health_badge(s.mean_activation, s.grad_norm, s.layer_type),
                        _make_bar(s.grad_norm, max_expected=0.5),
                        key=name,
                    )
                    self._row_keys.add(name)
                else:
                    # Non-blocking, flicker-free cell updates
                    table.update_cell(name, "layer_name", Text(name, style="bold cyan"))
                    table.update_cell(name, "type", Text(s.layer_type, style="white"))
                    table.update_cell(name, "mean_act", f"{s.mean_activation:+.5f}")
                    table.update_cell(name, "grad_norm", f"{s.grad_norm:.5f}")
                    table.update_cell(name, "health", _get_health_badge(s.mean_activation, s.grad_norm, s.layer_type))
                    table.update_cell(name, "activity", _make_bar(s.grad_norm, max_expected=0.5))

                if s.grad_norm > 1e-6:
                    learning_layers += 1

            # Update banner
            banner = self.query_one(StatsHeader)
            banner.update_metrics(
                step=self.monitor.step_count,
                loss=self.monitor.latest_loss,
                total_layers=total_layers,
                learning_layers=learning_layers,
                paused=self.is_paused,
            )
        except Exception:
            # Prevent unexpected errors during shutdown or before mount from crashing the app
            pass

    def action_toggle_pause(self) -> None:
        """Toggles pause/resume state for the background training loop."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_event.set()
        else:
            self.pause_event.clear()
        self.update_table()

    def action_reset_stats(self) -> None:
        """Resets the monitor step counter."""
        self.monitor.record_step(step=0)
        self.update_table()

    def on_unmount(self) -> None:
        """Signals stop event when UI is exited."""
        if self.stop_event:
            self.stop_event.set()


def launch_tui(
    target: Any = None,
    monitor: Optional[Monitor] = None,
    model: Optional[Any] = None,
    train_fn: Optional[Callable] = None,
    step_delay: float = 0.05,
    *args,
    **kwargs,
):
    """Launches the nn-htop TUI for a given Monitor instance or nn.Module."""
    # Resolve target defensively across all calling conventions
    candidate = monitor if monitor is not None else (target if target is not None else model)
    if candidate is None:
        raise ValueError("A Monitor instance or torch.nn.Module must be provided to launch_tui.")

    if isinstance(candidate, Monitor):
        resolved_monitor = candidate
        resolved_model = candidate.model
    elif isinstance(candidate, nn.Module):
        resolved_model = candidate
        resolved_monitor = Monitor(resolved_model)
    else:
        raise TypeError(f"Expected Monitor or torch.nn.Module, got {type(candidate).__name__}")

    monitor = resolved_monitor
    model = resolved_model

    stop_event = threading.Event()
    pause_event = threading.Event()

    def default_training_runner():
        from nn_htop.harness import run_training_loop
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        step = 0

        while not stop_event.is_set():
            # Handle pause
            if pause_event.is_set():
                time.sleep(0.05)
                continue

            # Run synthetic training batch
            inputs = torch.randn(32, 16)
            targets = torch.randint(0, 4, (32,))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            monitor.record_step(loss=loss.item())
            step += 1

            if step_delay > 0:
                time.sleep(step_delay)

    if train_fn is not None:
        try:
            if len(inspect.signature(train_fn).parameters) == 0:
                worker_thread = threading.Thread(target=train_fn, daemon=True)
            else:
                worker_thread = threading.Thread(target=train_fn, args=(monitor,), daemon=True)
        except (ValueError, TypeError):
            worker_thread = threading.Thread(target=train_fn, daemon=True)
    else:
        worker_thread = threading.Thread(target=default_training_runner, daemon=True)
    worker_thread.start()

    app = NNHtopApp(
        monitor=monitor,
        pause_event=pause_event,
        stop_event=stop_event,
    )

    try:
        app.run()
    finally:
        stop_event.set()
        worker_thread.join(timeout=1.0)
        monitor.remove_hooks()


# Backwards compatibility alias
launch_htop = launch_tui
