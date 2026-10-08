"""PyTorch layer monitoring core logic using forward and backward hooks."""

from dataclasses import dataclass, field
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


@dataclass
class LayerStats:
    """Stores telemetry metrics for an individual neural network layer."""
    name: str
    layer_type: str
    mean_activation: float = 0.0
    grad_norm: float = 0.0
    activation_history: List[float] = field(default_factory=list)
    grad_history: List[float] = field(default_factory=list)
    last_updated: float = 0.0

    def copy(self) -> "LayerStats":
        return LayerStats(
            name=self.name,
            layer_type=self.layer_type,
            mean_activation=self.mean_activation,
            grad_norm=self.grad_norm,
            activation_history=list(self.activation_history),
            grad_history=list(self.grad_history),
            last_updated=self.last_updated,
        )


class Monitor:
    """Attaches forward and backward hooks to PyTorch nn.Module to track activations and gradients."""

    def __init__(self, model: nn.Module, max_history: int = 50):
        self.model = model
        self.max_history = max_history
        self._lock = threading.Lock()
        self._stats: Dict[str, LayerStats] = {}
        self._hook_handles: List[Any] = []
        self._layer_order: List[str] = []
        self._original_inplace: Dict[nn.Module, bool] = {}
        self.step_count: int = 0
        self.latest_loss: float = 0.0

        self._attach_hooks()

    def _is_target_layer(self, module: nn.Module) -> bool:
        """Determines if a module should be monitored. Monitors leaf modules by default."""
        children = list(module.children())
        return len(children) == 0 and module != self.model

    def _attach_hooks(self):
        """Discovers modules and attaches forward and backward hooks."""
        for name, module in self.model.named_modules():
            # If named_modules has leaf modules, register them
            if self._is_target_layer(module):
                # Gracefully handle layers with inplace=True to prevent autograd BackwardHookFunction conflicts
                if hasattr(module, "inplace") and getattr(module, "inplace", False):
                    self._original_inplace[module] = module.inplace
                    module.inplace = False

                # Fallback name for root or unnamed layers
                layer_name = name if name else module.__class__.__name__
                layer_type = module.__class__.__name__

                with self._lock:
                    self._stats[layer_name] = LayerStats(
                        name=layer_name,
                        layer_type=layer_type,
                    )
                    self._layer_order.append(layer_name)

                # 1. Forward hook: compute mean activation
                f_handle = module.register_forward_hook(self._make_forward_hook(layer_name))
                self._hook_handles.append(f_handle)

                # 2. Backward hook: compute gradient norm using register_full_backward_hook
                b_handle = module.register_full_backward_hook(self._make_backward_hook(layer_name))
                self._hook_handles.append(b_handle)

    def _extract_tensor(self, val: Any) -> Optional[torch.Tensor]:
        """Extracts the primary Tensor from hook arguments or outputs."""
        if isinstance(val, torch.Tensor):
            return val
        elif isinstance(val, (tuple, list)):
            for item in val:
                if isinstance(item, torch.Tensor):
                    return item
        return None

    def _make_forward_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs: Any, output: Any):
            try:
                tensor = self._extract_tensor(output)
                if tensor is not None:
                    # Calculate mean activation value (detached float to prevent memory leak)
                    mean_val = float(tensor.detach().float().mean().item())
                    now = time.time()
                    with self._lock:
                        # Auto-increment step if this is the first registered layer
                        if self._layer_order and layer_name == self._layer_order[0]:
                            self.step_count += 1
                        if layer_name in self._stats:
                            stat = self._stats[layer_name]
                            stat.mean_activation = mean_val
                            stat.activation_history.append(mean_val)
                            if len(stat.activation_history) > self.max_history:
                                stat.activation_history.pop(0)
                            stat.last_updated = now
            except Exception:
                # Gracefully handle unexpected tensor structures or in-place mutations
                pass
        return hook

    def _make_backward_hook(self, layer_name: str):
        def hook(module: nn.Module, grad_input: Any, grad_output: Any):
            try:
                tensor = self._extract_tensor(grad_output)
                if tensor is not None:
                    # Calculate L2 gradient norm (detached float to prevent memory leak)
                    norm_val = float(tensor.detach().float().norm(2).item())
                    now = time.time()
                    with self._lock:
                        if layer_name in self._stats:
                            stat = self._stats[layer_name]
                            stat.grad_norm = norm_val
                            stat.grad_history.append(norm_val)
                            if len(stat.grad_history) > self.max_history:
                                stat.grad_history.pop(0)
                            stat.last_updated = now
            except Exception:
                # Gracefully handle layers that use inplace=True or view modifications
                pass
        return hook

    def record_step(self, step: Optional[int] = None, loss: Optional[float] = None):
        """Optionally called by the training loop to track step numbers and loss."""
        with self._lock:
            if step is not None:
                self.step_count = step
            if loss is not None:
                self.latest_loss = float(loss)

    def get_stats(self) -> Dict[str, LayerStats]:
        """Thread-safe snapshot copy of current layer statistics."""
        with self._lock:
            return {k: v.copy() for k, v in self._stats.items()}

    def get_layer_names(self) -> List[str]:
        """Ordered list of monitored layer names."""
        with self._lock:
            return list(self._layer_order)

    def remove_hooks(self):
        """Safely removes all registered hooks and restores module attributes."""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles.clear()

        # Restore original inplace attributes
        for module, orig_val in self._original_inplace.items():
            try:
                module.inplace = orig_val
            except Exception:
                pass
        self._original_inplace.clear()

    def close(self):
        """Alias for remove_hooks."""
        self.remove_hooks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
