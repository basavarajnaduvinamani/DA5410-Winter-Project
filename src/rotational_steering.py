import torch
import torch.nn as nn

def slerp_rotate(hidden_state, unit_steering_vector, alpha):
    """
    Spherical Linear Interpolation (SLERP) on the activation manifold.
    
    Rotates the direction of the hidden state towards the truth vector
    by fraction alpha, while strictly preserving the activation L2 norm.
    Numerical norm preservation is tested; linguistic effects require model runs.
    """
    norm = torch.norm(hidden_state, dim=-1, keepdim=True)
    h_unit = hidden_state / (norm + 1e-8)
    
    u = unit_steering_vector.to(hidden_state.device, hidden_state.dtype)
    u = u / (torch.norm(u) + 1e-8)
    
    # Cosine similarity clamp
    cos_theta = torch.sum(h_unit * u, dim=-1, keepdim=True).clamp(-0.9999, 0.9999)
    theta = torch.acos(cos_theta)
    
    sin_theta = torch.sin(theta)
    w1 = torch.sin((1.0 - alpha) * theta) / (sin_theta + 1e-8)
    w2 = torch.sin(alpha * theta) / (sin_theta + 1e-8)
    
    h_rot = w1 * h_unit + w2 * u
    h_rot = h_rot / (torch.norm(h_rot, dim=-1, keepdim=True) + 1e-8)
    
    return h_rot * norm

class RotationalSteeringHook:
    """
    PyTorch forward hook for conditional rotational steering (Smart Switch).
    """
    def __init__(self, steering_tensor, alpha=0.25, threshold=0.0, mode="adaptive", operation="slerp", record_trace=False):
        """
        Args:
            steering_tensor: The PCA Truth Axis tensor.
            alpha: Rotational angle strength [0.0, 1.0].
            threshold: Cosine alignment in [-1,1] below which steering triggers.
            mode: 'adaptive' (steer only when drifting toward hallucination),
                  'constant' (unconditional rotational steering),
                  'inception' (reverse steering with negative alpha to induce hallucination).
        """
        if torch.norm(steering_tensor) == 0:
            raise ValueError("Steering vector cannot be zero")
        if mode not in {"adaptive", "constant", "inception"}:
            raise ValueError(mode)
        if operation not in {"slerp", "addition"}:
            raise ValueError(operation)
        if not -1 <= threshold <= 1:
            raise ValueError("Gate is cosine alignment in [-1, 1]")
        self.record_trace = record_trace
        self.trace = []
        self.operation = operation
        self.steering_vector = steering_tensor / torch.norm(steering_tensor)
        self.alpha = alpha
        self.threshold = threshold
        self.mode = mode
        self.handle = None

    def __call__(self, module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        terminal = hidden_states[:, -1:, :]
        vec = self.steering_vector.to(hidden_states.device, hidden_states.dtype)
        
        # Measure alignment with truth axis
        alignment = torch.nn.functional.cosine_similarity(terminal, vec, dim=-1)
        mask = alignment.lt(self.threshold).unsqueeze(-1)
        if self.mode != "adaptive":
            mask = torch.ones_like(mask)
        if self.record_trace:
            self.trace.append(dict(alignment=alignment.detach().float().cpu().flatten().tolist(), steered=mask.detach().cpu().flatten().tolist()))
        target_alpha = -self.alpha if self.mode == "inception" else self.alpha
        changed = (slerp_rotate(terminal, vec, target_alpha) if self.operation == "slerp"
                   else terminal + target_alpha * vec)
        new_states = hidden_states.clone()
        new_states[:, -1:, :] = torch.where(mask, changed, terminal)
        return (new_states,) + output[1:] if isinstance(output, tuple) else new_states

class SteeringContext:
    """Safe context manager to register and automatically detach forward hooks."""
    def __init__(self, layer_module, hook_fn):
        self.layer = layer_module
        self.hook_fn = hook_fn
        self.handle = None

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self.hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle:
            self.handle.remove()
