import torch
import torch.nn as nn

def slerp_rotate(hidden_state, unit_steering_vector, alpha):
    """
    Spherical Linear Interpolation (SLERP) on the activation manifold.
    
    Rotates the direction of the hidden state towards the truth vector
    by fraction alpha, while strictly preserving the activation L2 norm.
    This avoids the activation explosion and stuttering ('p-p-p-p') of naive addition.
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
    def __init__(self, steering_tensor, alpha=0.25, threshold=15.0, mode="adaptive"):
        """
        Args:
            steering_tensor: The PCA Truth Axis tensor.
            alpha: Rotational angle strength [0.0, 1.0].
            threshold: Alignment score below which steering triggers.
            mode: 'adaptive' (steer only when drifting toward hallucination),
                  'constant' (unconditional rotational steering),
                  'inception' (reverse steering with negative alpha to induce hallucination).
        """
        self.steering_vector = steering_tensor / torch.norm(steering_tensor)
        self.alpha = alpha
        self.threshold = threshold
        self.mode = mode
        self.handle = None

    def __call__(self, module, inputs, output):
        hidden_states = output[0]
        vec = self.steering_vector.to(hidden_states.device, hidden_states.dtype)
        
        # Measure alignment with truth axis
        alignment = torch.matmul(hidden_states, vec)
        last_score = alignment[:, -1].mean().item() if len(alignment.shape) == 2 else alignment.mean().item()
        
        should_steer = False
        if self.mode == "constant":
            should_steer = True
        elif self.mode == "inception":
            # Negative rotation forces the hallucinated state
            should_steer = True
        elif self.mode == "adaptive":
            if last_score < self.threshold:
                should_steer = True
                
        if should_steer:
            target_alpha = -self.alpha if self.mode == "inception" else self.alpha
            output[0][:] = slerp_rotate(hidden_states, vec, target_alpha)
            
        return output

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
