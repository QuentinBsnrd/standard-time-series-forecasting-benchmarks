import torch
import torch.nn.functional as F
import numpy as np

def _find_target_layer(model):
    for name, m in model.named_modules():
        cname = m.__class__.__name__.lower()
        if 'conv' in cname or 'linear' in cname or 'layernorm' in cname:
            return name
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            return name
    raise RuntimeError("None of the model's layers can be used for Grad-CAM.") 

class ActivationsAndGradients:
    """
    Object to hook and retrieve activations and gradients from a module.
    """
    def __init__(self, module):
        self.module = module
        self.activations = None
        self.gradients = None
        # forward hook
        def forward_hook(module, input, output):
            self.activations = output
        # backward hook (full backward hook recommended)
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        # register
        self.fh = module.register_forward_hook(forward_hook)
        if hasattr(module, "register_full_backward_hook"):
            self.bh = module.register_full_backward_hook(backward_hook)
        else:
            self.bh = module.register_backward_hook(backward_hook)

    def remove(self):
        try:
            self.fh.remove()
        except Exception:
            pass
        try:
            self.bh.remove()
        except Exception:
            pass

def grad_cam_timeseries(args, model, inputs, target=None, device=None, eps=1e-8):
    """
    Compute Grad-CAM for time series models.
    """
    layer_name = None
    if args.model == "PatchMixer":
        layer_name = 'model.W_P'
    elif args.model == "PatchTST":
        layer_name = 'model.head.linear'
    elif args.model == "DLinear":
        layer_name = 'Linear_Seasonal'
        
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    inputs = inputs.to(device)
    # find target layer if not specified
    if layer_name is None:
        layer_name = _find_target_layer(model)
    named = dict(model.named_modules())
    if layer_name not in named:
        raise ValueError(f"Layer {layer_name} not found in model. Available: {list(named.keys())[:20]}")

    target_module = named[layer_name]
    hook = ActivationsAndGradients(target_module)

    # Forward
    outputs = model(inputs)
    if target is None:
        if isinstance(outputs, (list, tuple)):
            out_tensor = outputs[0]
        else:
            out_tensor = outputs
        target_scalar = out_tensor.mean()
    else:
        target_scalar = target(outputs)

    model.zero_grad()
    # backward to retrieve gradients
    target_scalar.backward(retain_graph=True)

    activations = hook.activations
    gradients = hook.gradients
    hook.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Impossible to retrieve activations or gradients (hook not triggered). Choose another target layer.") 

    # normalization and gradients weighting computing
    a = activations
    g = gradients

    if a.dim() == 3:
        if a.shape[1] < a.shape[2]:
            # [B, L, C]
            a_proc = a.permute(0,2,1).contiguous()
            g_proc = g.permute(0,2,1).contiguous()
        else:
            # assume [B, C, L]
            a_proc = a
            g_proc = g
    elif a.dim() == 2:
        # [B, L] or [B, C] -> treat L = last dim, add channel dim
        a_proc = a.unsqueeze(1)
        g_proc = g.unsqueeze(1)
    else:
        *batch_dims, last1, last2 = a.shape
        a_proc = a.view(a.shape[0], -1, a.shape[-1])
        g_proc = g.view(g.shape[0], -1, g.shape[-1])

    # a_proc: [B, C, L], g_proc: [B, C, L]
    # weights alpha_k = global average pooling over time -> shape [B, C, 1]
    alphas = g_proc.mean(dim=2, keepdim=True)

    # weighted sum over channels: cam = ReLU(sum_k alpha_k * A_k)
    cam = (alphas * a_proc).sum(dim=1)
    cam = F.relu(cam)

    # normalization per sample
    cam_np = cam.detach().cpu().numpy()
    cam_norm = np.zeros_like(cam_np)
    for i in range(cam_np.shape[0]):
        m = cam_np[i]
        # min-max normalize
        mm = m.max()
        if mm > eps:
            cam_norm[i] = m / (mm + eps)
        else:
            cam_norm[i] = m - m.min()
    return cam_norm  # shape (B, L)