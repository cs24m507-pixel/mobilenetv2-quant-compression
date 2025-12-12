
# quant.py
import torch, torch.nn as nn

def _eps(): return torch.finfo(torch.float32).eps
def bits_to_mb(bits: float): return (bits / 8.0) / (1024 * 1024)

# --- per-tensor affine quant (weights or activations)
def fake_quant_per_tensor_affine(x: torch.Tensor, num_bits=8, symmetric=False):
    if symmetric:
        qmin, qmax = -128, 127
        max_abs = torch.max(x.min().abs(), x.max().abs())
        scale = torch.clamp(2 * max_abs / (qmax - qmin), min=_eps())
        zp = torch.tensor(0.0, device=x.device)
    else:
        qmin, qmax = 0, 2**num_bits - 1
        mn, mx = x.min(), x.max()
        scale = torch.clamp((mx - mn) / max(qmax - qmin, 1), min=_eps())
        zp = torch.tensor(qmin, device=x.device) - torch.round(mn / scale)
    q = torch.round(x / scale + zp).clamp(qmin, qmax)
    return (q - zp) * scale, scale, zp

# --- per-channel symmetric quant (Conv/Linear weights)
def fake_quant_per_channel_symmetric(w: torch.Tensor, axis=0, num_bits=8):
    qmin, qmax = -128, 127
    min_vals = w.min(dim=axis, keepdim=True).values
    max_vals = w.max(dim=axis, keepdim=True).values
    scales = torch.clamp(2 * torch.max(min_vals.abs(), max_vals.abs()) / (qmax - qmin), min=_eps())
    q = torch.round(w / scales).clamp(qmin, qmax)
    return q * scales, scales, 0.0

def apply_weight_quantization_per_channel(model: nn.Module, axis=0, num_bits=8):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            wq, _, _ = fake_quant_per_channel_symmetric(m.weight.data, axis=axis, num_bits=num_bits)
            m.weight.data.copy_(wq)
    return model

# --- EMA observer for activation calibration
class EMAMinMaxObserver:
    def __init__(self, momentum=0.9, symmetric=False, num_bits=8):
        self.momentum, self.symmetric, self.num_bits = momentum, symmetric, num_bits
        self.initialized, self.min_val, self.max_val = False, None, None
    def update(self, x: torch.Tensor):
        mn, mx = x.min().detach(), x.max().detach()
        if not self.initialized:
            self.min_val, self.max_val, self.initialized = mn, mx, True
        else:
            m = self.momentum
            self.min_val = m*self.min_val + (1-m)*mn
            self.max_val = m*self.max_val + (1-m)*mx
    def get_params(self):
        if not self.initialized:
            return torch.tensor(1.0), torch.tensor(0.0)
        if self.symmetric:
            qmin, qmax = -128, 127
            max_abs = torch.max(self.min_val.abs(), self.max_val.abs())
            scale = torch.clamp(2 * max_abs / (qmax - qmin), min=_eps())
            zp = torch.tensor(0.0)
        else:
            qmin, qmax = 0, 2**self.num_bits - 1
            scale = torch.clamp((self.max_val - self.min_val) / max(qmax - qmin, 1), min=_eps())
            zp = torch.tensor(qmin) - torch.round(self.min_val / scale)
        return scale, zp

def attach_activation_observers(model: nn.Module, modules=None, symmetric=False, num_bits=8, momentum=0.9):
    modules = modules or list(model.features)
    obs_map = {}
    for layer in modules:
        obs = EMAMinMaxObserver(momentum, symmetric, num_bits); obs_map[layer] = obs
        def calib_hook(module, inputs, output, _obs=obs):
            _obs.update(output.detach()); return output
        layer.register_forward_hook(calib_hook)
    return obs_map

def replace_activation_with_fake_quant(model: nn.Module, obs_map: dict):
    for layer, obs in obs_map.items():
        scale, zp = obs.get_params()
        def fq_hook(module, inputs, output, _scale=scale, _zp=zp, _nb=obs.num_bits, _sym=obs.symmetric):
            if _sym:
                q = torch.round(output / _scale).clamp(-128, 127); return q * _scale
            else:
                q = torch.round(output / _scale + _zp).clamp(0, 2**_nb - 1); return (q - _zp) * _scale
        layer.register_forward_hook(fq_hook)

def compress_model_calibrated(model, calib_loader, device,
                              weight_bits=8, activation_bits=8,
                              act_symmetric=False, ema_momentum=0.9, calib_batches=50):
    model = apply_weight_quantization_per_channel(model, axis=0, num_bits=weight_bits)
    obs_map = attach_activation_observers(model, list(model.features),
                                          symmetric=act_symmetric, num_bits=activation_bits, momentum=ema_momentum)
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            if i >= calib_batches: break
            _ = model(x.to(device))
    replace_activation_with_fake_quant(model, obs_map)
    return model

# --- storage/size reporters (Q4)
def estimate_weight_storage_bits(model: nn.Module, weight_bits=8, per_channel=True):
    payload_bits, overhead_bits = 0, 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            w = m.weight
            payload_bits += w.numel() * weight_bits
            overhead_bits += (w.size(0) * 32) if per_channel else 32
    return payload_bits, overhead_bits

def profile_activations(model: nn.Module, dataloader, device, act_bits=8, samples=20):
    fp32_bits, q_bits, overhead_bits = 0, 0, 0
    def record_hook(module, inputs, output):
        nonlocal fp32_bits, q_bits, overhead_bits
        n = output.numel()
        fp32_bits += n * 32; q_bits += n * act_bits; overhead_bits += 32
    handles = [layer.register_forward_hook(record_hook) for layer in model.features]
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= samples: break
            _ = model(x.to(device))
    for h in handles: h.remove()
    return fp32_bits, q_bits, overhead_bits

def approximate_model_size_mb(model: nn.Module, weight_bits=8, per_channel=True):
    total_fp32 = sum(p.numel()*32 for p in model.parameters())
    convlin_fp32, quant_bits, overhead_bits = 0, 0, 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            w = m.weight
            convlin_fp32 += w.numel()*32
            quant_bits   += w.numel()*weight_bits
            overhead_bits += (w.size(0)*32) if per_channel else 32
    approx_bits = total_fp32 - convlin_fp32 + quant_bits + overhead_bits
    return bits_to_mb(approx_bits)

def report_sizes(model: nn.Module, weight_bits: int, activation_bits: int,
                 testloader, device, act_profile_samples: int = 20):
    w_payload_bits, w_over_bits = estimate_weight_storage_bits(model, weight_bits, per_channel=True)
    weights_mb_payload = bits_to_mb(w_payload_bits)
    weights_mb_over    = bits_to_mb(w_over_bits)
    weights_mb_total   = bits_to_mb(w_payload_bits + w_over_bits)

    act_fp32_bits, act_q_bits, act_over_bits = profile_activations(model, testloader, device,
                                                                   act_bits=activation_bits, samples=act_profile_samples)
    act_mb_fp32  = bits_to_mb(act_fp32_bits)
    act_mb_quant = bits_to_mb(act_q_bits + act_over_bits)

    model_mb_fp32_total   = bits_to_mb(sum(p.numel()*32 for p in model.parameters()))
    model_mb_approx_total = approximate_model_size_mb(model, weight_bits, per_channel=True)

    cr_weights     = (bits_to_mb(w_payload_bits*32) / bits_to_mb((w_payload_bits + w_over_bits)*weight_bits)) if (w_payload_bits + w_over_bits)>0 else float('inf')
    cr_activations = (act_mb_fp32 / act_mb_quant) if act_mb_quant>0 else float('inf')
    cr_model       = (model_mb_fp32_total / model_mb_approx_total) if model_mb_approx_total>0 else float('inf')

    return {
        "weights_mb_payload": weights_mb_payload,
        "weights_mb_overhead": weights_mb_over,
        "weights_mb_total": weights_mb_total,
        "activations_mb_fp32": act_mb_fp32,
        "activations_mb_quant": act_mb_quant,
        "model_mb_fp32_total": model_mb_fp32_total,
        "model_mb_approx_total": model_mb_approx_total,
        "compression_ratio_weights": cr_weights,
        "compression_ratio_activations": cr_activations,
        "compression_ratio_model": cr_model,
    }
