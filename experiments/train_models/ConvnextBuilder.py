# Requires: Python 3.10+ (for match/case), PyTorch >= 1.12
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_filter_label(f):
    if not isinstance(f, float):
        if len(f.shape) == 2:
            f = f[3][3].item()
        else:
            f = f[0][3][3].item()

    match f:
        case 0.9827055335044861:
            return 'DoG'
        case 0.9208850860595703:
            return 'Gaussian'
        case 0.01930227316915989:
            return 'd1 v'
        case 0.23042117059230804:
            return 'Unknown d'
        case -0.45654532313346863:
            return 'Unknown r'
        case -0.35520854592323303:
            return 'Unknown u'
        case -0.3508988320827484:
            return 'Unknown l'
        case -0.03321698680520058:
            return 'd1 h'
        case _:
            raise ValueError(f"Unknown filter")

# Map  labels -> operator kinds used below
LABEL_TO_KIND = {
    'Gaussian':  'smooth_T',     # T
    'DoG':       'lap_combo',    # T - γ7 ∇²₅ T
    'd1 v':      'dy_center_T',  # δy T (centered)
    'd1 h':      'dx_center_T',  # δx T (centered)
    'Unknown d': 'dy_plus_T',    # δy+ T
    'Unknown u': 'dy_minus_T',   # δy- T
    'Unknown r': 'dx_plus_T',    # δx+ T
    'Unknown l': 'dx_minus_T',   # δx- T
}

# ---------------------------
# Constrained DW-Conv module
# ---------------------------
Use_bessel = False
def discrete_gaussian_kernel_1d_sampled(sigmas, size=7, device=None, dtype=None):
    assert size % 2 == 1
    k = size // 2
    x = torch.arange(-k, k+1, device=device, dtype=dtype).view(1, -1)
    var = (sigmas ** 2).view(-1, 1)
    kern = torch.exp(- (x**2) / (2.0 * var + 1e-12))
    kern = kern / (kern.sum(dim=1, keepdim=True) + 1e-12)
    return kern

from torch.cuda.amp import autocast

def discrete_gaussian_kernel_1d_bessel_recursive(sigmas, size=7, device=None, dtype=None):
    assert size % 2 == 1
    k = size // 2
    var = (sigmas ** 2).view(-1, 1)

    with autocast(enabled=False):          # compute in fp32
        x = var.to(torch.float32)
        I0 = torch.special.i0(x)
        I1 = torch.special.i1(x)
        orders = [I0, I1]
        if k >= 2:
            Inm1, In = I0, I1
            eps = torch.finfo(x.dtype).eps
            for n in range(1, k):
                Inp1 = Inm1 - (2.0 * n) * In / (x + eps)
                Inp1 = torch.where(x == 0, torch.zeros_like(Inp1), Inp1)
                orders.append(Inp1)
                Inm1, In = In, Inp1
        I_all = torch.cat(orders[:k+1], dim=1)

    n_idx = torch.arange(-k, k+1, device=var.device).abs().view(1, -1).long()
    Iv = I_all.gather(1, n_idx.expand(var.size(0), n_idx.size(1)))

    kern = torch.exp(-var.to(torch.float32)) * Iv
    kern = kern / (kern.sum(dim=1, keepdim=True) + 1e-12)
    return kern.to(var.dtype)


    # Stack to (B, k+1) and gather |n| for positions -k..k
    I_all = torch.cat(orders[:k+1], dim=1)                     # (B, k+1)
    n_idx = torch.arange(-k, k+1, device=device).abs().view(1, -1)  # (1, size)
    Iv = I_all.gather(1, n_idx.expand(B, size))                # (B, size)

    # Discrete Gaussian kernel via Bessel form
    kern = torch.exp(-var) * Iv                                # (B, size)
    kern = kern / (kern.sum(dim=1, keepdim=True) + 1e-12)
    return kern


def discrete_gaussian_kernel_1d_bessel(sigmas, size=7, device=None, dtype=None):
    assert size % 2 == 1
    if not Use_bessel:
        return discrete_gaussian_kernel_1d_sampled(sigmas, size, device, dtype)
    k = size // 2
    n = torch.arange(-k, k+1, device=device, dtype=dtype).abs().view(1, -1)
    var = (sigmas ** 2).view(-1, 1)
    Iv = torch.special.iv(n.to(var.dtype), var)  # I_|n|(σ^2)
    kern = torch.exp(-var) * Iv
    kern = kern / (kern.sum(dim=1, keepdim=True) + 1e-12)
    return kern

def discrete_gaussian_kernel_2d(sigmas_x, sigmas_y, size=7, device=None, dtype=None):
    builder = discrete_gaussian_kernel_1d_bessel_recursive if Use_bessel else discrete_gaussian_kernel_1d_sampled
    kx = builder(sigmas_x, size=size, device=device, dtype=dtype)
    ky = builder(sigmas_y, size=size, device=device, dtype=dtype)
    return torch.einsum("bi,bj->bij", ky, kx)  # (B, k, k)

def diff_op_forward_x(K):
    out = torch.zeros_like(K); out[..., :, :-1] = K[..., :, 1:] - K[..., :, :-1]; return out
def diff_op_backward_x(K):
    out = torch.zeros_like(K); out[..., :, 1:] = K[..., :, 1:] - K[..., :, :-1]; return out
def diff_op_forward_y(K):
    out = torch.zeros_like(K); out[..., :-1, :] = K[..., 1:, :] - K[..., :-1, :]; return out
def diff_op_backward_y(K):
    out = torch.zeros_like(K); out[..., 1:, :] = K[..., 1:, :] - K[..., :-1, :]; return out
def diff_op_center_x(K):
    out = torch.zeros_like(K); out[..., :, 1:-1] = 0.5 * (K[..., :, 2:] - K[..., :, :-2]); return out
def diff_op_center_y(K):
    out = torch.zeros_like(K); out[..., 1:-1, :] = 0.5 * (K[..., 2:, :] - K[..., :-2, :]); return out
def laplacian_5point(K):
    out = torch.zeros_like(K)
    out[..., 1:-1, 1:-1] = (
        K[..., 0:-2, 1:-1] + K[..., 2:, 1:-1] +
        K[..., 1:-1, 0:-2] + K[..., 1:-1, 2:] -
        4.0 * K[..., 1:-1, 1:-1]
    )
    return out

FILTER_KINDS = [
    "dy_plus_T","dx_minus_T","dy_minus_T","dx_plus_T",
    "dx_center_T","dy_center_T","lap_combo","smooth_T",
]

def apply_operator(K, kind, gamma7=0.5):
    if   kind == "dy_plus_T":    return diff_op_forward_y(K)
    elif kind == "dx_minus_T":   return diff_op_backward_x(K)
    elif kind == "dy_minus_T":   return diff_op_backward_y(K)
    elif kind == "dx_plus_T":    return diff_op_forward_x(K)
    elif kind == "dx_center_T":  return diff_op_center_x(K)
    elif kind == "dy_center_T":  return diff_op_center_y(K)
    elif kind == "lap_combo":    return K - gamma7 * laplacian_5point(K)
    elif kind == "smooth_T":     return K
    else:
        raise ValueError(f"Unknown kind: {kind}")

class DepthwiseGaussianConv2d(nn.Module):
    """
    Depthwise 7x7 conv with kernels constrained to Gaussian-derived filters.
    Per-channel learnable (σx, σy) only; operator kind chosen per channel.
    """
    def __init__(self, in_channels, kernel_size=7, stride=1, padding=None, dilation=1,
                 bias=True, filter_kinds=None, gamma7=0.5, sigma_init=1.5, sigma_min=1e-3,
                 normalize_kernel=True):
        super().__init__()
        assert kernel_size % 2 == 1
        self.C = in_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = (kernel_size // 2) if padding is None else padding
        self.dilation = dilation
        self.gamma7 = gamma7
        self.normalize_kernel = normalize_kernel
        self.sigma_min = sigma_min

        if filter_kinds is None:
            kinds = (FILTER_KINDS * ((in_channels + 7) // 8))[:in_channels]
            self.filter_kinds = kinds
        else:
            assert len(filter_kinds) == in_channels
            for k in filter_kinds:
                assert k in FILTER_KINDS, f"Bad filter kind: {k}"
            self.filter_kinds = list(filter_kinds)

        s0 = torch.full((in_channels, 2), float(sigma_init))
        # softplus^-1 init
        self.sigma_raw = nn.Parameter(torch.log(torch.expm1(torch.clamp(s0, min=1e-6))))
        self.bias = nn.Parameter(torch.zeros(in_channels)) if bias else None

    def _make_kernels(self, device, dtype):
        sigmas = F.softplus(self.sigma_raw) + self.sigma_min
        sx, sy = sigmas[:, 0], sigmas[:, 1]
        K = discrete_gaussian_kernel_2d(sx, sy, size=self.k, device=device, dtype=dtype)  # (C,k,k)

        out = torch.empty_like(K)
        # group by kind for vectorized ops
        unique_kinds = sorted(set(self.filter_kinds))
        for kind in unique_kinds:
            idx = [i for i, knd in enumerate(self.filter_kinds) if knd == kind]
            idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            Ksub = K.index_select(0, idx_t)
            Fsub = apply_operator(Ksub, kind, gamma7=self.gamma7)
            out.index_copy_(0, idx_t, Fsub)

        if self.normalize_kernel:
            norms = out.flatten(1).norm(p=2, dim=1, keepdim=True) + 1e-12
            out = out / norms.view(-1, 1, 1)
        return out.unsqueeze(1)  # (C,1,k,k)

    @property
    def weight(self) -> torch.Tensor:
        """
        Compatibility shim so external code can read `.weight` like a Conv2d.
        This is NOT a Parameter; it is computed from (sigma_x, sigma_y) each call.
        """
        return self._make_kernels(self.sigma_raw.device, self.sigma_raw.dtype)

    def forward(self, x):
        W = self._make_kernels(x.device, x.dtype)
        return F.conv2d(x, W, bias=self.bias, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        groups=self.C)

# ---------------------------
# Helpers to traverse/replace
# ---------------------------
def _is_depthwise_7x7(conv: nn.Conv2d) -> bool:
    return isinstance(conv, nn.Conv2d) and conv.groups == conv.in_channels and conv.kernel_size == (7, 7)

def _set_module_by_name(root: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)

def _derive_kinds_from_weight(w: torch.Tensor) -> list[str]:
    """
    w: (C, 1, 7, 7) weight of a depthwise conv
    Returns list of operator kinds (len C) per channel using your classifier.
    """
    C = w.shape[0]
    kinds = []
    for c in range(C):
        lbl = get_filter_label(w[c])           # your function
        kinds.append(LABEL_TO_KIND[lbl])       # map -> operator kind
    return kinds

# ---------------------------
# Main API
# ---------------------------
def load_and_constrain_convnext(
    model: nn.Module,  # ConvNeXt model to modify
    gamma7: float = 0.5,
    sigma_init: float = 1.5,
):
    """
    1) Load a ConvNeXt model.
    2) For each depthwise 7x7 conv, classify each channel's filter via get_filter_label.
    3) Replace it with DepthwiseGaussianConv2d using the inferred kinds.
    Returns the modified model (ready for training).
    """

    # --- Collect replacements first (don’t mutate during iteration)
    replacements = []  # list of (full_name, new_module)
    for full_name, module in model.named_modules():
        if _is_depthwise_7x7(module):
            w = module.weight.detach().cpu()
            kinds = _derive_kinds_from_weight(w)  # may raise if unknown label
            new = DepthwiseGaussianConv2d(
                in_channels=module.in_channels,
                kernel_size=7,
                stride=module.stride[0],
                padding=module.padding[0],
                dilation=module.dilation[0],
                bias=(module.bias is not None),
                filter_kinds=kinds,
                gamma7=gamma7,
                sigma_init=sigma_init,
                normalize_kernel=True,
            )
            # init bias & place on same device/dtype
            if module.bias is not None:
                with torch.no_grad():
                    new.bias.copy_(module.bias.data)
            new.to(module.weight.device, dtype=module.weight.dtype)
            replacements.append((full_name, new))

    # --- Apply replacements
    for name, new_mod in replacements:
        _set_module_by_name(model, name, new_mod)

    return model

