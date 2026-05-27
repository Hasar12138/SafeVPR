"""Stage 2 verifier: DINOv2 patch-level mutual-nearest-neighbour (MNN) + Lowe-ratio score.

For a query window and a candidate window (each ``T_FRAMES`` frames):

  - Extract DINOv2 patch features (``PATCH_DIM``-d per patch, 256 patches per
    224x224 frame).
  - For each diagonal frame pair ``(i, i)``, compute a patch-MNN + Lowe-ratio
    match ratio: the fraction of query patches whose best candidate patch is a
    mutual nearest neighbour *and* passes the Lowe second-to-best ratio test.
  - The verification score ``s`` for the pair is the mean of the per-frame match
    ratios over the ``T_FRAMES`` diagonal frame pairs.

The score ``s`` lies in [0, 1] and is the only quantity passed downstream to the
conformal calibrator (Stage 3). The backbone retrieval (Stage 1) is treated as a
black box.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ── numerical core (must match the paper) ──
T_FRAMES = 10
RATIO_THRESH = 0.9
PATCH_DIM = 1024

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_dinov2(weights_path: Optional[str] = None, device=None) -> torch.nn.Module:
    """Load DINOv2 ViT-L/14.

    If ``weights_path`` is None the pretrained weights are fetched via
    ``torch.hub`` (``facebookresearch/dinov2``). Otherwise a local state dict is
    loaded into a non-pretrained backbone.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if weights_path is None:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14",
                               pretrained=True, source="github",
                               skip_validation=True)
    else:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14",
                               pretrained=False, source="github",
                               skip_validation=True)
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def extract_patch_features(model, image: "Image.Image", device=None) -> "torch.Tensor":
    """Extract DINOv2 patch tokens for one image. Returns a ``(256, PATCH_DIM)`` tensor."""
    if device is None:
        device = next(model.parameters()).device
    x = _TRANSFORM(image).unsqueeze(0).to(device)
    out = model.forward_features(x)
    return out["x_norm_patchtokens"][0]  # (256, PATCH_DIM)


def mnn_lowe_ratio(feat_q: "torch.Tensor", feat_c: "torch.Tensor",
                   ratio_thresh: float = RATIO_THRESH) -> float:
    """Patch-MNN + Lowe-ratio match ratio between two patch-feature sets.

    ``feat_q``: ``(Nq, D)``; ``feat_c``: ``(Nc, D)``. Both are L2-normalized
    internally. A query patch is kept iff (a) its Lowe ratio (second-best /
    best cosine) is below ``ratio_thresh`` and (b) it is a mutual nearest
    neighbour with its best candidate patch. Returns the kept fraction in [0, 1].
    """
    feat_q = F.normalize(feat_q.float(), dim=-1)
    feat_c = F.normalize(feat_c.float(), dim=-1)
    sim = feat_q @ feat_c.T
    top2 = sim.topk(2, dim=1)
    nn_q = top2.indices[:, 0]
    second_q = top2.values[:, 1]
    best_q = top2.values[:, 0]
    ratio_q = second_q / best_q.clamp(min=1e-9)
    pass_ratio = ratio_q < ratio_thresh
    nn_c_to_q = sim.argmax(dim=0)
    is_mnn = nn_c_to_q[nn_q] == torch.arange(feat_q.shape[0])
    keep = pass_ratio & is_mnn
    return float(keep.float().mean().item())


def verify_pair(model, q_frames: list, c_frames: list,
                t_frames: int = T_FRAMES, ratio_thresh: float = RATIO_THRESH,
                device=None) -> float:
    """Verification score for a (query window, candidate window) frame-pair list.

    ``q_frames`` / ``c_frames`` are lists of PIL images (length >= ``t_frames``).
    The score is the mean patch-MNN + Lowe ratio over the diagonal frame pairs
    ``(i, i)`` for ``i`` in ``range(t_frames)``. Returns ``nan`` if no frame pair
    could be scored.
    """
    per_frame = []
    n = min(t_frames, len(q_frames), len(c_frames))
    for i in range(n):
        fq = extract_patch_features(model, q_frames[i], device=device)
        fc = extract_patch_features(model, c_frames[i], device=device)
        per_frame.append(mnn_lowe_ratio(fq, fc, ratio_thresh=ratio_thresh))
    return float(np.mean(per_frame)) if per_frame else float("nan")
