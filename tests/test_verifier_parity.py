import numpy as np
import torch

from safevpr.verifier import mnn_lowe_ratio


def test_mnn_lowe_ratio_matches_reference():
    # Deterministic synthetic patch features: 256 patches x 1024 dim
    rng = np.random.default_rng(0)
    q = torch.tensor(rng.standard_normal((256, 1024)), dtype=torch.float32)
    c = q.clone()                      # identical -> every patch is its own MNN
    assert mnn_lowe_ratio(q, c, ratio_thresh=0.9) == 1.0
    c2 = torch.tensor(rng.standard_normal((256, 1024)), dtype=torch.float32)
    r = mnn_lowe_ratio(q, c2, ratio_thresh=0.9)
    assert 0.0 <= r <= 1.0             # random -> bounded score
