import torch

from motionguard.config import load_config
from motionguard.models import MotionSTGCN


def test_stgcn_is_causal_and_learns_skeleton_edge_weights():
    cfg = dict(load_config(), channels=8, dilations=[1, 2])
    model = MotionSTGCN(288, cfg).eval()
    features = torch.randn(2, 8, 288)
    logits = model(features)
    assert logits.shape == (2, 8, 7)
    torch.testing.assert_close(model(features[:, :4]), logits[:, :4])
    logits.square().mean().backward()
    assert model.edge_logits.grad[model.edge_mask].abs().sum() > 0
    assert not model.edge_logits.grad[~model.edge_mask].any()
