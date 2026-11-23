import pytest
import numpy as np

torch=pytest.importorskip('torch')
from motionguard.models import MotionTCN
from motionguard.config import load_config

@pytest.mark.parametrize('rgb',[False,True])
def test_no_future_leakage(rgb):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg=load_config(); cfg.update(channels=8,dilations=[1,2],dropout=0.,use_rgb=rgb)
    model=MotionTCN(12,cfg).eval()
    x=torch.randn(2,20,12); altered=x.clone(); altered[:,10:]+=100
    crops=torch.randn(2,20,3,24,24) if rgb else None
    other=crops.clone() if rgb else None
    if rgb: other[:,10:]+=100
    with torch.no_grad():
        a=model(x,crops); b=model(altered,other)
    assert a.shape==(2,20,7)
    assert torch.allclose(a[:,:10],b[:,:10],atol=1e-6)
    loss=a.sum(); assert torch.isfinite(loss)


def test_tcn_backward():
    cfg=load_config(); cfg.update(channels=8,dilations=[1,2])
    m=MotionTCN(12,cfg)
    m(torch.randn(2,16,12)).square().mean().backward()
    assert m.head.weight.grad is not None
