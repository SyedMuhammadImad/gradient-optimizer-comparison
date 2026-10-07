import torch
import pytest
import models as m

@pytest.mark.parametrize('name',['SGD','SGDMomentum','RMSProp','Adam'])
def test_custom_trajectory_matches_torch_including_absent_gradients(name):
    torch.manual_seed(13)
    a=[torch.nn.Parameter(torch.randn(3,4,dtype=torch.float64)),torch.nn.Parameter(torch.randn(4,dtype=torch.float64))]
    b=[torch.nn.Parameter(p.detach().clone()) for p in a]
    settings={'lr':.013,'weight_decay':.02}
    if name=='SGD':ref=torch.optim.SGD(b,**settings)
    elif name=='SGDMomentum':ref=torch.optim.SGD(b,momentum=.9,**settings)
    elif name=='RMSProp':ref=torch.optim.RMSprop(b,alpha=.9,eps=1e-8,**settings)
    else:ref=torch.optim.Adam(b,betas=(.9,.999),eps=1e-8,**settings)
    custom=getattr(m,name)(a,**settings)
    for step in range(20):
        for i,(p,q) in enumerate(zip(a,b)):
            g=None if i==1 and step%3==0 else torch.randn_like(p)
            p.grad=g; q.grad=None if g is None else g.clone()
        custom.step();ref.step()
        for p,q in zip(a,b):torch.testing.assert_close(p,q,atol=1e-12,rtol=1e-11)
    custom.zero_grad()
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in a)

@pytest.mark.parametrize('cls',[m.SGD,m.SGDMomentum,m.RMSProp,m.Adam])
def test_invalid_optimizer_settings_and_atomic_nonfinite_rejection(cls):
    a=torch.nn.Parameter(torch.ones(2));b=torch.nn.Parameter(torch.ones(2))
    for kwargs in [{'lr':0},{'lr':float('nan')},{'weight_decay':-1}]:
        with pytest.raises(ValueError):cls([a],**kwargs)
    with pytest.raises(ValueError):cls([a,a])
    optimizer=cls([a,b]);a.grad=torch.ones(2);b.grad=torch.full((2,),float('nan'))
    with pytest.raises(ValueError):optimizer.step()
    torch.testing.assert_close(a,torch.ones(2))
    with pytest.raises(ValueError):m.Adam([a],beta1=1)
    with pytest.raises(ValueError):m.RMSProp([a],eps=0)
