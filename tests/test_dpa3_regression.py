"""DPA3 regression integration uses a deterministic frozen-backbone stand-in.

Exercises real data loading, cache, PCA, training, resume and CSV inference paths.
A separate real DPA3 model test is still needed on the target DeePMD environment.
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from probe.backends.dpa3_regression import (RegressionDPA3Processor, read_frames,
    split_frames, geometry_digest)


class FakeExtractor:
    type_map = ['H', 'C']
    feat_dim = None
    def __init__(self):
        self.calls = 0
    def forward_structure(self, positions, numbers, cell=None, compute_force=True):
        assert torch.is_grad_enabled()  # force evaluation retains positional autograd
        self.calls += 1
        x = torch.tensor(positions, dtype=torch.float32)
        return dict(node_feats=torch.cat([x, torch.tensor(numbers)[:, None].float()], dim=1),
                    pred_energy=x.square().sum(), pred_forces=-2*x)


def frame(i=0, n=2):
    return dict(positions=np.arange(n*3, dtype=np.float64).reshape(n,3)/10+i,
                atomic_numbers=np.ones(n, dtype=np.int32), cell=None,
                true_energy=float(i+1), true_forces=np.ones((n,3)))


def test_partial_cache_hits_disk_restore_and_namespace(tmp_path):
    ext = FakeExtractor()
    proc = RegressionDPA3Processor(ext, 'backbone-a', tmp_path)
    a,b = frame(), frame(1,3)
    proc([a], 'cpu')
    out = proc([a,b], 'cpu')
    assert ext.calls == 2  # mixed-hit batch must compute only b
    assert out[0].shape == (2,3,4)
    assert out[1].sum().item() == 5
    ext2 = FakeExtractor()
    restored = RegressionDPA3Processor(ext2, 'backbone-a', tmp_path)
    changed_labels = dict(a, true_energy=42.)
    out = restored([changed_labels,b], 'cpu')
    assert ext2.calls == 0 and ext2.feat_dim == 4
    assert out[3][0] == 42.  # references come from current input, never cache
    other = RegressionDPA3Processor(FakeExtractor(), 'backbone-b', tmp_path)
    assert other.cache_dir != proc.cache_dir
    other([a], 'cpu')
    assert other.misses == 1
    assert geometry_digest(a) != geometry_digest(b)


def test_no_cache_and_bad_outputs():
    ext=FakeExtractor()
    p=RegressionDPA3Processor(ext, 'model', cache=False)
    p([frame()], 'cpu'); p([frame()], 'cpu')
    assert ext.calls == 2
    with pytest.raises(ValueError, match='Nonfinite DPA3'):
        p._validate(dict(node_feats=torch.full((2,4), float('nan')),
                         pred_energy=0., pred_forces=torch.zeros(2,3)), 2)


def write_data(path, n=8):
    # Reordered extra property tests ASE Properties parsing rather than column guesses.
    lines=[]
    for i in range(n):
        nat=2+i%2
        lines += [str(nat), 'Properties=species:S:1:pos:R:3:charge:R:1:forces:R:3 energy='+str(i+.2)]
        for j in range(nat):
            lines.append(f'H {i/10} {j/10} 0.2 99.0 0.1 0.2 0.3')
    path.write_text('\n'.join(lines)+'\n')


def test_io_labels_missing_and_pbc(tmp_path):
    path=tmp_path/'data.xyz'; write_data(path)
    fs=read_frames(path)
    assert fs[0]['true_energy'] == .2
    np.testing.assert_allclose(fs[0]['true_forces'][0], [.1,.2,.3])
    train,val=split_frames(fs,.25,42)
    assert len(train)==6 and len(val)==2
    assert not {geometry_digest(f) for f in train} & {geometry_digest(f) for f in val}
    path.write_text('1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n')
    f=read_frames(path)[0]
    assert np.isnan(f['true_energy']) and np.isnan(f['true_forces']).all()
    path.write_text('1\nLattice="5 0 0 0 5 0 0 0 5" pbc="F F F"\nH 0 0 0\n')
    assert read_frames(path)[0]['cell'] is None
    path.write_text('1\nLattice="5 0 0 0 5 0 0 0 5" pbc="T T F"\nH 0 0 0\n')
    with pytest.raises(ValueError, match='Partially periodic'):
        read_frames(path)


@pytest.mark.parametrize('dim', [None,2])
def test_train_resume_infer_and_provenance(monkeypatch, tmp_path, dim):
    import train_dpa3_regression as training
    import infer_dpa3_regression as inference
    from probe.pca import restore_model
    from probe.regression import ErrorRegressor
    data=tmp_path/'train.xyz'; write_data(data)
    backbone=tmp_path/'dpa.pt'; backbone.write_bytes(b'fake frozen backbone')
    ext=FakeExtractor()
    monkeypatch.setattr(training, 'load_backbone', lambda *a: ext)
    monkeypatch.setattr(inference, 'load_backbone', lambda *a: FakeExtractor())
    out=tmp_path/'out'; cache=tmp_path/'cache'
    argv=['train', '--model', str(backbone), '--train-xyz', str(data), '--device', 'cpu',
          '--output-dir', str(out), '--cache-dir', str(cache), '--batch-size', '2',
          '--valid-fraction', '.25', '--atom-encoder-hidden','8',
          '--atom-encoder-output-dim','8','--mol-attention-heads','2',
          '--classifier-hidden','8','--atom-force-head-hidden','8','--dropout','0']
    monkeypatch.setattr(sys,'argv', argv+['--epochs','1']+(['--pca-dim',str(dim)] if dim else []))
    training.main()
    before=torch.load(out/'last_checkpoint.pt',weights_only=False)
    assert ext.calls == 8  # training and validation each populate exactly once
    assert before['model_config']['backbone_dim'] == (dim or 4)
    def no_refit(*a,**k):
        raise AssertionError('PCA refitted during resume')
    monkeypatch.setattr(training,'make_pca',no_refit)
    fresh=FakeExtractor()
    monkeypatch.setattr(training,'load_backbone',lambda *a:fresh)
    monkeypatch.setattr(sys,'argv',argv+['--epochs','2','--resume',str(out/'last_checkpoint.pt')])
    training.main()
    after=torch.load(out/'last_checkpoint.pt',weights_only=False)
    assert fresh.calls==0 and after['epoch']==1
    if dim:
        train_frames,_=split_frames(read_frames(data),.25,42)
        expected=torch.cat([torch.cat([torch.tensor(f['positions']).float(),
                         torch.tensor(f['atomic_numbers'])[:,None].float()],dim=1) for f in train_frames]).mean(0)
        torch.testing.assert_close(before['model_state_dict']['pca.mean'], expected)
        for k in ('pca.mean','pca.components'):
            assert torch.equal(before['model_state_dict'][k], after['model_state_dict'][k])
    # Predictions match after an actual save/load, for every regression task.
    model=restore_model(ErrorRegressor,after,'cpu').eval()
    reloaded=restore_model(ErrorRegressor,torch.load(out/'last_checkpoint.pt',weights_only=False),'cpu').eval()
    x=torch.randn(1,3,4); mask=torch.tensor([[True,True,False]])
    for a,b in zip(model(x,mask),reloaded(x,mask)):
        torch.testing.assert_close(a,b)
    # Geometry-only inference still exports predicted errors and missing metrics.
    test=tmp_path/'test.xyz'; test.write_text('2\nProperties=species:S:1:pos:R:3\nH 0 0 0\nH 0 0 .5\n')
    infer_argv=['infer','--model',str(backbone),'--checkpoint',str(out/'best_regressor.pt'),
                '--test-xyz',str(test),'--device','cpu','--output-dir',str(tmp_path/'pred')]
    monkeypatch.setattr(sys,'argv',infer_argv)
    inference.main()
    import json,csv
    metrics=json.loads((tmp_path/'pred'/'metrics_regression.json').read_text())
    assert metrics['energy']['n']==0
    rows=list(csv.DictReader((tmp_path/'pred'/'predictions_structure_regression.csv').open()))
    assert len(rows)==1 and rows[0]['reference_energy_error_eV']==''
    data.write_text(data.read_text()+'\n')
    monkeypatch.setattr(sys,'argv',argv+['--epochs','3','--resume',str(out/'last_checkpoint.pt')])
    with pytest.raises(ValueError,match='same backbone, training data'):
        training.main()
    backbone.write_bytes(b'changed model')
    monkeypatch.setattr(sys,'argv',infer_argv)
    with pytest.raises(ValueError,match='same DPA3 model'):
        inference.main()


def test_deepmd_descriptor_adapter_preserves_type_map_and_atom_rows():
    from probe.backends.dpa3 import DPA3FeatureExtractor
    class Wrapper:
        def eval(self, coords, box, atypes):
            assert torch.is_grad_enabled()
            # In H,C,H input order, model type map is C,H.
            np.testing.assert_array_equal(atypes.reshape(-1), [1,0,1])
            assert box is None
            return np.array([[2.]]), np.arange(9).reshape(1,3,3), np.zeros((1,9))
        def eval_descriptor(self, coords, box, atypes):
            assert torch.is_grad_enabled()
            return np.arange(12).reshape(1,3,4)
    ext=DPA3FeatureExtractor(None, ['C','H'], infer_wrapper=Wrapper())
    fr=dict(positions=np.zeros((3,3)), atomic_numbers=np.array([1,6,1],dtype=np.int32),
            cell=None, true_energy=3., true_forces=np.ones((3,3)))
    proc=RegressionDPA3Processor(ext,'model')
    with torch.no_grad():
        x,mask,pe,te,pf,tf,n=proc([fr],'cpu')
    torch.testing.assert_close(x[0],torch.arange(12).reshape(3,4).float())
    torch.testing.assert_close(pf[0],torch.arange(9).reshape(3,3).float())
    assert pe.item()==2. and te.item()==3. and n.item()==3.


def test_prefit_pca_artifact_and_atom_cap(tmp_path):
    from probe.pca import make_pca
    artifact=tmp_path/'pca.pt'
    torch.save(dict(mean=torch.arange(4.), components=torch.eye(4)), artifact)
    projection,_=make_pca(None,None,'cpu',2,4,pca_model=artifact)
    torch.testing.assert_close(projection.components,torch.eye(4)[:2])
    x=torch.arange(24.).reshape(1,6,4)
    mask=torch.tensor([[True,True,True,True,False,False]])
    p,info=make_pca(lambda b,d:b,[(x,mask)],'cpu',2,4,max_atoms=3)
    assert info['n_training_atoms']==3
    torch.testing.assert_close(p.mean,x[0,:3].mean(0))
    with pytest.raises(ValueError, match='must exceed'):
        make_pca(None,None,'cpu',2,4,max_atoms=2)
