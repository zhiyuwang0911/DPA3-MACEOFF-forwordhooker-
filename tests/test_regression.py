import io
import torch
import pytest
from probe.regression import ErrorRegressor, error_targets, regression_loss, regression_metrics, run_epoch


def small_model():
    return ErrorRegressor(4, atom_encoder_hidden=[8], atom_encoder_output_dim=8,
        mol_attention_heads=2, classifier_hidden=[8], atom_force_head_hidden=[8], dropout=0.)


def test_targets_and_padding():
    mask = torch.tensor([[True, True, False]])
    pf = torch.tensor([[[1., -2., 3.], [3., 3., 3.], [float('nan')]*3]])
    e, fa, fm = error_targets(torch.tensor([4.]), torch.tensor([1.]), pf, torch.zeros_like(pf), mask)
    torch.testing.assert_close(e, torch.tensor([3.]))
    torch.testing.assert_close(fa, torch.tensor([[2., 3., 0.]]))
    torch.testing.assert_close(fm, torch.tensor([2.5]))


def test_nonnegative_outputs_padding_invariance_and_checkpoint():
    torch.manual_seed(1)
    model = small_model().eval()
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    e, fa, fm = model(x, mask)
    assert e.shape == (2,) and fa.shape == (2, 3) and fm.shape == (2,)
    assert (e >= 0).all() and (fa >= 0).all()
    torch.testing.assert_close(fm, fa.sum(1) / mask.sum(1))
    padded = torch.cat([x, torch.randn(2, 2, 4)], 1)
    mask2 = torch.cat([mask, torch.zeros(2, 2, dtype=torch.bool)], 1)
    e2, fa2, fm2 = model(padded, mask2)
    torch.testing.assert_close(e, e2)
    torch.testing.assert_close(fa, fa2[:, :3])
    torch.testing.assert_close(fm, fm2)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = small_model().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(x, mask)[0], e)


def test_missing_labels_and_gradients():
    pred = (torch.tensor([2.], requires_grad=True),
            torch.tensor([[1., 9.]], requires_grad=True), torch.tensor([1.], requires_grad=True))
    target = (torch.tensor([float('nan')]), torch.tensor([[3., float('nan')]]), torch.tensor([float('nan')]))
    loss = regression_loss(pred, target, torch.tensor([[True, False]]))
    assert loss.item() == 4.
    loss.backward()
    torch.testing.assert_close(pred[1].grad, torch.tensor([[-4., 0.]]))


def test_metrics():
    result = regression_metrics(torch.tensor([1., 3., 5.]), torch.tensor([1., 2., float('nan')]))
    assert result['n'] == 2 and result['mae'] == .5
    assert result['rmse'] == pytest.approx(2 ** -.5)
    assert result['r2'] == -1.
    assert regression_metrics(torch.ones(2), torch.ones(2))['r2'] is None


def test_training_step_and_unlabeled_failure():
    model = small_model()
    mask = torch.tensor([[True, True, False], [True, False, False]])
    batch = (torch.randn(2, 3, 4), mask, torch.zeros(2), torch.ones(2),
             torch.zeros(2, 3, 3), torch.ones(2, 3, 3), mask.sum(1))
    process = lambda _, device: batch
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    before = model.atom_force_head[-1].weight.detach().clone()
    loss, metrics = run_epoch(model, process, [None], 'cpu', [1., 1., 1.], optimizer)
    assert torch.isfinite(torch.tensor(loss))
    assert metrics['force_atom']['n'] == 3
    assert not torch.equal(before, model.atom_force_head[-1].weight)
    unlabeled = list(batch)
    unlabeled[3] = torch.full((2,), float('nan'))
    unlabeled[5] = torch.full((2, 3, 3), float('nan'))
    with pytest.raises(ValueError, match='No batches'):
        run_epoch(model, lambda *_: unlabeled, [None], 'cpu', [1., 1., 1.])


def test_unlabeled_xyz(tmp_path):
    from probe.io_extxyz import load_probe_extxyz
    path = tmp_path / 'unlabeled.xyz'
    path.write_text('1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n')
    frame = load_probe_extxyz(path)[0]
    assert frame['true_energy'] is None
    assert __import__('numpy').isnan(frame['true_forces']).all()
