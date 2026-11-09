import torch
from probe.pca import PCAProjection, ModelWithPCA, fit_streaming_pca, restore_model


def test_projection_and_padding():
    projection = PCAProjection([1., 2., 3.], [[1., 0., 0.], [0., 1., 0.]])
    x = torch.tensor([[[3., 5., 8.], [0., 0., 0.]]])
    mask = torch.tensor([[True, False]])
    assert torch.allclose(projection(x, mask), torch.tensor([[[2., 3.], [0., 0.]]]))


def test_checkpoint_reconstructs_same_coordinates():
    from probe.model import PROBEModel
    p = PCAProjection([1., 2., 3.], [[1., 0., 0.], [0., 1., 0.]])
    probe = PROBEModel(backbone_dim=2, atom_encoder_output_dim=32,
                       mol_attention_heads=4, dropout=0.0)
    model = ModelWithPCA(probe, p).eval()
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    energy = torch.randn(2)
    y = model(x, mask, energy=energy)
    cfg = dict(backbone_dim=2, atom_encoder_output_dim=32, mol_attention_heads=4,
               dropout=0.0)
    checkpoint = dict(model_config=cfg, pca_dim=2, model_state_dict=model.state_dict())
    restored = restore_model(PROBEModel, checkpoint, 'cpu').eval()
    assert torch.allclose(y, restored(x, mask, energy=energy), atol=1e-6)


def test_fit_train_data_only():
    batch = (torch.tensor([[[1., 2., 3.], [2., 3., 4.], [3., 4., 5.], [4., 5., 6.]]]),
             torch.tensor([[True, True, True, True]]))
    def processor(b, device):
        return b[0], b[1], None
    projection, info = fit_streaming_pca(processor, [batch], 'cpu', 2, 3)
    assert info['n_training_atoms'] == 4
    assert torch.allclose(projection.mean, torch.tensor([2.5, 3.5, 4.5]))


def test_disabled_pca_preserves_model_checkpoint():
    from probe.model import PROBEModel
    from probe.pca import make_pca
    projection, info = make_pca(None, None, 'cpu', None, 3)
    assert projection is None and info is None
    cfg = dict(backbone_dim=3, atom_encoder_output_dim=32, mol_attention_heads=4, dropout=0.)
    model = PROBEModel(**cfg).eval()
    restored = restore_model(PROBEModel, dict(model_config=cfg, pca_dim=None,
                             model_state_dict=model.state_dict()), 'cpu').eval()
    x, mask = torch.randn(2, 4, 3), torch.ones(2, 4, dtype=torch.bool)
    assert torch.allclose(model(x, mask), restored(x, mask))


def test_invalid_pca_options():
    import pytest
    from probe.pca import make_pca
    for dim in (0, -1, 4):
        with pytest.raises(ValueError):
            make_pca(None, None, 'cpu', dim, 3)
