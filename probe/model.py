"""
PROBE model architecture.

Contains:
  - MultiHeadSelfAttention
  - build_mlp
  - PROBEModel   (generic base — works with any flat [B, N, D] atom tensor)

Backend-specific wrappers (AIMNet2, MACE) are in probe_aimnet2.py and probe_mace.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 32, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                return_attention: bool = False):
        """
        Args:
            x:    [B, N, C]
            mask: [B, N] bool, True = valid atom
        Returns:
            out:  [B, N, C]
            attn: [B, num_heads, N, N]  (only if return_attention=True)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, N, N]

        if mask is not None:
            # mask=False means padding → fill with -inf
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        if return_attention:
            return out, attn_weights
        return out


def build_mlp(input_dim: int, hidden_dims: list, output_dim: int,
              dropout: float = 0.1, use_layernorm: bool = True,
              last_activation: bool = False, last_layernorm: bool = False) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        if use_layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    if last_layernorm:
        layers.append(nn.LayerNorm(output_dim))
    if last_activation:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class PROBEModel(nn.Module):
    """
    PROBE binary reliability classifier.

    Expects pre-encoded atom features as [B, N, backbone_dim] and a
    boolean atom mask [B, N].  The two backend-specific subclasses handle
    extracting these tensors from AIMNet2 / MACE outputs.

    Architecture:
        atom encoder MLP  →  [B, N, atom_enc_dim]
        (+ optional charge injection)
        multi-head self-attention  →  [B, N, atom_enc_dim]
        masked mean-pool + masked max-pool  →  [B, 2*atom_enc_dim]
        concat energy + N_atoms  →  [B, 2*atom_enc_dim + 2]
        linear projection  →  [B, 256]   (the molecular embedding)
        classifier MLP  →  [B, 2]        (logits: reliable / unreliable)
    """

    def __init__(
        self,
        backbone_dim: int,
        n_classes: int = 2,
        atom_encoder_hidden: list = [256, 128],
        atom_encoder_output_dim: int = 256,
        mol_attention_heads: int = 32,
        classifier_hidden: list = [256, 128, 32],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.atom_encoder_output_dim = atom_encoder_output_dim

        self.atom_encoder = build_mlp(
            backbone_dim, atom_encoder_hidden, atom_encoder_output_dim,
            dropout, use_layernorm=True, last_layernorm=True,
        )
        self.mol_attention = MultiHeadSelfAttention(
            atom_encoder_output_dim, mol_attention_heads, dropout
        )
        self.mol_attention_norm = nn.LayerNorm(atom_encoder_output_dim)

        # mean-pool (D) + max-pool (D) + energy (1) + n_atoms (1)
        pool_dim = atom_encoder_output_dim * 2 + 2
        self.proj = nn.Linear(pool_dim, 256)          # → molecular embedding

        self.classifier = build_mlp(
            256, classifier_hidden, n_classes, dropout,
            use_layernorm=True,
        )
        self._last_attention_weights = None

    def encode_atoms(self, atom_feats: torch.Tensor,
                     atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Atom encoder + self-attention.

        Args:
            atom_feats: [B, N, backbone_dim]
            atom_mask:  [B, N] bool
        Returns:
            attended: [B, N, atom_enc_dim]
        """
        z = self.atom_encoder(atom_feats)                        # [B, N, D]
        attended, attn_w = self.mol_attention(z, mask=atom_mask, return_attention=True)
        attended = self.mol_attention_norm(attended + z)
        self._last_attention_weights = attn_w.detach()
        return attended

    def pool_and_classify(self, attended: torch.Tensor,
                          atom_mask: torch.Tensor,
                          energy: torch.Tensor = None,
                          return_embeddings: bool = False):
        """
        Pooling → projection → classifier.

        Args:
            attended:  [B, N, D]
            atom_mask: [B, N] bool
            energy:    [B] predicted energy (optional)
        Returns:
            logits [B, n_classes]
            embedding [B, 256]  (only if return_embeddings=True)
        """
        mask_f = atom_mask.unsqueeze(-1).float()
        n_valid = atom_mask.sum(dim=1, keepdim=True).clamp(min=1).float()

        # mean-pool
        global_mean = (attended * mask_f).sum(dim=1) / n_valid   # [B, D]

        # max-pool (mask padding with -inf)
        tmp = attended.clone()
        tmp[~atom_mask.unsqueeze(-1).expand_as(tmp)] = float('-inf')
        global_max = tmp.max(dim=1)[0]
        global_max[global_max == float('-inf')] = 0.0              # [B, D]

        if energy is None:
            energy = torch.zeros(attended.size(0), device=attended.device)
        energy_f = energy.unsqueeze(-1)                            # [B, 1]
        n_atoms_f = n_valid                                        # [B, 1]

        pool = torch.cat([global_mean, global_max, energy_f, n_atoms_f], dim=-1)
        embedding = self.proj(pool)                                # [B, 256]
        logits = self.classifier(embedding)

        if return_embeddings:
            return logits, embedding
        return logits

    def forward(self, atom_feats: torch.Tensor, atom_mask: torch.Tensor,
                energy: torch.Tensor = None,
                return_attention: bool = False,
                return_embeddings: bool = False):
        """
        Args:
            atom_feats: [B, N, backbone_dim]
            atom_mask:  [B, N] bool  (True = real atom, False = padding)
            energy:     [B] MLIP predicted energy (eV)
        Returns:
            logits [B, 2]
        """
        attended = self.encode_atoms(atom_feats, atom_mask)
        out = self.pool_and_classify(attended, atom_mask, energy,
                                     return_embeddings=return_embeddings)
        if return_attention:
            if return_embeddings:
                logits, emb = out
                return logits, self._last_attention_weights, emb
            return out, self._last_attention_weights
        return out

    def get_atom_importance(self, atom_feats: torch.Tensor,
                            atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Per-atom importance = column sum of attention matrix (received attention),
        averaged over heads, normalized per molecule.

        Returns: [B, N] importance scores summing to 1 per molecule.
        """
        _, attn_w = self.forward(atom_feats, atom_mask, return_attention=True)
        # attn_w: [B, H, N, N] — sum over query dim (dim=2) = received attention
        importance = attn_w.mean(dim=1).sum(dim=1)   # [B, N]
        importance = importance * atom_mask.float()
        importance = importance / importance.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return importance

    def get_attention_weights(self) -> torch.Tensor:
        """Returns [B, H, N, N] attention weights from the last forward pass."""
        return self._last_attention_weights


def aggregate_atom_force_logits(logits_atom: torch.Tensor,
                                atom_mask: torch.Tensor) -> torch.Tensor:
    """Mean-aggregate per-atom force logits to structure-level logits.

    Takes the masked mean of P(unreliable) over atoms and converts back to
  2-class log-probabilities for cross-entropy.

    Args:
        logits_atom: [B, N, 2]
        atom_mask:   [B, N] bool
    Returns:
        logits_mol: [B, 2]
    """
    probs = F.softmax(logits_atom, dim=-1)
    p_unrel = probs[..., 1]
    mask_f = atom_mask.float()
    p_mol = (p_unrel * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
    p_mol = p_mol.clamp(1e-6, 1 - 1e-6)
    return torch.stack([torch.log(1 - p_mol), torch.log(p_mol)], dim=-1)


class MultitaskPROBEModel(PROBEModel):
    """
    Multitask PROBE: energy reliability (structure) + force reliability
    (per-atom + structure via mean aggregation of atom predictions).

    Architecture:
        shared atom encoder + self-attention
        ├─ energy branch:  mean/max pool + energy + N_atoms → energy classifier
        ├─ atom force head: per-atom MLP → [B, N, 2]
        └─ structure force: mean aggregate atom logits → [B, 2]  (no extra head)
    """

    def __init__(
        self,
        backbone_dim: int,
        n_classes: int = 2,
        atom_encoder_hidden: list = [256, 128],
        atom_encoder_output_dim: int = 256,
        mol_attention_heads: int = 32,
        classifier_hidden: list = [256, 128, 32],
        atom_force_head_hidden: list = [128, 32],
        dropout: float = 0.1,
    ):
        super().__init__(
            backbone_dim=backbone_dim,
            n_classes=n_classes,
            atom_encoder_hidden=atom_encoder_hidden,
            atom_encoder_output_dim=atom_encoder_output_dim,
            mol_attention_heads=mol_attention_heads,
            classifier_hidden=classifier_hidden,
            dropout=dropout,
        )
        self.energy_classifier = self.classifier
        self.atom_force_head = build_mlp(
            atom_encoder_output_dim, atom_force_head_hidden, n_classes,
            dropout, use_layernorm=True,
        )

    def forward(self, atom_feats: torch.Tensor, atom_mask: torch.Tensor,
                energy: torch.Tensor = None,
                return_attention: bool = False,
                return_embeddings: bool = False):
        """
        Returns:
            logits_energy:      [B, 2]
            logits_force_atom:  [B, N, 2]
            logits_force_mol:   [B, 2]  (mean-aggregated from atom logits)
        """
        attended = self.encode_atoms(atom_feats, atom_mask)
        logits_energy = self.pool_and_classify(attended, atom_mask, energy)
        logits_force_atom = self.atom_force_head(attended)
        logits_force_mol = aggregate_atom_force_logits(logits_force_atom, atom_mask)

        if return_attention:
            if return_embeddings:
                _, emb = self.pool_and_classify(
                    attended, atom_mask, energy, return_embeddings=True)
                return (logits_energy, logits_force_atom, logits_force_mol,
                        self._last_attention_weights, emb)
            return (logits_energy, logits_force_atom, logits_force_mol,
                    self._last_attention_weights)
        if return_embeddings:
            _, emb = self.pool_and_classify(
                attended, atom_mask, energy, return_embeddings=True)
            return logits_energy, logits_force_atom, logits_force_mol, emb
        return logits_energy, logits_force_atom, logits_force_mol
