"""V:N:M pack / unpack for 2D weight matrices (N is always 2 for Sparse Tensor Cores).

Conv weights must be reshaped first:
    W2d = w.permute(0, 2, 3, 1).reshape(C_out, kh * kw * C_in)
so C_in is innermost, matching im2col.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


def conv_to_2d(w: torch.Tensor) -> torch.Tensor:
    c_out, c_in, kh, kw = w.shape
    return w.permute(0, 2, 3, 1).reshape(c_out, kh * kw * c_in)


def conv_from_2d(w2d: torch.Tensor, c_out: int, c_in: int, kh: int, kw: int) -> torch.Tensor:
    return w2d.reshape(c_out, kh, kw, c_in).permute(0, 3, 1, 2).contiguous()


def _pad_2d(w: torch.Tensor, v: int, m: int) -> Tuple[torch.Tensor, int, int]:
    r, k = w.shape
    r_pad = int(math.ceil(r / v) * v)
    k_pad = int(math.ceil(k / m) * m)
    if r_pad == r and k_pad == k:
        return w, r, k
    out = w.new_zeros(r_pad, k_pad)
    out[:r, :k] = w
    return out, r, k


def _select_vnm(w_pad: torch.Tensor, v: int, m: int):
    """Shared V:N:M index selection. N is 2 values inside 4 columns of each V x M block."""
    r_pad, k_pad = w_pad.shape
    nb = k_pad // m
    ns = r_pad // v
    w_blocks = w_pad.view(ns, v, nb, m)
    mag = w_blocks.abs().sum(dim=1)  # (ns, nb, m)
    _, col_keep = torch.topk(mag, k=4, dim=-1)  # (ns, nb, 4)
    col_keep, _ = torch.sort(col_keep, dim=-1)
    col_exp = col_keep[:, None, :, :].expand(-1, v, -1, -1)
    slab = torch.gather(w_blocks, 3, col_exp)  # (ns, v, nb, 4)
    _, row_keep = torch.topk(slab.abs(), k=2, dim=-1)  # (ns, v, nb, 2)
    row_keep, _ = torch.sort(row_keep, dim=-1)
    chosen = torch.gather(col_exp, 3, row_keep)  # column inside the M-block
    values = torch.gather(slab, 3, row_keep)
    return col_keep, row_keep, chosen, values


def prune_vnm_2d(w: torch.Tensor, v: int, m: int, n: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Magnitude V:N:M prune. Returns (pruned dense matrix same shape as w, mask)."""
    if n != 2:
        raise ValueError("This packer stores N=2 (hardware 2:4 inside each block)")
    w_pad, r, k = _pad_2d(w.detach(), v, m)
    r_pad, k_pad = w_pad.shape
    nb = k_pad // m
    ns = r_pad // v
    pruned = torch.zeros_like(w_pad)
    mask = torch.zeros_like(w_pad, dtype=torch.bool)

    _, _, chosen, values = _select_vnm(w_pad, v, m)
    s_idx = torch.arange(ns, device=w_pad.device)[:, None, None, None]
    i_idx = torch.arange(v, device=w_pad.device)[None, :, None, None]
    b_idx = torch.arange(nb, device=w_pad.device)[None, None, :, None]
    rows = (s_idx * v + i_idx).expand_as(chosen)
    cols = (b_idx * m + chosen)
    pruned[rows.reshape(-1), cols.reshape(-1)] = values.reshape(-1)
    mask[rows.reshape(-1), cols.reshape(-1)] = True

    return pruned[:r, :k], mask[:r, :k]


def pack_vnm_2d(w: torch.Tensor, v: int, m: int, n: int = 2) -> Dict[str, torch.Tensor]:
    """Pack a (possibly already pruned) 2D matrix into VENOM-style arrays."""
    w_pad, r, k = _pad_2d(w.detach(), v, m)
    r_pad, k_pad = w_pad.shape
    nb = k_pad // m
    ns = r_pad // v

    col_keep, row_keep, _, values_blk = _select_vnm(w_pad, v, m)
    values = values_blk.permute(0, 1, 2, 3).reshape(r_pad, nb, 2).contiguous()
    m_idx = row_keep.reshape(r_pad, nb, 2).to(torch.int8).contiguous()
    col_loc = col_keep.to(torch.int16).contiguous()

    return {
        "values": values,
        "m_idx": m_idx,
        "col_loc": col_loc,
        "shape": torch.tensor([r, k, v, m, n], device=w_pad.device),
        "padded": torch.tensor([r_pad, k_pad], device=w_pad.device),
    }


def unpack_vnm_2d(packed: Dict[str, torch.Tensor]) -> torch.Tensor:
    values = packed["values"]
    m_idx = packed["m_idx"].long()
    col_loc = packed["col_loc"].long()
    r, k, v, m, _ = packed["shape"].tolist()
    r_pad, k_pad = packed["padded"].tolist()
    nb = k_pad // m
    W = values.new_zeros(r_pad, k_pad)
    rows = torch.arange(r_pad, device=values.device)[:, None, None]
    blocks = torch.arange(nb, device=values.device)[None, :, None]
    in_block = col_loc[rows // v, blocks, m_idx]
    cols = blocks * m + in_block
    W[rows.expand_as(cols), cols] = values
    return W[:r, :k]


def packed_nbytes(packed: Dict[str, torch.Tensor]) -> int:
    # values: fp16; m_idx: 2 bits each stored here as int8; col_loc: int16
    v = packed["values"]
    n_val = v.numel() * 2  # fp16
    n_idx = packed["m_idx"].numel()  # 1 byte each (could pack to 2 bits)
    n_col = packed["col_loc"].numel() * 2
    return n_val + n_idx + n_col


def packed_nbytes_tight(packed: Dict[str, torch.Tensor]) -> int:
    """Theoretical VENOM storage: fp16 values + 2-bit m_idx + uint8/uint16 col_loc."""
    values = packed["values"]
    m_idx = packed["m_idx"]
    col_loc = packed["col_loc"]
    n_val = values.numel() * 2
    n_idx = math.ceil(m_idx.numel() * 2 / 8)
    n_col = col_loc.numel()  # 0..M-1 fits in 1 byte for M<=256
    return n_val + n_idx + n_col
