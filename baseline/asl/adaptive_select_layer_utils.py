import torch
import torch.nn.functional as F
import itertools
import torch
from typing import List, Optional, Tuple, Sequence
from flash_attn.flash_attn_interface import flash_attn_varlen_func

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def customized_flash_attn(
    q: torch.Tensor,                          # [bs, head_num, seqlen_q, dim]
    k_list: Sequence[torch.Tensor],           # len = kv_head_num; each [bs, seqlen_k_i, dim]
    v_list: Sequence[torch.Tensor],           # same as k_list
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Grouped-query attention with per-group variable key/value lengths.

    Returns:
        out: [bs, head_num, seqlen_q, dim]
        (or, if return_attn_probs=True, also the usual attn-probs/softmax-lse)
    """
    bs, head_num, seqlen_q, dim = q.shape
    kv_head_num = len(k_list)
    assert head_num % kv_head_num == 0, "head_num must be divisible by kv_head_num"
    heads_per_kv = head_num // kv_head_num

    device, dtype = q.device, q.dtype

    #
    # 1) reshape & group queries into (kv_head_num*bs) ragged-batch samples
    #
    #   [bs, head_num, seqlen_q, dim]
    # -> [bs, kv_head_num, heads_per_kv, seqlen_q, dim]
    # -> permute to [kv_head_num, bs, seqlen_q, heads_per_kv, dim]
    # -> flatten to [kv_head_num*bs*seqlen_q, heads_per_kv, dim]
    #
    q_grouped = q.view(bs, kv_head_num, heads_per_kv, seqlen_q, dim)
    q_grouped = q_grouped.permute(1, 0, 3, 2, 4)  # [kv_head, bs, seqlen_q, heads_per_kv, dim]
    total_q = kv_head_num * bs * seqlen_q
    q_flat = q_grouped.reshape(total_q, heads_per_kv, dim)

    # cumulative lengths for queries: each “sample” has seqlen_q
    cu_seqlens_q = torch.arange(
        0, total_q + 1, seqlen_q, dtype=torch.int32, device=device
    )

    #
    # 2) flatten & concatenate all K/V groups into one ragged batch
    #
    # For each i in [0..kv_head_num):
    #   k_list[i]: [bs, seqlen_k_i, dim]  →  reshape to [bs * seqlen_k_i, dim]
    #   v_list[i]: same
    #
    seqlens_k = []
    k_parts, v_parts = [], []
    for k_i, v_i in zip(k_list, v_list):
        bs_i, sk_i, _ = k_i.shape
        assert bs_i == bs, "batch dims must match"
        seqlens_k.extend([sk_i] * bs)
        k_parts.append(k_i.reshape(bs * sk_i, dim))
        v_parts.append(v_i.reshape(bs * sk_i, dim))

    total_k = sum(seqlens_k)
    # stack all groups in time, then unsqueeze head-dim=1
    k_flat = torch.cat(k_parts, dim=0).unsqueeze(1)  # [total_k, 1, dim]
    v_flat = torch.cat(v_parts, dim=0).unsqueeze(1)  # [total_k, 1, dim]

    # cumulative lengths for keys/values
    cu_seqlens_k = torch.tensor(
        [0] + list(itertools.accumulate(seqlens_k)),
        dtype=torch.int32,
        device=device
    )

    #
    # 3) call FlashAttention varlen
    #
    out_flat = flash_attn_varlen_func(
        q_flat,            # [total_q, heads_per_kv, dim]
        k_flat,            # [total_k,     1,        dim]
        v_flat,            # [total_k,     1,        dim]
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=max(seqlens_k),
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        block_table=block_table,
    )

    #
    # 4) reshape back to [bs, head_num, seqlen_q, dim]
    #
    # If return_attn_probs=False, out_flat is just the output tensor:
    #     [total_q, heads_per_kv, dim]
    # Otherwise it’s a tuple (out_flat, …) – adjust similarly.
    if return_attn_probs:
        out_flat, lse, attn = out_flat  # unpack testing-only outputs
    # now out_flat: [total_q, heads_per_kv, dim]
    # reshape to [kv_head_num, bs, seqlen_q, heads_per_kv, dim]
    out_grouped = out_flat.view(kv_head_num, bs, seqlen_q, heads_per_kv, dim)
    # permute back to [bs, kv_head_num, heads_per_kv, seqlen_q, dim]
    out_grouped = out_grouped.permute(1, 0, 3, 2, 4)
    # final flatten of head groups → [bs, head_num, seqlen_q, dim]
    out = out_grouped.reshape(bs, head_num, seqlen_q, dim)

    if return_attn_probs:
        return out, lse, attn
    else:
        return out
    
def relative_normalized_variance(X: torch.Tensor,initial_variance = None) -> float:
    """
    X: Tensor of shape [l, k], each element in [0, N]
    N: Maximum possible value in X
    Returns:
        normalized variance score in [0, 1]
    """
    # var_per_dim = torch.var(X.float(), dim=0, unbiased=False)  # shape: [k]

    #add bias
    # bias = torch.linspace(0.5,1,steps=X.shape[0]).to(X) #old -> new　という順番
    # # bias = torch.exp(-1*bias)
    # bias = bias.view(-1,1).repeat(1,X.shape[-1])
    # X = X*bias

    mu = torch.mean(X, dim=0)
    var_per_dim = torch.mean((X - mu)**2)
    # normalize by using N^2 / 4 
    # max_var = (N ** 2) / 4
    # normalized_var_per_dim = var_per_dim / max_var
    if initial_variance==None:
        initial_variance = var_per_dim.mean().item()
        mean_normalized_variance = 1
    else:
        mean_normalized_variance  = var_per_dim.mean().item() /initial_variance

    return mean_normalized_variance,initial_variance



def get_rank_descending(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)  # shape [1, N]
        squeeze = True
    else:
        squeeze = False

    bs, N = x.shape

    sorted_indices = torch.argsort(x, dim=1, descending=True) #[bs, N]
    rank_values = torch.arange(1, N + 1, device=x.device).expand(bs, -1)
    ranks = torch.empty_like(sorted_indices)
    ranks.scatter_(dim=1, index=sorted_indices, src=rank_values)

    if squeeze:
        return ranks.squeeze(0)
    else:
        return ranks
    
def floor_multiple_torch(n, m):
    return m * (n // m)


def get_top_p_indices(
    probs: torch.Tensor,
    top_p: float,
    _is_softmaxed: bool = False
):
    # softmax
    if not _is_softmaxed:
        probs = F.softmax(probs.float(), dim=-1)
    # floor multiple function
    def _floor_mult(x, m=8):
        return (x // m) * m

    if probs.dim() == 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumprobs = torch.cumsum(sorted_probs, dim=0)
        mask = cumprobs >= top_p
        has = mask.any()
        first = mask.float().argmax().item()  # 0-based
        raw_k = first + 1 if has else probs.size(0)
        k = int(_floor_mult(raw_k, 8))
        sel = sorted_idx[:k]
        return torch.sort(sel)[0]

    B, V = probs.shape
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=1)  # (B, V)
    cumprobs = torch.cumsum(sorted_probs, dim=1)                           # (B, V)

    mask = cumprobs >= top_p        # (B, V), bool
    has_any = mask.any(dim=1)       # (B,), bool
    first_pos = mask.float().argmax(dim=1)          # (B,),0 if True is not included
    raw_k = torch.where(
        has_any,
        first_pos + 1,
        torch.full_like(first_pos, V)
    )
    k_per_batch = _floor_mult(raw_k, 8) 
    M = int(k_per_batch.max().item())

    topM_idx = sorted_idx[:, :M]
    arange = torch.arange(M, device=probs.device).unsqueeze(0)  # (1, M)
    valid_mask = arange < k_per_batch.unsqueeze(1)             # (B, M)
    
    sentinel = V
    padded = torch.full((B, M), sentinel, device=probs.device, dtype=torch.long)
    padded[valid_mask] = topM_idx[valid_mask]

    sel_sorted, _ = torch.sort(padded, dim=1)      # sort, (B, M)
    sel_sorted[sel_sorted == sentinel] = -1

    return sel_sorted, k_per_batch




    