"""kev.benchmark with each record's question branches run in chunks off one read of its state (evaluation only).

Kev's shared-prefix pass (kev.shared_prefix._forward) reads a record's state once, but for every question branch it
gathers a full copy of the state's attention cache, and it builds one boolean mask over all branches (branches x branch
length x (state + branch length)). At 256k tokens with ~980 questions that is ~38 GB of mask plus ~1 GB of keys and
values per question per attention layer (4 GB once the fp32 kernel repeats them across query heads), so Kev's scorer
can only take such a document as many small records, each re-reading the whole state.

This wrapper patches `_forward` for no-grad calls only: the state pass runs once per layer exactly as in Kev, then the
branches run in chunks of KEV_BRANCH_TOKENS // state length rows (default 1M tokens: 4 rows at 256k, 16 at 64k), each
chunk gathering its own copy of the cache and building its own mask. Branches are independent given the state (a
branch attends to its record's state and to itself), so this computes the same thing in less memory. Checked against
Kev's unpatched scorer on 64k documents (bench/chunked_check.py). Training never calls this path.

    python bench/chunked_scorer.py --run <checkpoint> --suite <suite dir> --out <dir> --device cuda
"""
import os
import runpy
import sys

import torch
from kev import shared_prefix as SP

BUDGET = int(os.environ.get("KEV_BRANCH_TOKENS", str(1 << 20)))
_original = SP._forward


def _forward_chunked(base, s_ids, s_pos, s_real, b_ids, b_pos, b_real, owner):
    if torch.is_grad_enabled() or base.training:
        return _original(base, s_ids, s_pos, s_real, b_ids, b_pos, b_real, owner)
    device, (Ls, Lb) = s_ids.device, (s_ids.shape[1], b_ids.shape[1])
    h_s = base.embed_tokens(s_ids) * s_real[..., None]
    h_b = base.embed_tokens(b_ids) * b_real[..., None]
    dtype, attn = h_s.dtype, base.config._attn_implementation
    causal = lambda n: torch.ones(n, n, dtype=torch.bool, device=device).tril()
    eye = lambda n: torch.eye(n, dtype=torch.bool, device=device)
    plain = attn == "sdpa" and bool(s_real.all())   # as in Kev: an unpadded state runs causal with no explicit mask
    masks_s = {"full_attention": None if plain else SP._masks((causal(Ls)[None] & s_real[:, None, :]) | eye(Ls)[None], dtype, attn),
               "linear_attention": s_real.to(dtype)}
    rope_s, rope_b = base.rotary_emb(h_s, s_pos[None].expand(3, -1, -1)), base.rotary_emb(h_b, b_pos[None].expand(3, -1, -1))
    bc, be = causal(Lb)[None], eye(Lb)[None]
    rows = max(1, BUDGET // (Ls + Lb))
    chunks = [slice(k, k + rows) for k in range(0, h_b.shape[0], rows)]

    def branch_masks(sl):   # Kev's allow_b, for this chunk's rows only
        allow = torch.cat([s_real[owner[sl]][:, None, :].expand(-1, Lb, -1), (bc & b_real[sl][:, None, :]) | be], -1)
        return {"full_attention": SP._masks(allow, dtype, attn), "linear_attention": b_real[sl].to(dtype)}

    for layer in base.layers:
        prefix = SP.Prefix()
        h_s = layer.forward(h_s, position_embeddings=rope_s, attention_mask=masks_s[layer.block_type], position_ids=s_pos, past_key_values=prefix)
        out = []
        for sl in chunks:
            p = SP.Prefix()   # this chunk's view of what the state pass left; `branch` gathers it per row
            p.keys, p.values, p.conv, p.recurrent_states = prefix.keys, prefix.values, prefix.conv, dict(prefix.recurrent_states)
            p.branch(owner[sl])
            out.append(layer.forward(h_b[sl], position_embeddings=(rope_b[0][sl], rope_b[1][sl]), attention_mask=branch_masks(sl)[layer.block_type],
                                     position_ids=b_pos[sl], past_key_values=p))
        h_s, h_b = h_s * s_real[..., None], torch.cat(out, 0)
    return base.norm(h_b).float()


SP._forward = _forward_chunked

if __name__ == "__main__":
    sys.argv = ["kev.benchmark"] + sys.argv[1:]
    runpy.run_module("kev.benchmark", run_name="__main__", alter_sys=True)
