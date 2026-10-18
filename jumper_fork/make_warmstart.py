#!/usr/bin/env python3
"""Turn ref_free_jump's trained checkpoint into a starting point for ref_free_jump_dir.

The new task has the same observations plus `jump_target` (3 wide) as the LAST term of both groups, so the first
Linear layer of actor (406 -> 409) and critic (435 -> 438) gets 3 zero-initialised input columns, and the normalisers
get mean 0 / var 1 / std 1 for them. With zero weights the padded policy is EXACTLY the old one, whatever the target.
The optimiser's moment estimates are dropped (shapes changed) and the iteration counter restarts; the environment
step counter is kept, so the prior schedule stays finished (feedforward 0).

    python make_warmstart.py IN_model_5999.pt OUT_warm.pt
"""
import sys
import torch

N_NEW = 3
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=True)

def pad(sd, in_dim):
    w = sd["mlp.0.weight"]
    assert w.shape[1] == in_dim, (w.shape, in_dim)
    sd["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], N_NEW)], dim=1)
    for k, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        v = sd["obs_normalizer." + k]
        sd["obs_normalizer." + k] = torch.cat([v, torch.full((1, N_NEW), fill)], dim=1)

pad(ck["actor_state_dict"], 406)
pad(ck["critic_state_dict"], 435)
opt = ck["optimizer_state_dict"]
opt["state"] = {}                       # Adam re-initialises lazily
for g in opt["param_groups"]:
    g["lr"] = 3e-4                      # the old run had annealed to 1e-5; the adaptive schedule takes over from here
ck["iter"] = 0
torch.save(ck, dst)
print(f"wrote {dst}: actor 409 inputs, critic 438 inputs, env_state {ck['infos']['env_state']}")
