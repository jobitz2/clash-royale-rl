"""PPO update step (clipped surrogate, masked action distribution)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.997
    lam: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    epochs: int = 3
    minibatches: int = 8
    ent_coef: float = 0.012
    ent_coef_final: float = 0.002
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03          # early-stop the epoch loop past this KL
    norm_adv: bool = True


class PPO:
    def __init__(self, net: nn.Module, cfg: PPOConfig | None = None):
        self.net = net
        self.cfg = cfg or PPOConfig()
        self.opt = torch.optim.Adam(net.parameters(), lr=self.cfg.lr, eps=1e-5)

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def update(self, batch, ent_coef: float | None = None):
        cfg = self.cfg
        ent_coef = cfg.ent_coef if ent_coef is None else ent_coef
        self.net.train()

        n = batch["action"].shape[0]
        mb = max(1, n // cfg.minibatches)
        idx = torch.arange(n, device=batch["action"].device)

        # Normalise advantages over the whole batch, not per minibatch. Early in
        # training the only reward is small potential shaping, so a minibatch std
        # can be ~0; dividing by it amplifies noise into huge pseudo-advantages
        # and collapses the policy's entropy within a few updates. clamp_min caps
        # the amplification instead of letting 1/std explode.
        adv_all = batch["adv"]
        if cfg.norm_adv:
            adv_all = (adv_all - adv_all.mean()) / adv_all.std().clamp_min(1e-4)

        logs = {"pg_loss": [], "v_loss": [], "entropy": [], "kl": [],
                "clipfrac": [], "explained_var": []}
        stop = False

        for _ in range(cfg.epochs):
            perm = idx[torch.randperm(n, device=idx.device)]
            for s in range(0, n, mb):
                j = perm[s:s + mb]
                adv = adv_all[j]

                logp, ent, value = self.net.evaluate(
                    batch["spatial"][j], batch["scalar"][j],
                    batch["mask"][j], batch["action"][j])

                ratio = (logp - batch["logp"][j]).exp()
                pg = -torch.min(ratio * adv,
                                torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()

                # Clipped value loss keeps the critic from lurching on one batch.
                v_old = batch["value"][j]
                v_clip = v_old + (value - v_old).clamp(-cfg.value_clip, cfg.value_clip)
                v_loss = 0.5 * torch.max((value - batch["ret"][j]) ** 2,
                                         (v_clip - batch["ret"][j]) ** 2).mean()

                loss = pg + cfg.vf_coef * v_loss - ent_coef * ent.mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = (batch["logp"][j] - logp).mean()
                    logs["pg_loss"].append(float(pg))
                    logs["v_loss"].append(float(v_loss))
                    logs["entropy"].append(float(ent.mean()))
                    logs["kl"].append(float(kl))
                    logs["clipfrac"].append(
                        float(((ratio - 1).abs() > cfg.clip).float().mean()))
            if np.mean(logs["kl"][-cfg.minibatches:]) > cfg.target_kl:
                stop = True
            if stop:
                break

        with torch.no_grad():
            y, yhat = batch["ret"], batch["value"]
            var = y.var()
            ev = float(1 - (y - yhat).var() / var) if float(var) > 1e-8 else float("nan")
        out = {k: float(np.mean(v)) if v else float("nan") for k, v in logs.items()}
        out["explained_var"] = ev
        out["early_stop"] = float(stop)
        out["ent_coef"] = ent_coef
        return out
