"""Command, observation and rewards for a jump with a commanded landing offset.

Everything is bolted onto jumper.ref_free_jump's own command (flight bookkeeping, landing, second-jump
clawback), so "a jump" means exactly what it means there.

* target_xy  [N,2] metres, in the take-off body frame; sampled per episode. Spawn yaw is 0 (reset to default / RSI), so
             this is also the world offset from the env origin.
* land_xy    [N,2] where the base was at the moment the episode's own jump touched down (latched).
* curriculum: the largest forward target grows linearly with the number of environment steps taken in THIS run
             (not common_step_counter, which a warm start restores to 288000).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from ..ref_free_jump.mdp.commands import RefFreeJumpCommand, RefFreeJumpCommandCfg


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(kw_only=True)
class DirJumpCommandCfg(RefFreeJumpCommandCfg):
    target_max_m: float = 0.30
    """Largest forward target once the curriculum is complete."""
    lateral_frac: float = 0.5
    """Lateral target is U(-1,1) * lateral_frac * the current forward maximum."""
    zero_frac: float = 0.2
    """Share of episodes with target (0,0): keeps the vertical jump alive."""
    ramp_start_step: int = 4800
    ramp_end_step: int = 72000
    """Environment steps (this run). Defaults = iteration 100 .. 1500 at 48 steps per iteration."""
    fixed_target: tuple[float, float] | None = None
    """If set (replay), every episode gets exactly this target and there is no curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.class_type = DirJumpCommand

    def build(self, env):
        return DirJumpCommand(self, env)


class DirJumpCommand(RefFreeJumpCommand):
    cfg: DirJumpCommandCfg

    def __init__(self, cfg, env):
        n = env.num_envs
        dev = env.device
        # state used by _resample_command, which the parent constructor may call
        self.target_xy = torch.zeros(n, 2, device=dev)
        self.land_xy = torch.zeros(n, 2, device=dev)
        self._prev_landed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._run_steps = 0
        super().__init__(cfg, env)
        self._err = torch.zeros(n, device=self.device)
        self._tgt_fwd = torch.zeros(n, device=self.device)
        self.metrics["landing_err_m"] = self._err
        self.metrics["target_fwd_m"] = self._tgt_fwd

    # -- curriculum ---------------------------------------------------------------------------------
    def current_max_fwd(self) -> float:
        c = self.cfg
        if c.fixed_target is not None:
            return float(c.fixed_target[0])
        frac = (self._run_steps - c.ramp_start_step) / max(c.ramp_end_step - c.ramp_start_step, 1)
        return c.target_max_m * min(max(frac, 0.0), 1.0)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        c = self.cfg
        k = len(env_ids)
        if c.fixed_target is not None:
            t = torch.tensor(c.fixed_target, device=self.device).expand(k, 2)
        else:
            m = self.current_max_fwd()
            fwd = torch.rand(k, device=self.device) * m
            lat = (torch.rand(k, device=self.device) * 2 - 1) * c.lateral_frac * m
            t = torch.stack([fwd, lat], dim=1)
            t = t * (torch.rand(k, 1, device=self.device) >= c.zero_frac).float()
        self.target_xy[env_ids] = t
        self.land_xy[env_ids] = 0.0
        self._prev_landed[env_ids] = False

    # -- bookkeeping --------------------------------------------------------------------------------
    def base_xy(self) -> torch.Tensor:
        return self.robot.data.root_link_pos_w[:, :2] - self._env.scene.env_origins[:, :2]

    def _update_command(self, env_ids=None) -> None:
        super()._update_command(env_ids)
        if env_ids is not None:
            return
        self._run_steps += 1
        newly = self.landed & ~self._prev_landed
        if bool(newly.any()):
            self.land_xy[newly] = self.base_xy()[newly]
        self._prev_landed.copy_(self.landed)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self._err[:] = torch.where(
            self.landed, (self.land_xy - self.target_xy).norm(dim=1), torch.zeros_like(self._err)
        )
        self._tgt_fwd[:] = self.target_xy[:, 0]
        self.metrics["landing_err_m"] = self._err
        self.metrics["target_fwd_m"] = self._tgt_fwd


# -- observation (actor AND critic) -------------------------------------------------------------------
def jump_target(env, command_name: str = "jump", scale: float = 0.10, mirror_kind: str | None = None):
    """(dx, dy, 0)/scale, the commanded landing offset. 3 wide so the symmetry augmentation can mirror it as a
    'vec3' (negate y)."""
    del mirror_kind
    cmd = env.command_manager.get_term(command_name)
    t = cmd.target_xy / scale
    return torch.cat([t, torch.zeros_like(t[:, :1])], dim=1)


# -- rewards ------------------------------------------------------------------------------------------
def landing_target(env, command_name: str = "jump", sigma: float = 0.10) -> torch.Tensor:
    """exp(-|land - target|^2 / sigma^2), paid every step after the episode's own jump has landed (not once a
    second jump has forfeited it). Latched at touchdown, so shuffling afterwards earns nothing."""
    cmd = env.command_manager.get_term(command_name)
    d2 = ((cmd.land_xy - cmd.target_xy) ** 2).sum(dim=1)
    return torch.exp(-d2 / sigma**2) * (cmd.landed & ~cmd.second_jump).float()


def flight_target(env, command_name: str = "jump", sigma: float = 0.25) -> torch.Tensor:
    """Dense half: the same kernel on the base's current position while the episode's own flight is under way."""
    cmd = env.command_manager.get_term(command_name)
    d2 = ((cmd.base_xy() - cmd.target_xy) ** 2).sum(dim=1)
    return torch.exp(-d2 / sigma**2) * cmd.in_flight.float()


__all__ = ["DirJumpCommand", "DirJumpCommandCfg", "flight_target", "jump_target", "landing_target"]
