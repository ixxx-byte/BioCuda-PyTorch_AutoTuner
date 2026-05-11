"""PyTorch API for BioCUDA AutoTune."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .core import FormulaEngine, GPUSpec, detect_gpu_spec


@dataclass(frozen=True)
class MatmulConfig:
    bm: int
    bn: int
    bk: int
    warps: int
    stages: int
    group: int
    score: float
    occupancy: float
    eta_partial: float

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class BioCUDAOptimizer:
    """Zero-shot optimizer driven by the BioCUDA formula engine."""

    def __init__(self, engine: FormulaEngine, verbose: bool = False):
        self.e = engine
        self.g = engine.g
        self.verbose = verbose

    def _compute_tau_eff(self, bm: int, bn: int, bk: int, nw: int, ns: int, m: int, n: int, k: int):
        tile_bytes = (bm * bk + bk * bn) * 2
        k_iters = max(1, math.ceil(k / bk))
        n_trans = self.e.g3_transactions(tile_bytes) * k_iters
        h_l2 = self.e.g24_h_l2_footprint(tile_bytes * ns, w_active=nw)
        tau_mem = self.e.g14_e_mem(n_trans * (1.0 - h_l2), n_trans * h_l2, (bm * bn) / 32.0, w_active=nw)
        t_crit = self.g.tau_tc + self.g.tau_shfl if ns >= 3 else self.g.tau_smem + self.g.tau_tc + self.g.tau_shfl
        r_last = m % bm if m % bm else bm
        c_last = n % bn if n % bn else bn
        eta_partial = self.e.g35_tc_partial(r_last, c_last)
        t_tc = self.g.tau_tc * math.ceil(bm / 16) * math.ceil(bn / 16) / max(eta_partial, 0.01) / max(nw, 1)
        return max(tau_mem, t_crit, t_tc)

    def select_matmul_configs(self, m: int, n: int, k: int, top_k: int = 6):
        candidates = []
        for bm in (32, 64, 128, 256):
            for bn in (32, 64, 128, 256):
                for bk in (32, 64):
                    for nw in (4, 8):
                        for ns in (3, 4):
                            threads = nw * 32
                            smem = (bm * bk + bk * bn) * 2 * ns
                            regs = max(32, (bm * bn) // threads + 16)
                            pruned, _ = self.e.g34_prune_hard(regs, smem, threads)
                            if pruned:
                                continue
                            r_last = m % bm if m % bm else bm
                            c_last = n % bn if n % bn else bn
                            eta_partial = self.e.g35_tc_partial(r_last, c_last)
                            if eta_partial < 0.10:
                                continue
                            occ = self.e.g16_occupancy(regs, smem, threads)
                            if occ <= 0:
                                continue
                            eta_exec = min(1.0, self.e.g15_hill_occupancy(occ) * self.e.g13_hill(occ, v=1.0, k=0.5, n=2.0))
                            tau_eff = self._compute_tau_eff(bm, bn, bk, nw, ns, m, n, k)
                            energy = self.e.g14_e_energy_norm(tile_bytes := (bm * bk + bk * bn) * 2, tile_bytes * 0.3, smem, 1e-4)
                            score = self.e.meta_score_cycles(tau_eff, eta_exec, e_energy_norm=energy)
                            candidates.append(MatmulConfig(bm, bn, bk, nw, ns, 8, score, occ, eta_partial))
        candidates.sort(key=lambda c: (c.score, -c.occupancy))
        if self.verbose:
            print(f"BioCUDA: selected {min(top_k, len(candidates))} / {len(candidates)} feasible matmul configs")
            for cfg in candidates[: min(4, top_k)]:
                print(f"  BM={cfg.bm} BN={cfg.bn} BK={cfg.bk} warps={cfg.warps} stages={cfg.stages} score={cfg.score:.2f}")
        return candidates[:top_k]

    def optimize_hamming(self, n_words: int):
        results = []
        for block in (128, 256, 512, 1024):
            occ = self.e.g16_occupancy(16, 0, block)
            q_g = self.e.g3_transactions(n_words * 8)
            q_l2 = q_g * min(1.0, self.g.l2_bytes / max(n_words * 8, 1))
            tau_mem = self.e.g14_e_mem(q_g - q_l2, q_l2, 0.0, w_active=block // 32)
            tau_eff = max(tau_mem, self.e.g18_critical_path([self.g.tau_hbm, 4, self.e.g4_latency_binary(), self.e.g7_scan_lb_cycles()]))
            eta = min(1.0, self.e.g15_hill_occupancy(occ) * self.e.g13_hill(occ, v=1.0, k=0.3, n=1.5))
            score = self.e.meta_score_cycles(tau_eff, eta)
            results.append({"block": block, "score": score, "occ": occ, "tau_eff": tau_eff, "eta_exec": eta})
        return min(results, key=lambda r: r["score"])

    def optimize_sw(self, la: int, lb: int):
        results = []
        for block in (32, 64, 128, 256):
            warps = math.ceil(block / 32)
            smem = (2 * (la + 1) + warps) * 4
            occ = self.e.g16_occupancy(24, smem, block)
            staging_ok = lb > 1.0 / max(self.e.g12_crossover(0.0), 1e-9)
            tau_eff = max(self.e.g14_e_mem(2.0, la / 32.0, la * lb / 32.0, warps), (la + lb - 1) * self.e.g5_antidiag_lb_cycles() / block)
            eta = min(1.0, self.e.g15_hill_occupancy(occ)) * (1.0 if staging_ok else 0.5)
            score = self.e.meta_score_cycles(tau_eff, eta)
            results.append({"block": block, "score": score, "occ": occ, "smem": smem, "staging_ok": staging_ok})
        return min(results, key=lambda r: r["score"])


class BioCUDAAutoTune:
    """PyTorch training optimizer plus BioCUDA kernel/config selection helpers."""

    def __init__(self, model: Optional[nn.Module] = None, device: str = "cuda", verbose: bool = True):
        self.model = model
        self.device = device
        self.verbose = verbose
        self.gpu: GPUSpec = detect_gpu_spec(device if torch.cuda.is_available() else None)
        self.engine = FormulaEngine(self.gpu)
        self.optimizer = BioCUDAOptimizer(self.engine, verbose=verbose)

    def optimize(self, optimizer_name: str = "AdamW", learning_rate: float = 1e-3, compile_mode: str = "default", use_amp: Optional[bool] = None, use_compile: Optional[bool] = None, use_fused: Optional[bool] = None) -> Tuple[nn.Module, Any, Optional[torch.amp.GradScaler]]:
        if self.model is None:
            raise ValueError("BioCUDAAutoTune.optimize() requires a torch.nn.Module")
        has_cuda = torch.cuda.is_available() and self.device.startswith("cuda")
        use_amp = has_cuda if use_amp is None else use_amp
        use_compile = hasattr(torch, "compile") if use_compile is None else use_compile
        use_fused = has_cuda and self.gpu.cc[0] >= 7 if use_fused is None else use_fused
        self.model = self.model.to(self.device)
        if use_compile:
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
            except Exception as exc:
                if self.verbose:
                    print(f"BioCUDA: torch.compile skipped: {exc}")
        opt_cls = {"adamw": optim.AdamW, "adam": optim.Adam, "sgd": optim.SGD}.get(optimizer_name.lower())
        if opt_cls is None:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        kwargs = {"lr": learning_rate}
        if optimizer_name.lower() == "sgd":
            kwargs["momentum"] = 0.9
        if use_fused and optimizer_name.lower() in {"adamw", "adam"}:
            kwargs["fused"] = True
        try:
            optimizer = opt_cls(self.model.parameters(), **kwargs)
        except TypeError:
            kwargs.pop("fused", None)
            optimizer = opt_cls(self.model.parameters(), **kwargs)
        scaler = torch.amp.GradScaler("cuda") if use_amp and has_cuda else None
        if has_cuda and self.gpu.cc[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if self.verbose:
            print(f"BioCUDA: {self.gpu.name} ({self.gpu.sm_arch}), AMP={bool(scaler)}, fused={use_fused}")
        return self.model, optimizer, scaler

    def matmul_configs(self, a: torch.Tensor, b: torch.Tensor, top_k: int = 6):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("matmul_configs expects rank-2 tensors")
        return self.optimizer.select_matmul_configs(a.shape[0], b.shape[1], a.shape[1], top_k=top_k)

    def summary(self) -> Dict[str, Any]:
        return {
            "gpu": self.gpu.__dict__,
            "g19_reuse_thresholds": self.engine.g19_reuse(),
            "g34_omega_size": self.engine.g34_omega_size(),
            "g6_roofline_crossover": self.engine.g6_roofline_crossover(),
        }


def quick_optimize(model: nn.Module, **kwargs):
    return BioCUDAAutoTune(model, verbose=kwargs.pop("verbose", True)).optimize(**kwargs)
