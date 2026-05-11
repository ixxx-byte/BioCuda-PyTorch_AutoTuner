"""Core BioCUDA formula engine used by the PyTorch-facing autotuner.

This module is intentionally import-safe: it does not launch notebooks,
benchmarks, CUDA compilation, or long sweeps at import time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch


@dataclass(frozen=True)
class GPUSpec:
    key: str
    name: str
    arch: str
    sm_arch: str
    cc: Tuple[int, int]
    n_sm: int
    tc_per_sm: int
    smem_per_sm: int
    l2_bytes: int
    hbm_bw: float
    boost_ghz: float
    tdp_w: int
    tau_shfl: int = 4
    tau_smem: int = 23
    tau_l2: int = 193
    tau_hbm: int = 600
    tau_tc: int = 16
    tau_dpx: int = 2
    r_max: int = 65536
    g_reg: int = 256
    w_max: int = 64
    b_sm_max: int = 32
    t_sm_max: int = 2048
    w_warp: int = 32


GPU_DB = {
    "T4": GPUSpec("T4", "Tesla T4", "turing", "sm_75", (7, 5), 40, 8, 65536, 4 * 1024**2, 320e9, 1.59, 70, tau_smem=28, tau_l2=220, tau_hbm=650),
    "V100": GPUSpec("V100", "Tesla V100", "volta", "sm_70", (7, 0), 80, 8, 98304, 6 * 1024**2, 900e9, 1.53, 300, tau_smem=28, tau_l2=230, tau_hbm=670),
    "A100": GPUSpec("A100", "A100", "ampere", "sm_80", (8, 0), 108, 4, 167936, 40 * 1024**2, 2039e9, 1.41, 400, tau_smem=23, tau_l2=200, tau_hbm=620),
    "RTX4090": GPUSpec("RTX4090", "GeForce RTX 4090", "ada", "sm_89", (8, 9), 128, 4, 102400, 72 * 1024**2, 1008e9, 2.52, 450, tau_smem=24, tau_l2=195, tau_hbm=610, tau_tc=14),
    "H100": GPUSpec("H100", "H100", "hopper", "sm_90a", (9, 0), 132, 4, 233472, 50 * 1024**2, 3350e9, 1.83, 700, tau_smem=23, tau_l2=193, tau_hbm=600),
}


def detect_gpu_spec(device: Optional[Union[torch.device, str]] = None) -> GPUSpec:
    if not torch.cuda.is_available():
        return GPU_DB["T4"]
    dev = torch.device(device or "cuda")
    name = torch.cuda.get_device_name(dev).lower()
    cc = torch.cuda.get_device_capability(dev)
    for key, spec in GPU_DB.items():
        if key.lower() in name or spec.cc == cc:
            return spec
    major, minor = cc
    return GPUSpec("CUDA", torch.cuda.get_device_name(dev), "unknown", f"sm_{major}{minor}", cc, torch.cuda.get_device_properties(dev).multi_processor_count, 4, 65536, 4 * 1024**2, 600e9, 1.5, 250)


class FormulaEngine:
    """Subset of BioCUDA v39 formulas needed for PyTorch autotuning."""

    def __init__(self, gpu: GPUSpec):
        self.g = gpu

    def g3_transactions(self, n_bytes: float, l_seg: int = 128) -> int:
        return int(math.ceil(max(n_bytes, 0.0) / l_seg))

    def g4_latency_binary(self) -> int:
        return 8

    def g5_antidiag_lb_cycles(self) -> int:
        return self.g.tau_shfl + self.g.tau_dpx + 4

    def g6_roofline_crossover(self) -> float:
        # 512 FMA/cycle/SM, each FMA is 2 FLOP.
        phi_tc = self.g.tc_per_sm * 512 * 2 * self.g.n_sm * self.g.boost_ghz * 1e9
        return phi_tc / self.g.hbm_bw

    def g7_scan_lb_cycles(self) -> int:
        return 5 * self.g.tau_shfl

    def g8_hmm_flops(self, t: int, s: int) -> float:
        return 2.0 * t * s * s

    def g8_arithmetic_intensity(self, n_states: int, batch: int) -> float:
        return float(n_states) if batch >= max(1, n_states // 4) else float(batch)

    def g9_pi_hw(self, cycles_active) -> list:
        total = float(sum(cycles_active))
        return [float(c) / total if total > 0 else 0.0 for c in cycles_active]

    def g9_psi_hw(self, cycles_active) -> list:
        return [-math.log(max(p, 1e-12)) for p in self.g9_pi_hw(cycles_active)]

    def g9_theta(self, w_elig: float) -> float:
        return (float(w_elig) - 4.0) / 4.0

    def g10_pi_boltz(self, e_mem, theta: float) -> list:
        vals = [math.exp(-float(e) / max(theta, 1e-9)) for e in e_mem]
        z = sum(vals)
        return [v / z if z > 0 else 0.0 for v in vals]

    def g10_mi_decision(self, i_mut_bits: float) -> str:
        if i_mut_bits < 0.1:
            return "USE_BOLTZ"
        if i_mut_bits < 0.3:
            return "USE_HW"
        return "USE_HW_ONLY"

    def g11_odds_hw(self, c_a: float, c_b: float) -> float:
        return float(c_b) / max(float(c_a), 1e-12)

    def g11_k_ab(self, odds_hw: float, odds_pred: float) -> float:
        return float(odds_hw) / max(float(odds_pred), 1e-12) - 1.0

    def g12_crossover(self, epsilon: float = 0.0) -> float:
        return (self.g.tau_smem + epsilon) / max(self.g.tau_hbm - self.g.tau_smem, 1)

    def g13_hill(self, x: float, v: float = 1.0, k: float = 0.5, n: float = 2.0, **kw) -> float:
        v = kw.get("V", v)
        k = kw.get("K", k)
        x_n = max(x, 0.0) ** n
        return v * x_n / (k**n + x_n + 1e-12)

    def g14_e_mem(self, q_g: float, q_l2: float, q_s: float, w_active: int = 8) -> float:
        denom = max(w_active, 1)
        return (q_g * self.g.tau_hbm + q_l2 * self.g.tau_l2 + q_s * self.g.tau_smem) / denom

    def g14_entropy(self, q_g: float, q_l2: float, q_s: float) -> float:
        total = q_g + q_l2 + q_s
        if total <= 0:
            return 0.0
        h = 0.0
        for q in (q_g, q_l2, q_s):
            if q > 0:
                p = q / total
                h -= p * math.log(p)
        return h

    def g14_e_energy_norm(self, q_g: float, q_l2: float, q_s: float, t_kernel_s: float) -> float:
        work = q_g * self.g.tau_hbm + q_l2 * self.g.tau_l2 + q_s * self.g.tau_smem
        return min(1.0, work * max(t_kernel_s, 1e-9) / 1e8)

    def g14_sigma_sm_h_l2(self, b_g: float, b_l2_sust: float, h_l2_max: float = 1.0) -> float:
        if b_g <= b_l2_sust:
            return 0.0
        return math.sqrt((b_g - b_l2_sust) / max(b_g + b_l2_sust, 1e-12)) * h_l2_max

    def g15_hill_occupancy(self, rho: float) -> float:
        return self.g13_hill(rho, v=1.0, k=0.5, n=2.0)

    def g16_occupancy(self, regs_per_thread: int, smem_bytes: int, threads_per_block: int) -> float:
        if threads_per_block <= 0:
            return 0.0
        blocks_by_threads = self.g.t_sm_max // threads_per_block
        regs_per_block = math.ceil(regs_per_thread * threads_per_block / self.g.g_reg) * self.g.g_reg
        blocks_by_regs = self.g.r_max // max(regs_per_block, 1)
        blocks_by_smem = self.g.smem_per_sm // max(smem_bytes, 1)
        resident_blocks = min(self.g.b_sm_max, blocks_by_threads, blocks_by_regs, blocks_by_smem)
        active_warps = resident_blocks * math.ceil(threads_per_block / self.g.w_warp)
        return min(1.0, active_warps / self.g.w_max)

    def g18_critical_path(self, dag_cycles) -> float:
        return float(sum(dag_cycles))

    def g20_syndrome_zero_iff_equal(self, d: int, d_prime: int) -> bool:
        return (d ^ d_prime) == 0

    def g20_uncorr_two_bit(self, p_bit: float) -> float:
        return 2016.0 * p_bit * p_bit

    def g21_sr_error_bound_scale(self, n: int) -> float:
        return math.sqrt(max(n, 0))

    def g22_eta_star(self, k: int, t: int, psi_max: float = 1.0) -> float:
        return math.sqrt(2.0 * math.log(max(k, 2)) / max(t * psi_max * psi_max, 1e-12))

    def g22_exp3_update(self, weights, arm: int, cost: float, eta: float):
        updated = list(float(w) for w in weights)
        updated[arm] *= math.exp(-eta * cost)
        z = sum(updated)
        return [w / z if z > 0 else 1.0 / len(updated) for w in updated]

    def g22_regret_bound(self, t: int, k: int, eta: float, eps_corr_max: float = 0.0) -> float:
        return math.sqrt(2.0 * t * k * math.log(max(k, 2))) + t * eta * abs(eps_corr_max)

    def g23_e_addr(self, xi_seg: float, xi_lane: float, xi_bank: float) -> float:
        return float(xi_seg + xi_lane + xi_bank)

    def g23_curvature(self, a_prev: float, a_cur: float, a_next: float) -> float:
        return a_next - 2.0 * a_cur + a_prev

    def g24_h_l2_footprint(self, bytes_footprint: int, w_active: int = 8) -> float:
        pressure = bytes_footprint * max(w_active, 1) / max(self.g.l2_bytes, 1)
        return max(0.0, min(1.0, 1.0 - pressure))

    def g24_hit_prob(self, reuse_distance: float, l_reuse: float) -> float:
        return math.exp(-float(reuse_distance) / max(float(l_reuse), 1e-12))

    def g26_u_excl(self, bank_counts, atomic_counts=(), tau_bc: float = 1.0, tau_at: float = 1.0) -> float:
        bank = tau_bc * sum(int(n) * (int(n) - 1) for n in bank_counts)
        atom = tau_at * sum(int(n) * (int(n) - 1) for n in atomic_counts)
        return float(bank + atom)

    def g26_rho_conflict(self, u_excl: float, w_active: int, tau_bc: float = 1.0) -> float:
        return float(u_excl) / max(tau_bc * max(w_active, 1), 1e-12)

    def g27_digit_lb_cycles(self) -> int:
        return self.g7_scan_lb_cycles() + self.g.tau_smem

    def g27_digit_full_cycles(self) -> int:
        return 10 * self.g.tau_shfl + self.g.tau_smem

    def g27_t_scatter(self, n: int) -> float:
        return self.g3_transactions(n * 4) * self.g.tau_hbm

    def g28_l1i_pressure(self, n_instr: int, w_instr: int = 16, m_l1i: int = 32768) -> float:
        return n_instr * w_instr / m_l1i

    def g29_g_bw(self, bw_r: float, bw_s: float) -> float:
        total = bw_r + bw_s
        return (total * total / (1.0 + total)) if total > 0.5 else 0.0

    def g29_g_res(self, u_r, u_s) -> float:
        return float(u_r.get("reg", 0.0) * u_s.get("reg", 0.0) + u_r.get("shm", 0.0) * u_s.get("shm", 0.0) + u_r.get("warp", 0.0) * u_s.get("warp", 0.0))

    def g29_g_sm(self, n_r: float, n_s: float, eps: float = 0.01) -> float:
        return n_r * n_s / (n_r + n_s + eps)

    def g29_a_rs(self, u_r, u_s, n_r: float = 1.0, n_s: float = 1.0, lambdas=(0.4, 0.4, 0.2)) -> float:
        l_res, l_bw, l_sm = lambdas
        return l_res * self.g29_g_res(u_r, u_s) + l_bw * self.g29_g_bw(u_r.get("bw", 0.0), u_s.get("bw", 0.0)) + l_sm * self.g29_g_sm(n_r, n_s)

    def g30_psi_rt(self, rho_remote: float, rho_conflict: float, w_elig: float, alpha=(1.0, 1.0, 1.0)) -> float:
        a1, a2, a3 = alpha
        return a1 * rho_remote + a2 * rho_conflict - a3 * math.log1p(w_elig) / math.log1p(self.g.w_max)

    def g30_cupti_correct(self, psi_rt: float, t_kernel_s: float, delta_cupti_s: float) -> float:
        if t_kernel_s < 50e-6:
            return math.log(2.0)
        return psi_rt * t_kernel_s / max(t_kernel_s - delta_cupti_s, 1e-12)

    def g30_min_rt(self, k: int) -> float:
        return math.log(max(k, 2))

    def g31_mode_utility(self, alpha_r: float, pi_hw: float, tau_lag_s: float) -> float:
        delta_r = 1.0 / max(tau_lag_s, 1e-12)
        return alpha_r * pi_hw / delta_r

    def g32_jitter(self, times) -> float:
        vals = [float(t) for t in times]
        if not vals:
            return 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return var / max(mean, 1e-12)

    def g32_jitter_rel(self, times) -> float:
        vals = [float(t) for t in times]
        if not vals:
            return 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return var / max(mean * mean, 1e-12)

    def g33_hard_conflict(self, u_a, u_b) -> bool:
        return (u_a.get("reg", 0.0) + u_b.get("reg", 0.0) > 1.0) or (u_a.get("shm", 0.0) + u_b.get("shm", 0.0) > 1.0) or (u_a.get("warp", 0.0) + u_b.get("warp", 0.0) > 1.0)

    def g33_delta_h_l2_pred(self, footprint_intersection: float, h_l2_a: float, h_l2_b: float) -> float:
        return footprint_intersection / max(self.g.l2_bytes, 1) * min(h_l2_a, h_l2_b)

    def g34_prune_hard(self, regs_per_thread: int, smem_bytes: int, threads: int):
        if regs_per_thread * threads > self.g.r_max:
            return True, "registers"
        if smem_bytes > self.g.smem_per_sm:
            return True, "smem"
        if threads > self.g.t_sm_max:
            return True, "threads"
        return False, ""

    def g34_omega_size(self) -> int:
        return 1536

    def g35_tc_partial(self, r: int, c: int) -> float:
        pad_r = 16 * math.ceil(max(r, 1) / 16)
        pad_c = 16 * math.ceil(max(c, 1) / 16)
        return (r * c) / (pad_r * pad_c)

    def g19_reuse(self):
        return {
            "A_min_smem": (self.g.tau_hbm - self.g.tau_smem) / self.g.tau_smem,
            "A_min_shfl": (self.g.tau_hbm - self.g.tau_shfl) / self.g.tau_shfl,
            "A_min_l2": (self.g.tau_hbm - self.g.tau_l2) / self.g.tau_l2,
        }

    def meta_score_cycles(self, tau_eff: float, eta_exec: float, delta_tau_interact: float = 0.0, regret_cycles: float = 0.0, e_energy_norm: float = 0.0, lambdas=(0.6, 0.2, 0.1, 0.1)) -> float:
        le, li, lr, len_ = lambdas
        return le * tau_eff * (1.0 - eta_exec) + li * delta_tau_interact + lr * regret_cycles + len_ * e_energy_norm * tau_eff
