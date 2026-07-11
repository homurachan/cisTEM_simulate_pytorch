# This version fixes two major bugs. 
# 1. The radiation damage filter that generates checkboard artifacts.
# 2. Near zero frequency there was a very high amplitude artifact.
# And speed-up significantly.
"""
Python scaffold for cisTEM simulate with an integrated simplified explicit-water path.

Implemented core paths:
    PDB atoms -> voxel-integrated elastic scattering potential
              -> direct protein phase slabs without full 3D volume
              -> optional simplified explicit-water projected templates per slab
              -> projection or cisTEM-like multislice wave propagation
              -> optional DQE-like Fourier filter
              -> optional Poisson shot noise
              -> MRC or NumPy output

Not implemented yet by design:
    beam tilt full propagation
    objective aperture cosine mask
    CTFFIND amplitude contrast fitting
    phase plate
    mean solvent path and save-volume path in this direct-slab variant
    water-box shaking / periodic boundary
    tilt-series / star-parameter stack generation

The explicit-water implementation follows the useful cisTEM design idea that water
is added as projected 2D water templates into per-slab projected potentials, rather
than building a full 3D volume of all water molecules.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.special import erf

try:
    import mrcfile  # type: ignore
except Exception:  # pragma: no cover
    mrcfile = None


# -----------------------------
# cisTEM-like constants
# -----------------------------

BOND_SCALING_DEFAULT = 1.043
WN = 0.8045 * 0.79

ATOM_INDEX: Dict[str, int] = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
    "NA": 5,
    "MG": 6,
    "P": 7,
    "S": 8,
    "CL": 9,
    "K": 10,
    "CA": 11,
    "MN": 12,
    "FE": 13,
    "ZN": 14,
    "H2O": 15,
    "O-": 16,
}

ATOMIC_NUMBER = np.array(
    [1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20, 25, 26, 30, 10, 8],
    dtype=np.float32,
)

SCATTERING_A = np.array(
    [
        [0.0349, 0.1201, 0.1970, 0.0573, 0.1195],
        [0.0893, 0.2563, 0.7570, 1.0487, 0.3575],
        [0.1022, 0.3219, 0.7982, 0.8197, 0.1715],
        [0.0974, 0.2921, 0.6910, 0.6990, 0.2039],
        [0.1083, 0.3175, 0.6487, 0.5846, 0.1421],
        [0.2142, 0.6853, 0.7692, 1.6589, 1.4482],
        [0.2314, 0.6866, 0.9677, 2.1882, 1.1339],
        [0.2548, 0.6106, 1.4541, 2.3204, 0.8477],
        [0.2497, 0.5628, 1.3899, 2.1865, 0.7715],
        [0.2443, 0.5397, 1.3919, 2.0197, 0.6621],
        [0.4115, -1.4031, 2.2784, 2.6742, 2.2162],
        [0.4054, 1.3880, 2.1602, 3.7532, 2.2063],
        [0.3796, 1.2094, 1.7815, 2.5420, 1.5937],
        [0.3946, 1.2725, 1.7031, 2.3140, 1.4795],
        [0.4288, 1.2646, 1.4472, 1.8294, 1.0934],
        [WN * 0.07967, WN * 0.1053, WN * 0.2933, WN * 0.6831, WN * 1.304],
        [0.2050, 0.6280, 1.1700, 1.0300, 0.290],
    ],
    dtype=np.float64,
)

SCATTERING_B = np.array(
    [
        [0.5347, 3.5867, 12.347, 18.9525, 38.6269],
        [0.2465, 1.7100, 6.4094, 18.6113, 50.2523],
        [0.2451, 1.7481, 6.1925, 17.3894, 48.1431],
        [0.2067, 1.3815, 4.6943, 12.7105, 32.4726],
        [0.2057, 1.3439, 4.2788, 11.3932, 28.7881],
        [0.3334, 2.3446, 10.083, 48.3037, 138.270],
        [0.3278, 2.2720, 10.924, 39.2898, 101.9748],
        [0.2908, 1.8740, 8.5176, 24.3434, 63.2996],
        [0.2681, 1.6711, 7.0267, 19.5377, 50.3888],
        [0.2468, 1.5242, 6.1537, 16.6687, 42.3086],
        [0.3703, 3.3874, 13.1029, 68.9592, 194.4329],
        [0.3499, 3.0991, 11.9608, 53.9353, 142.3892],
        [0.2699, 2.0455, 7.4726, 31.0604, 91.5622],
        [0.2717, 2.0443, 7.6007, 29.9714, 86.2265],
        [0.2593, 1.7998, 6.7500, 25.5860, 73.5284],
        [WN * 4.718, WN * 16.75, WN * 0.4524, WN * 13.43, WN * 4.4480],
        [0.397, 2.6400, 8.8000, 27.1, 91.8],
    ],
    dtype=np.float64,
)

HYDRATION_RADIUS_EXTRA_SHIFT = -0.5
HYDRATION_RADIUS_VALS = np.array([0.1750, -0.1350, 2.23, 3.43, 4.78, 1.0, 1.77, 0.955], dtype=np.float64)
PUSH_BACK_BY = -1.48

DQE_A = np.array([-0.01516, -0.5662, -0.09731, -0.01551, 21.47], dtype=np.float64)
DQE_B = np.array([0.02671, -0.02504, 0.162, 0.2831, -2.28], dtype=np.float64)
DQE_C = np.array([0.01774, 0.1441, 0.1082, 0.07916, 1.372], dtype=np.float64)

WATER_DENSITY_PER_A3 = 0.94 * 0.6022140857 / 18.01528  # ~0.0314 waters/A^3
INELASTIC_SCALAR_WATER = 0.0725


@dataclass
class Atom:
    element: str
    xyz: np.ndarray
    bfactor: float = 0.0
    occupancy: float = 1.0
@dataclass
class WaterCache:
    templates: List[np.ndarray]
    water_coords: np.ndarray
    distance_2d: Optional[np.ndarray]
# Water cache related functions
def prepare_water_cache(atoms: List[Atom], cfg: SimConfig) -> Optional[WaterCache]:
    if not cfg.explicit_water:
        return None

    if cfg.solvent:
        raise ValueError(
            "Do not use --solvent and --explicit-water together. "
            "--solvent is mean solvent; --explicit-water is explicit solvent."
        )

    if cfg.verbose:
        print("Generating explicit waters with 3D protein exclusion...")

    water_coords = generate_explicit_water_coords(
        cfg,
        atoms=atoms,
        exclude_from_atoms=True,
    )

    if cfg.verbose:
        print("Precomputing projected water templates...")

    templates = precompute_projected_water_templates(cfg)

    if cfg.water_soft_weight:
        distance_2d = nearest_atom_distance_2d(atoms, cfg, max_r_a=9.0)
    else:
        distance_2d = None

    return WaterCache(
        templates=templates,
        water_coords=water_coords,
        distance_2d=distance_2d,
    )

@dataclass
class SimConfig:
    box: int = 256
    pixel_size: float = 1.0
    kv: float = 300.0
    cs_mm: float = 2.7
    defocus_u: float = 15000.0
    defocus_v: Optional[float] = None
    defocus_angle_deg: float = 0.0
    amplitude_contrast: float = 0.07
    phase_shift_rad: float = 0.0
    dose_e_per_a2: float = 30.0
    n_slices: int = 16
    min_bfactor: float = 15.0
    bfactor_scaling: float = 0.0
    bond_scaling: float = BOND_SCALING_DEFAULT
    center_by_mass: bool = True
    euler_rot_deg: float = 0.0
    euler_tilt_deg: float = 0.0
    euler_psi_deg: float = 0.0
    euler_inverse: bool = False
    use_hydrogen: bool = False
    solvent: bool = False
    solvent_weight: float = 1.0
    solvent_shell_only: bool = True
    explicit_water: bool = False
    water_density_scale: float = 1.0
    water_max_count: Optional[int] = None
    water_template_radius_pix: int = 4
    water_subpix_n: int = 5
    water_exclude_below_a: float = 2.5
    water_soft_weight: bool = False
    water_bfactor: float = 34.0
    mode: str = "projection"
    poisson: bool = False
    dqe: bool = False
    seed: Optional[int] = None
    verbose: bool = False
# radiation damage
    radiation_damage: bool = False
    radiation_damage_where: str = "protein"  # "protein" or "all"
    pre_exposure_e_per_a2: float = 0.0
    exposure_filter_modify_signal: int = 0
    per_frame: bool = False
    number_of_frames: int = 1
    dose_per_frame_e_per_a2: Optional[float] = None
    save_frames: bool = False
    normalize_frame_sum: bool = True
    shake_waters: bool = False
    use_cache_atom: bool = False
    use_torch: bool = True
    device: str = "cuda"
    # cisTEM-like solvent working area. cfg.box remains the final output size;
    # explicit-water simulations are computed on box + 2*solvent_padding_pix
    # and cropped back after propagation/CTF.
    solvent_padding_pix: int = 64
    edge_taper_width_pix: int = 24
    sampled_mask_erode_pix: int = 7
    sampled_mask_lowpass: float = 0.05
    disable_cistem_edge_pipeline: bool = False
    # Stage-2 GPU-utilization controls.  The defaults keep memory use
    # conservative while enabling the convolutional water splat path requested
    # for explicit-water movie simulations.
    frame_batch_size: int = 1
    water_splat_method: str = "convolution"  # "convolution" or "scatter"
    water_template_chunk_size: int = 16
    gpu_water_shake: bool = True
    # Explicit-water coordinate seeding / protein-exclusion backend.  The
    # torch backend keeps stochastic octant seeding and atom exclusion on the
    # selected torch device; numpy is the legacy CPU RNG + cKDTree path.
    water_generation_backend: str = "torch"  # "torch" or "numpy"
    water_seed_z_chunk: int = 16
    water_seed_max_octants_per_chunk: int = 67108864
    water_filter_chunk_size: int = 250000
    water_filter_cell_size_a: Optional[float] = None
    # Legacy names retained for command-line compatibility; the current grid
    # exclusion backend does not use them.
    water_exclusion_atom_chunk_size: int = 1024
    water_exclusion_offset_chunk_size: int = 2048
    # Cached atom-slab GPU backend.  When --use-cache-atom is enabled and
    # bfactor_scaling == 0, the protein potential is built by grouping atom
    # centers into impulse maps and applying cached xy kernels by grouped conv2d.
    atom_cache_backend: str = "torch-convolution"  # "torch-convolution" or "numpy"
    atom_cache_subpix_n: int = 9
    atom_cache_radius_pix: int = 9
    atom_template_chunk_size: int = 16
    # cisTEM WaveFunctionPropagator compatibility.  These controls affect only
    # --mode multislice.  The defaults follow the simulator code path: preserve
    # slab edge means during propagation, convert/filter inelastic potentials
    # before building amplitude gratings, use negative slab propagator distances,
    # apply a defocus offset from the scattering center of mass, and apply the
    # objective-aperture mask before taking |wave|^2.
    objective_aperture_diameter_micron: float = 100.0
    objective_aperture_falloff_pix: float = 14.0
    disable_cistem_inelastic_filter: bool = False
    disable_cistem_defocus_offset: bool = False
    legacy_subtract_phase_edge_mean: bool = False
# -----------------------------
# Physics / Fourier helpers
# -----------------------------

def electron_wavelength_angstrom(kv: float) -> float:
    return 1226.39 / math.sqrt(kv * 1000.0 + 0.97845e-6 * (kv * 1000.0) ** 2) * 1e-2

# simulate_raw_projection_from_slabs_torch is only used for testing.
def simulate_raw_projection_from_slabs_torch(
    phase_slabs,
    cfg: SimConfig,
) -> torch.Tensor:
    proj = torch.stack(phase_slabs, dim=0).sum(dim=0).to(torch.float32)

    # 建议先减掉均值，避免 relion_reconstruct 主要重构 DC/背景
    proj = proj - torch.mean(proj)

    return proj
def frequency_grid(shape: Tuple[int, int], pixel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = shape
    fx = np.fft.fftfreq(nx, d=pixel_size)
    fy = np.fft.fftfreq(ny, d=pixel_size)
    kx, ky = np.meshgrid(fx, fy)
    k2 = kx * kx + ky * ky
    return kx, ky, k2
def electron_dose_voltage_scaling(kv: float) -> float:
    """cisTEM ElectronDose voltage scaling for critical dose."""
    kv = float(kv)
    if 299.0 < kv < 301.0:
        return 1.0
    if 199.0 < kv < 201.0:
        return 0.8
    if 99.0 < kv < 101.0:
        return 0.532
    raise ValueError(
        f"Unsupported voltage for cisTEM ElectronDose: {kv}. "
        "cisTEM supports 100, 200, or 300 kV in this model."
    )


def critical_exposure_grant_grigorieff(
    freq_a_inv: np.ndarray,
    kv: float = 300.0,
) -> np.ndarray:
    """cisTEM/Grant-Grigorieff critical exposure N_e(f).

    cisTEM's ElectronDose stores spatial_frequency as f^2 and evaluates

        N_e(f) = (0.24499 * (f^2)^(-1.6649/2) + 2.8141) * voltage_scale

    which is equivalent to 0.24499 * f^-1.6649 + 2.8141 at 300 kV.
    """
    f = np.asarray(freq_a_inv, dtype=np.float32)
    ne = np.empty_like(f, dtype=np.float32)

    positive = f > 1.0e-6
    ne[positive] = 0.24499 * np.power(f[positive], -1.6649) + 2.8141
    ne[~positive] = 1.0e9

    ne *= np.float32(electron_dose_voltage_scaling(kv))
    return ne.astype(np.float32)


def dose_filter_interval(
    shape: Tuple[int, int],
    pixel_size: float,
    exposure_start_e_per_a2: float,
    exposure_end_e_per_a2: float,
    average_over_interval: bool = False,
    kv: float = 300.0,
) -> np.ndarray:
    """cisTEM ElectronDose Fourier-space radiation-damage filter.

    For the 2D simulate.cpp path, cisTEM calls CalculateDoseFilterAs1DArray(),
    whose inline ReturnDoseFilter is

        exp(-0.5 * dose_finish / critical_dose).

    The cumulative/integrated form is kept for completeness, but the direct
    2D slab filter should use average_over_interval=False.
    """
    _, _, k2 = frequency_grid(shape, pixel_size)
    freq = np.sqrt(k2).astype(np.float32)

    ne = critical_exposure_grant_grigorieff(freq, kv=kv)

    d0 = float(exposure_start_e_per_a2)
    d1 = float(exposure_end_e_per_a2)
    if d1 < d0:
        raise ValueError("exposure_end_e_per_a2 must be >= exposure_start_e_per_a2")

    if d1 == d0:
        return np.ones(shape, dtype=np.float32)

    if average_over_interval:
        # Matches ElectronDose::ReturnCummulativeDoseFilter. Note that cisTEM
        # divides by dose_at_end_of_exposure, not by (dose_end-dose_start).
        denom = max(d1, 1.0e-12)
        filt = (2.0 * ne / denom) * (
            np.exp(-0.5 * d0 / ne) - np.exp(-0.5 * d1 / ne)
        )
    else:
        filt = np.exp(-0.5 * d1 / ne)

    filt = np.asarray(filt, dtype=np.float32)
    filt[0, 0] = 1.0
    return filt

def apply_exposure_filter_2d(
    img: np.ndarray,
    pixel_size: float,
    exposure_start_e_per_a2: float,
    exposure_end_e_per_a2: float,
    modify_signal: int = 0,
    subtract_edge_mean: bool = True,
    kv: float = 300.0,
) -> np.ndarray:
    """
    Apply radiation-damage Fourier filter to a 2D projected potential.

    This corresponds to the cisTEM DO_EXPOSURE_FILTER == 2 idea:
        scattering_potential[iSlab].ForwardFFT()
        multiply by dose_filter
        BackwardFFT()

    subtract_edge_mean=True protects the DC/background term and avoids making
    a constant solvent/protein background fade with dose.
    """
    x = img.astype(np.float32, copy=False)

    if subtract_edge_mean:
        bg = edge_mean_2d(x)
        work = x - bg
    else:
        bg = 0.0
        work = x

    filt = dose_filter_interval(
        work.shape,
        pixel_size,
        exposure_start_e_per_a2,
        exposure_end_e_per_a2,
        average_over_interval=False,
        kv=kv,
    )

    if modify_signal == 1:
        # cisTEM has this form in the 3D reference path:
        # 1 - (1 - F)/(1 + F) = 2F/(1+F)
        filt = 1.0 - (1.0 - filt) / (1.0 + filt)
    elif modify_signal == 2:
        filt = np.sqrt(np.maximum(filt, 0.0)).astype(np.float32)

    out = np.fft.ifft2(np.fft.fft2(work) * filt).real.astype(np.float32)
    return out + np.float32(bg)


def apply_radiation_damage_to_slabs(
    phase_slabs: List[np.ndarray],
    cfg: SimConfig,
    exposure_start_e_per_a2: Optional[float] = None,
    exposure_end_e_per_a2: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Apply 2D exposure filter to each projected protein slab.

    For the current one-image Python simulator, default interval is:
        [cfg.pre_exposure_e_per_a2,
         cfg.pre_exposure_e_per_a2 + cfg.dose_e_per_a2]

    This is analogous to cisTEM's per-frame call using
    current_total_exposure -> current_total_exposure + dose_per_frame.
    """
    if exposure_start_e_per_a2 is None:
        exposure_start_e_per_a2 = float(getattr(cfg, "pre_exposure_e_per_a2", 0.0))
    if exposure_end_e_per_a2 is None:
        exposure_end_e_per_a2 = exposure_start_e_per_a2 + float(cfg.dose_e_per_a2)

    out: List[np.ndarray] = []
    for i, slab in enumerate(phase_slabs):
        filtered = apply_exposure_filter_2d(
            slab,
            cfg.pixel_size,
            exposure_start_e_per_a2,
            exposure_end_e_per_a2,
            modify_signal=int(getattr(cfg, "exposure_filter_modify_signal", 0)),
            subtract_edge_mean=True,
            kv=float(getattr(cfg, "kv", 300.0)),
        )
        out.append(filtered.astype(np.float32, copy=False))

        if getattr(cfg, "verbose", False):
            print(
                f"  radiation damage slab {i + 1}/{len(phase_slabs)}: "
                f"{exposure_start_e_per_a2:.2f} -> {exposure_end_e_per_a2:.2f} e-/A^2"
            )

    return out

def ctf_2d(
    shape: Tuple[int, int],
    pixel_size: float,
    kv: float,
    cs_mm: float,
    defocus_u: float,
    defocus_v: Optional[float] = None,
    defocus_angle_deg: float = 0.0,
    amplitude_contrast: float = 0.07,
    phase_shift_rad: float = 0.0,
) -> np.ndarray:
    if defocus_v is None:
        defocus_v = defocus_u
    lam = electron_wavelength_angstrom(kv)
    cs_a = cs_mm * 1e7
    kx, ky, k2 = frequency_grid(shape, pixel_size)
    theta = np.arctan2(ky, kx) - np.deg2rad(defocus_angle_deg)
    defocus = 0.5 * (defocus_u + defocus_v) + 0.5 * (defocus_u - defocus_v) * np.cos(2.0 * theta)
    chi = math.pi * lam * defocus * k2 - 0.5 * math.pi * cs_a * (lam ** 3) * (k2 ** 2) + phase_shift_rad
    amp = float(amplitude_contrast)
    return (-(max(0.0, 1.0 - amp * amp) ** 0.5) * np.sin(chi) - amp * np.cos(chi)).astype(np.float32)

def cistem_complex_lens_transfer(
    shape: Tuple[int, int],
    pixel_size: float,
    kv: float,
    cs_mm: float,
    defocus_u: float,
    defocus_v: Optional[float] = None,
    defocus_angle_deg: float = 0.0,
    phase_shift_rad: float = 0.0,
) -> np.ndarray:
    """
    Closer to cisTEM WaveFunctionPropagator final CTF step.

    cisTEM uses two CTF objects:
        ctf[0] with amplitude contrast = 1
        ctf[1] with amplitude contrast = 0

    The final real/imag recombination is equivalent to multiplying the complex
    wave by:

        H = ctf_amp1 + i * ctf_amp0

    up to a possible global sign convention.
    """
    h_real = ctf_2d(
        shape,
        pixel_size,
        kv,
        cs_mm,
        defocus_u,
        defocus_v,
        defocus_angle_deg,
        amplitude_contrast=1.0,
        phase_shift_rad=phase_shift_rad,
    )

    h_imag = ctf_2d(
        shape,
        pixel_size,
        kv,
        cs_mm,
        defocus_u,
        defocus_v,
        defocus_angle_deg,
        amplitude_contrast=0.0,
        phase_shift_rad=phase_shift_rad,
    )

    return (h_real + 1j * h_imag).astype(np.complex64)
def shake_waters_3d(
    water_coords: np.ndarray,
    cfg: SimConfig,
    dose_per_frame_e_per_a2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Approximate cisTEM Water::ShakeWaters3d.

    cisTEM uses:
        random_sigma = 1.5 * dose_per_frame
    and adds independent normal shifts to x/y/z, then applies periodic wrapping.

    Coordinates are in Angstrom and centered at box origin.
    """
    if water_coords.size == 0:
        return water_coords

    sigma_a = 1.5 * float(dose_per_frame_e_per_a2)

    shaken = water_coords.astype(np.float32, copy=True)
    shaken += rng.normal(0.0, sigma_a, size=shaken.shape).astype(np.float32)

    box_a = float(cfg.box) * float(cfg.pixel_size)
    half = 0.5 * box_a

    # periodic wrap to [-half, half)
    shaken = ((shaken + half) % box_a) - half

    return shaken.astype(np.float32, copy=False)

def fresnel_propagator(shape: Tuple[int, int], pixel_size: float, kv: float, dz_angstrom: float) -> np.ndarray:
    lam = electron_wavelength_angstrom(kv)
    _, _, k2 = frequency_grid(shape, pixel_size)
#    return np.exp(-1j * math.pi * lam * dz_angstrom * k2).astype(np.complex64)
    return np.exp(+1j * math.pi * lam * dz_angstrom * k2).astype(np.complex64)

def dqe_filter(shape: Tuple[int, int], pixel_size: float, root: bool = True) -> np.ndarray:
    _, _, k2 = frequency_grid(shape, pixel_size)
    freq = np.sqrt(k2)
    out = np.zeros_like(freq, dtype=np.float64)
    for a, b, c in zip(DQE_A, DQE_B, DQE_C):
        out += a * np.exp(-((freq - b) ** 2) / (2.0 * c * c))
    out = np.clip(out, 0.0, None)
    out /= out.max() if out.max() > 0 else 1.0
    if root:
        out = np.sqrt(out)
    return out.astype(np.float32)


def apply_fourier_filter(img: np.ndarray, filt: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.fft2(img) * filt).real.astype(np.float32)


# -----------------------------
# PDB and scattering potential
# -----------------------------

def infer_element(line: str) -> str:
    elem = line[76:78].strip().upper() if len(line) >= 78 else ""
    if not elem:
        name = line[12:16].strip().upper()
        name = "".join(ch for ch in name if ch.isalpha())
        if len(name) >= 2 and name[:2] in ATOM_INDEX:
            elem = name[:2]
        elif name:
            elem = name[0]
    return elem


def read_pdb_atoms(path: str | Path, use_hydrogen: bool = False, allow_hetatm: bool = True) -> List[Atom]:
    atoms: List[Atom] = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in {"ATOM", "HETATM"}:
                continue
            if rec == "HETATM" and not allow_hetatm:
                continue
            elem = infer_element(line)
            if not elem or elem not in ATOM_INDEX:
                continue
            if elem == "H" and not use_hydrogen:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occ = float(line[54:60]) if len(line) >= 60 and line[54:60].strip() else 1.0
                bf = float(line[60:66]) if len(line) >= 66 and line[60:66].strip() else 0.0
            except ValueError:
                continue
            atoms.append(Atom(elem, np.array([x, y, z], dtype=np.float64), bf, occ))
    if not atoms:
        raise ValueError(f"No supported atoms found in {path}")
    return atoms


def center_atoms(atoms: List[Atom]) -> None:
    coords = np.array([a.xyz for a in atoms], dtype=np.float64)
    weights = np.array([ATOMIC_NUMBER[ATOM_INDEX[a.element]] for a in atoms], dtype=np.float64)
    com = (coords * weights[:, None]).sum(axis=0) / weights.sum()
    for atom in atoms:
        atom.xyz = atom.xyz - com


def rotation_matrix_zyz_relion_old(rot_deg: float, tilt_deg: float, psi_deg: float) -> np.ndarray:
    """Return an active ZYZ Euler rotation matrix.

    This uses the common RELION/cisTEM-style angle names Rot, Tilt, Psi and
    composes the active coordinate rotation as:

        R = Rz(Psi) @ Ry(Tilt) @ Rz(Rot)

    Angles are in degrees. For comparison to a package that uses the opposite
    passive transform, use the transpose/inverse of this matrix.
    """
    r = math.radians(rot_deg)
    t = math.radians(tilt_deg)
    p = math.radians(psi_deg)

    cr, sr = math.cos(r), math.sin(r)
    ct, st = math.cos(t), math.sin(t)
    cp, sp = math.cos(p), math.sin(p)

    rz_rot = np.array(
        [[cr, -sr, 0.0],
         [sr,  cr, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ry_tilt = np.array(
        [[ ct, 0.0, st],
         [0.0, 1.0, 0.0],
         [-st, 0.0, ct]],
        dtype=np.float64,
    )
    rz_psi = np.array(
        [[cp, -sp, 0.0],
         [sp,  cp, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz_psi @ ry_tilt @ rz_rot
def rotation_matrix_zyz_relion(rot_deg: float, tilt_deg: float, psi_deg: float) -> np.ndarray:
    """
    RELION-style Euler_angles2matrix.

    This follows the matrix form used by RELION's Euler_angles2matrix(alpha,beta,gamma):

        alpha = rot
        beta  = tilt
        gamma = psi

    In your actual projection path, you use:

        rotation_matrix = Euler_angles2matrix(rot, tilt, psi)
        rotation_matrix = rotation_matrix.T

    Therefore, keep --euler-inverse enabled if you want the same behavior,
    because rotate_atoms_euler() will transpose this matrix when inverse=True.
    """
    alpha = math.radians(rot_deg)
    beta = math.radians(tilt_deg)
    gamma = math.radians(psi_deg)

    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)

    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)

    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    A = np.empty((3, 3), dtype=np.float64)

    A[0, 0] =  cg * cc - sg * sa
    A[0, 1] =  cg * cs + sg * ca
    A[0, 2] = -cg * sb

    A[1, 0] = -sg * cc - cg * sa
    A[1, 1] = -sg * cs + cg * ca
    A[1, 2] =  sg * sb

    A[2, 0] =  sc
    A[2, 1] =  ss
    A[2, 2] =  cb

    return A

def rotate_atoms_euler(atoms: List[Atom], rot_deg: float, tilt_deg: float, psi_deg: float, inverse: bool = False) -> None:
    """Rotate atom coordinates in-place around the current origin."""
    if rot_deg == 0.0 and tilt_deg == 0.0 and psi_deg == 0.0:
        return
    rmat = rotation_matrix_zyz_relion(rot_deg, tilt_deg, psi_deg)
    if inverse:
        rmat = rmat.T
    for atom in atoms:
        atom.xyz = rmat @ atom.xyz


def complete_bfactor(atom_b: float, bfactor_scaling: float, min_bfactor: float) -> float:
    return 0.25 * (atom_b * bfactor_scaling + min_bfactor)


def voxel_integrated_potential(
    x1: np.ndarray,
    x2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    atom_index: int,
    bfactor: float,
    lead_term: float,
) -> np.ndarray:
    """Fast separable voxel-integrated 3D Gaussian potential.

    The 3D integral factorizes into x/y/z 1D erf differences, so erf is only
    evaluated on 1D arrays and the 3D block is assembled by broadcasting.
    """
    out = np.zeros((z1.size, y1.size, x1.size), dtype=np.float32)
    for i in range(5):
        b_total = SCATTERING_B[atom_index, i] + bfactor
        if b_total <= 0:
            continue
        bplus = math.sqrt(4.0 * math.pi * math.pi / b_total)
        dx = (erf(bplus * x2) - erf(bplus * x1)).astype(np.float32, copy=False)
        dy = (erf(bplus * y2) - erf(bplus * y1)).astype(np.float32, copy=False)
        dz = (erf(bplus * z2) - erf(bplus * z1)).astype(np.float32, copy=False)
        out += (
            np.float32(SCATTERING_A[atom_index, i] * lead_term)
            * dz[:, None, None]
            * dy[None, :, None]
            * dx[None, None, :]
        )
    return out
def add_template_periodic(
    img: np.ndarray,
    template: np.ndarray,
    center_y: int,
    center_x: int,
    scale: float = 1.0,
) -> None:
    """
    Periodically add a 2D template centered at (center_y, center_x).

    This avoids edge clipping artifacts for explicit water.
    """
    ny, nx = img.shape
    ry = template.shape[0] // 2
    rx = template.shape[1] // 2

    ys = (np.arange(template.shape[0]) + center_y - ry) % ny
    xs = (np.arange(template.shape[1]) + center_x - rx) % nx

    img[np.ix_(ys, xs)] += scale * template



def add_template_clipped(
    img: np.ndarray,
    template: np.ndarray,
    center_y: int,
    center_x: int,
    scale: float = 1.0,
) -> None:
    """Add a 2D template with cisTEM-like edge clipping, not wrapping.

    simulate.cpp bounds-checks each water-template pixel before adding it to
    projected_water_atoms. Wrapping a projected water atom across the image
    boundary introduces periodic structure that becomes very visible after
    Fourier filtering.
    """
    ny, nx = img.shape
    ry = template.shape[0] // 2
    rx = template.shape[1] // 2

    y0 = int(center_y) - ry
    x0 = int(center_x) - rx
    y1 = y0 + template.shape[0]
    x1 = x0 + template.shape[1]

    iy0 = max(0, y0)
    ix0 = max(0, x0)
    iy1 = min(ny, y1)
    ix1 = min(nx, x1)

    if iy0 >= iy1 or ix0 >= ix1:
        return

    ty0 = iy0 - y0
    tx0 = ix0 - x0
    ty1 = ty0 + (iy1 - iy0)
    tx1 = tx0 + (ix1 - ix0)

    img[iy0:iy1, ix0:ix1] += scale * template[ty0:ty1, tx0:tx1]


def cistem_subpixel_offsets(subpix_n: int) -> np.ndarray:
    """Offsets used by simulate.cpp for projected water templates.

    With SUB_PIXEL_NEIGHBORHOOD=2, cisTEM creates five offsets along each
    axis: -2/6, -1/6, 0, 1/6, 2/6. Generalizing to odd subpix_n=2N+1 gives
    (idx-N)/(2N+2), i.e. denominator subpix_n+1.
    """
    subpix_n = int(subpix_n)
    if subpix_n <= 0 or subpix_n % 2 != 1:
        raise ValueError("cisTEM water subpixel grid must be a positive odd integer, e.g. 5")
    half = (subpix_n - 1) // 2
    return (np.arange(subpix_n, dtype=np.float32) - half) / float(subpix_n + 1)


def cistem_subpixel_index_from_fraction(frac: float, subpix_n: int) -> int:
    """Map frac in [-0.5, 0.5) to cisTEM's water subpixel index."""
    subpix_n = int(subpix_n)
    half = (subpix_n - 1) // 2
    idx = math.trunc(float(frac) * float(subpix_n)) + half
    return int(np.clip(idx, 0, subpix_n - 1))


def effective_water_dose_per_frame(cfg: SimConfig) -> float:
    """Dose used in cisTEM's water template B-factor.

    simulate.cpp uses WATER_BFACTOR_PER_ELECTRON_PER_SQANG * dose_per_frame.
    For a non-movie single-image run where no per-frame dose was supplied, keep
    the historical default equivalent to dose_per_frame=1 e-/A^2.
    """
    dose_per_frame = getattr(cfg, "dose_per_frame_e_per_a2", None)
    if dose_per_frame is not None:
        return float(dose_per_frame)

    n_frames = max(1, int(getattr(cfg, "number_of_frames", 1)))
    if getattr(cfg, "per_frame", False) or n_frames > 1:
        return float(getattr(cfg, "dose_e_per_a2", 1.0)) / float(n_frames)

    return 1.0


def cistem_water_template_bfactor(cfg: SimConfig) -> float:
    return 0.25 * float(cfg.water_bfactor) * max(effective_water_dose_per_frame(cfg), 0.0)

def atom_neighborhood_radius(pixel_size: float, bfactor: float) -> int:
    sigma = math.sqrt(max(SCATTERING_B.max() + bfactor, 1e-6)) / (2.0 * math.pi)
    radius_a = max(3.0 * pixel_size, 4.0 * sigma + 2.0 * pixel_size)
    return max(2, int(math.ceil(radius_a / pixel_size)))


def make_scattering_volume(atoms: List[Atom], cfg: SimConfig) -> np.ndarray:
    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    vol = np.zeros((n, n, n), dtype=np.float32)  # z, y, x
    lam = electron_wavelength_angstrom(cfg.kv)
    lead_term = cfg.bond_scaling * lam / 8.0 / (ps * ps)
    half = n / 2.0

    for i_atom, atom in enumerate(atoms):
        ai = ATOM_INDEX[atom.element]
        x, y, z = atom.xyz / ps + half
        ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
        bf = complete_bfactor(atom.bfactor, cfg.bfactor_scaling, cfg.min_bfactor)
        r = atom_neighborhood_radius(ps, bf)
        xs = np.arange(max(0, ix - r), min(n, ix + r + 1))
        ys = np.arange(max(0, iy - r), min(n, iy + r + 1))
        zs = np.arange(max(0, iz - r), min(n, iz + r + 1))
        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            continue

        x1 = ((xs - half) * ps - atom.xyz[0]) - 0.5 * ps
        x2 = x1 + ps
        y1 = ((ys - half) * ps - atom.xyz[1]) - 0.5 * ps
        y2 = y1 + ps
        z1 = ((zs - half) * ps - atom.xyz[2]) - 0.5 * ps
        z2 = z1 + ps

        pot = voxel_integrated_potential(x1, x2, y1, y2, z1, z2, ai, bf, lead_term)
        vol[np.ix_(zs, ys, xs)] += pot * atom.occupancy

        if cfg.verbose and (i_atom + 1) % 10000 == 0:
            print(f"  potential atoms: {i_atom + 1}/{len(atoms)}")

    return vol


# -----------------------------
# Mean hydration shell
# -----------------------------

def hydration_weight(radius_a: np.ndarray, pixel_size: float) -> np.ndarray:
    v = HYDRATION_RADIUS_VALS
    shifted = radius_a + PUSH_BACK_BY
    return (
        0.5
        + 0.5 * erf((shifted - (v[2] + HYDRATION_RADIUS_EXTRA_SHIFT * pixel_size)) / (math.sqrt(2.0) * v[5]))
        + v[0] * np.exp(-((shifted - (v[3] + HYDRATION_RADIUS_EXTRA_SHIFT * pixel_size)) ** 2) / (2.0 * v[6] ** 2))
        + v[1] * np.exp(-((shifted - (v[4] + HYDRATION_RADIUS_EXTRA_SHIFT * pixel_size)) ** 2) / (2.0 * v[7] ** 2))
    )


def add_mean_hydration_shell(vol: np.ndarray, atoms: List[Atom], cfg: SimConfig) -> np.ndarray:
    n = cfg.box
    ps = cfg.pixel_size
    half = n / 2.0
    dist2 = np.full(vol.shape, np.inf, dtype=np.float32)
    max_r_a = 9.0
    r_pix = int(math.ceil(max_r_a / ps))

    for i_atom, atom in enumerate(atoms):
        x, y, z = atom.xyz / ps + half
        ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
        x0, x1i = max(0, ix - r_pix), min(n, ix + r_pix + 1)
        y0, y1i = max(0, iy - r_pix), min(n, iy + r_pix + 1)
        z0, z1i = max(0, iz - r_pix), min(n, iz + r_pix + 1)
        if x0 >= x1i or y0 >= y1i or z0 >= z1i:
            continue

        xs = np.arange(x0, x1i, dtype=np.float32)
        ys = np.arange(y0, y1i, dtype=np.float32)
        zs = np.arange(z0, z1i, dtype=np.float32)
        dx2 = ((xs - half) * ps - atom.xyz[0]) ** 2
        dy2 = ((ys - half) * ps - atom.xyz[1]) ** 2
        dz2 = ((zs - half) * ps - atom.xyz[2]) ** 2
        d2 = (dz2[:, None, None] + dy2[None, :, None] + dx2[None, None, :]).astype(np.float32, copy=False)
        block = dist2[z0:z1i, y0:y1i, x0:x1i]
        np.minimum(block, d2, out=block)

        if cfg.verbose and (i_atom + 1) % 10000 == 0:
            print(f"  mean hydration distance atoms: {i_atom + 1}/{len(atoms)}")

    finite = np.isfinite(dist2)
    r = np.zeros_like(dist2, dtype=np.float32)
    np.sqrt(dist2, out=r, where=finite)
    shell = np.empty_like(dist2, dtype=np.float32)
    shell[finite] = hydration_weight(r[finite], ps).astype(np.float32, copy=False)
    shell[~finite] = 0.0 if cfg.solvent_shell_only else 1.0

    oxygen_scale = float(SCATTERING_A[ATOM_INDEX["O"]].sum()) * WATER_DENSITY_PER_A3 * (ps ** 3)
    return vol + cfg.solvent_weight * oxygen_scale * shell


# -----------------------------
# Slab representation and explicit water
# -----------------------------
def prepare_frame_slabs_from_base(
    base_phase_slabs: List[np.ndarray],
    base_amp_slabs: List[np.ndarray],
    dz_list: List[float],
    cfg: SimConfig,
    water_cache: Optional["WaterCache"],
    exposure_start_e_per_a2: float,
    exposure_end_e_per_a2: float,
    dose_per_frame_e_per_a2: float,
    iframe: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Prepare phase/amp slabs for one movie frame.

    Order:
        1. copy base protein slabs
        2. apply DO_EXPOSURE_FILTER == 2 style radiation damage to protein slabs
        3. optionally shake explicit waters for this frame
        4. add explicit water templates
    """
    phase_slabs = [s.copy() for s in base_phase_slabs]
    amp_slabs = [s.copy() for s in base_amp_slabs]

    if getattr(cfg, "radiation_damage", False):
        phase_slabs = apply_radiation_damage_to_slabs(
            phase_slabs,
            cfg,
            exposure_start_e_per_a2=exposure_start_e_per_a2,
            exposure_end_e_per_a2=exposure_end_e_per_a2,
        )

    if water_cache is not None:
        frame_water_coords = water_cache.water_coords

        if getattr(cfg, "shake_waters", False):
            seed = None if cfg.seed is None else int(cfg.seed) + int(iframe) + 1000003
            rng = np.random.default_rng(seed)

            frame_water_coords = shake_waters_3d(
                water_cache.water_coords,
                cfg,
                dose_per_frame_e_per_a2=dose_per_frame_e_per_a2,
                rng=rng,
            )
            # cisTEM shakes water_box.water_coords in place, so the next frame
            # continues from the previous frame rather than the original lattice.
            water_cache.water_coords = frame_water_coords

        phase_slabs, amp_slabs = fill_water_potential_python(
            phase_slabs,
            amp_slabs,
            frame_water_coords,
            water_cache.distance_2d,
            dz_list,
            cfg,
            water_cache.templates,
        )
        phase_slabs, amp_slabs = apply_cistem_edge_pipeline_numpy(phase_slabs, amp_slabs, cfg)

    return phase_slabs, amp_slabs
def volume_to_slabs(vol: np.ndarray, n_slices: int, pixel_size: float) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    chunks = np.array_split(vol, max(1, int(n_slices)), axis=0)
    phase_slabs = [c.sum(axis=0).astype(np.float32) for c in chunks]
    amp_slabs = [np.zeros_like(phase_slabs[0], dtype=np.float32) for _ in phase_slabs]
    dz_list = [float(c.shape[0] * pixel_size) for c in chunks]
    return phase_slabs, amp_slabs, dz_list


def nearest_atom_distance_2d(atoms: List[Atom], cfg: SimConfig, max_r_a: float = 9.0) -> np.ndarray:
    """2D nearest-protein distance map used to soften/exclude projected water."""
    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    half = n / 2.0
    dist2 = np.full((n, n), np.inf, dtype=np.float32)
    r_pix = int(math.ceil(max_r_a / ps))

    for i_atom, atom in enumerate(atoms):
        x, y = atom.xyz[:2] / ps + half
        ix, iy = int(round(x)), int(round(y))
        x0, x1i = max(0, ix - r_pix), min(n, ix + r_pix + 1)
        y0, y1i = max(0, iy - r_pix), min(n, iy + r_pix + 1)
        if x0 >= x1i or y0 >= y1i:
            continue
        xs = np.arange(x0, x1i, dtype=np.float32)
        ys = np.arange(y0, y1i, dtype=np.float32)
        dx2 = ((xs - half) * ps - atom.xyz[0]) ** 2
        dy2 = ((ys - half) * ps - atom.xyz[1]) ** 2
        d2 = (dy2[:, None] + dx2[None, :]).astype(np.float32, copy=False)
        block = dist2[y0:y1i, x0:x1i]
        np.minimum(block, d2, out=block)

        if cfg.verbose and (i_atom + 1) % 10000 == 0:
            print(f"  water distance atoms: {i_atom + 1}/{len(atoms)}")

    dist = np.full_like(dist2, np.inf, dtype=np.float32)
    finite = np.isfinite(dist2)
    np.sqrt(dist2, out=dist, where=finite)
    return dist

def generate_explicit_water_coords(
    cfg: SimConfig,
    atoms: Optional[List[Atom]] = None,
    exclude_from_atoms: bool = True,
) -> np.ndarray:
    """
    Generate explicit water oxygen coordinates.

    This is closer to cisTEM water.cpp::SeedWaters3d:
    - estimate water number density from solvent density
    - test 8 sub-voxel octants per voxel
    - optionally remove waters that fall inside/too close to the protein

    Coordinates are returned in Angstrom, centered at box origin.
    """
    rng = np.random.default_rng(cfg.seed)

    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    box_a = n * ps

    # Same density expression as cisTEM water.cpp:
    # g/cm^3 * molecules/mole * mole/gram * cm^3 / 1e24 A^3
    water_density_per_a3 = 0.94 * 0.6022140857 / 18.01528

    expected_n_water = water_density_per_a3 * (box_a ** 3) * float(cfg.water_density_scale)

    # cisTEM uses probability per voxel octant:
    # probability_of_no_water_in_octant = 1 - expected_n / (nX*nY*nZ*8)
    p_water_octant = expected_n_water / float(n * n * n * 8)
    p_water_octant = float(np.clip(p_water_octant, 0.0, 1.0))

    if cfg.verbose:
        print(f"Expected waters: {expected_n_water:.3e}")
        print(f"Water probability per voxel octant: {p_water_octant:.6g}")

    # For huge boxes this fully vectorized version can be memory-heavy.
    # It is usually OK for 128-256 boxes. For very large boxes, use the chunked
    # version below.
    total_octants = n * n * n * 8

    if cfg.water_max_count is not None and cfg.water_max_count > 0:
        # Generate somewhat more candidates than requested, then filter/exclude.
        # This is useful for debugging.
        target = int(cfg.water_max_count)
        oversample = 3 if exclude_from_atoms and atoms is not None else 1
        n_candidates = min(total_octants, max(target * oversample, target))

        # Sample random voxel/octant indices directly.
        flat = rng.integers(0, total_octants, size=n_candidates, endpoint=False)

        oct_id = flat % 8
        voxel = flat // 8

        ix = voxel % n
        iy = (voxel // n) % n
        iz = voxel // (n * n)

        qx = np.where((oct_id & 1) == 0, -0.5, 0.5).astype(np.float32)
        qy = np.where((oct_id & 2) == 0, -0.5, 0.5).astype(np.float32)
        qz = np.where((oct_id & 4) == 0, -0.5, 0.5).astype(np.float32)

        coords_pix = np.stack(
            [
                ix.astype(np.float32) + qx,
                iy.astype(np.float32) + qy,
                iz.astype(np.float32) + qz,
            ],
            axis=1,
        )

        water_coords = (coords_pix - n / 2.0) * ps

    else:
        # Full stochastic octant seeding.
        # To avoid making an enormous boolean array for very large boxes,
        # process by z chunks.
        water_chunks = []
        z_chunk = 16

        octant_offsets = np.array(
            [
                [-0.5, -0.5, -0.5],
                [ 0.5, -0.5, -0.5],
                [-0.5,  0.5, -0.5],
                [ 0.5,  0.5, -0.5],
                [-0.5, -0.5,  0.5],
                [ 0.5, -0.5,  0.5],
                [-0.5,  0.5,  0.5],
                [ 0.5,  0.5,  0.5],
            ],
            dtype=np.float32,
        )

        for z0 in range(0, n, z_chunk):
            z1 = min(n, z0 + z_chunk)
            shape = (z1 - z0, n, n, 8)

            occ = rng.random(shape) < p_water_octant
            idx = np.argwhere(occ)

            if idx.size == 0:
                continue

            iz = idx[:, 0] + z0
            iy = idx[:, 1]
            ix = idx[:, 2]
            io = idx[:, 3]

            coords_pix = np.empty((idx.shape[0], 3), dtype=np.float32)
            coords_pix[:, 0] = ix.astype(np.float32) + octant_offsets[io, 0]
            coords_pix[:, 1] = iy.astype(np.float32) + octant_offsets[io, 1]
            coords_pix[:, 2] = iz.astype(np.float32) + octant_offsets[io, 2]

            water_chunks.append((coords_pix - n / 2.0) * ps)

        if len(water_chunks) == 0:
            water_coords = np.zeros((0, 3), dtype=np.float32)
        else:
            water_coords = np.concatenate(water_chunks, axis=0).astype(np.float32, copy=False)

    if cfg.verbose:
        print(f"Generated waters before protein exclusion: {len(water_coords)}")

    if exclude_from_atoms and atoms is not None and len(water_coords) > 0:
        water_coords = filter_waters_by_atom_distance(
            water_coords,
            atoms,
            cfg,
            exclude_radius_a=float(cfg.water_exclude_below_a),
        )

        if cfg.verbose:
            print(f"Remaining waters after protein exclusion: {len(water_coords)}")

    if cfg.water_max_count is not None and cfg.water_max_count > 0 and len(water_coords) > cfg.water_max_count:
        pick = rng.choice(len(water_coords), size=int(cfg.water_max_count), replace=False)
        water_coords = water_coords[pick]

        if cfg.verbose:
            print(f"Downsampled waters to water_max_count: {len(water_coords)}")

    return water_coords.astype(np.float32, copy=False)
    
def filter_waters_by_atom_distance(
    water_coords: np.ndarray,
    atoms: List[Atom],
    cfg: SimConfig,
    exclude_radius_a: float = 2.5,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """
    Remove explicit waters too close to protein atoms.

    This is the Python equivalent of using a protein distance_slab / water_mask_slab:
    waters whose true 3D distance to the nearest atom is below exclude_radius_a
    are discarded.

    Parameters
    ----------
    water_coords
        (N, 3) water oxygen coordinates in Angstrom, centered at box origin.
    atoms
        Rotated and centered PDB atoms. Important: call this after Euler rotation.
    exclude_radius_a
        Waters closer than this to any protein atom are removed.
        Good starting values:
            2.0 A  : permissive
            2.5 A  : recommended initial value
            3.0 A+ : more conservative
    """
    if water_coords.size == 0 or len(atoms) == 0:
        return water_coords

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "filter_waters_by_atom_distance requires scipy.spatial.cKDTree. "
            "Install scipy or use the slower voxel-mask version."
        ) from exc

    atom_xyz = np.asarray([a.xyz for a in atoms], dtype=np.float32)
    tree = cKDTree(atom_xyz)

    keep = np.ones(len(water_coords), dtype=bool)

    for start in range(0, len(water_coords), chunk_size):
        end = min(len(water_coords), start + chunk_size)

        # query_ball_point returns a list of neighbors within cutoff.
        # This avoids computing exact nearest distance for all points when
        # only a cutoff decision is needed.
        neighbors = tree.query_ball_point(
            water_coords[start:end],
            r=float(exclude_radius_a),
            workers=-1,
        )

        too_close = np.fromiter((len(x) > 0 for x in neighbors), dtype=bool)
        keep[start:end] = ~too_close

    return water_coords[keep].astype(np.float32, copy=False)
    
def precompute_projected_water_templates(cfg: SimConfig) -> List[np.ndarray]:
    """Precompute 2D projected O-atom templates for sub-pixel offsets."""
    subpix_n = int(cfg.water_subpix_n)
    if subpix_n <= 0:
        raise ValueError("water_subpix_n must be positive")
    radius_pix = int(cfg.water_template_radius_pix)
    ai = ATOM_INDEX["O"]
    ps = float(cfg.pixel_size)
    lam = electron_wavelength_angstrom(cfg.kv)
    lead_term = cfg.bond_scaling * lam / 8.0 / (ps * ps)
    # simulate.cpp uses 0.25 * WATER_BFACTOR_PER_ELECTRON_PER_SQANG * dose_per_frame.
    bf = cistem_water_template_bfactor(cfg)

    xs = np.arange(-radius_pix, radius_pix + 1)
    ys = np.arange(-radius_pix, radius_pix + 1)
    zs = np.arange(-radius_pix, radius_pix + 1)

    templates: List[np.ndarray] = []
    offsets = cistem_subpixel_offsets(subpix_n)
    for sz, dz in enumerate(offsets):
        for sy, dy in enumerate(offsets):
            for sx, dx in enumerate(offsets):
                dx = float(dx)
                dy = float(dy)
                dz = float(dz)

                x1 = (xs - dx) * ps - 0.5 * ps
                x2 = x1 + ps
                y1 = (ys - dy) * ps - 0.5 * ps
                y2 = y1 + ps
                z1 = (zs - dz) * ps - 0.5 * ps
                z2 = z1 + ps
                pot3d = voxel_integrated_potential(x1, x2, y1, y2, z1, z2, ai, bf, lead_term)
                templates.append(pot3d.sum(axis=0).astype(np.float32))
    return templates


def subpixel_index(frac: float, subpix_n: int) -> int:
    # frac is approximately in [-0.5, 0.5]. Map to 0..subpix_n-1.
    idx = int(round((frac + 0.5) * (subpix_n - 1)))
    return int(np.clip(idx, 0, subpix_n - 1))


def fill_water_potential_python(
    phase_slabs: List[np.ndarray],
    amp_slabs: List[np.ndarray],
    water_coords: np.ndarray,
    distance_2d: Optional[np.ndarray],
    dz_list: List[float],
    cfg: SimConfig,
    templates: List[np.ndarray],
    inelastic_scalar_water: float = 0.0725,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Add explicit projected water templates to phase/amp slabs.

    Important:
    - Hard protein exclusion should already have been done in
      generate_explicit_water_coords(..., atoms=atoms, exclude_from_atoms=True)
      or filter_waters_by_atom_distance().
    - distance_2d is used only for optional soft surface weighting.
      It is not used to decide whether a water molecule is inside protein.
    - dz_list is used to define slab z-edges, so nonuniform slab thicknesses
      are allowed.
    """
    if water_coords is None or len(water_coords) == 0:
        return phase_slabs, amp_slabs

    n_slices = len(phase_slabs)
    if n_slices == 0:
        return phase_slabs, amp_slabs

    ny, nx = phase_slabs[0].shape
    ps = float(cfg.pixel_size)
    half_x = nx / 2.0
    half_y = ny / 2.0

    # Infer subpixel grid size from number of templates.
    subpix_n = int(round(len(templates) ** (1.0 / 3.0)))
    if subpix_n ** 3 != len(templates):
        raise ValueError(
            f"Number of water templates must be subpix_n^3, got {len(templates)}"
        )

    radius_pix = templates[0].shape[0] // 2

    # z-edges from dz_list. Coordinates are centered at z=0.
    dz_arr = np.asarray(dz_list, dtype=np.float32)
    total_z_a = float(dz_arr.sum())
    z_edges = np.empty(n_slices + 1, dtype=np.float32)
    z_edges[0] = -0.5 * total_z_a
    z_edges[1:] = z_edges[0] + np.cumsum(dz_arr)

    # In the earlier version this is an approximate inelastic scaling.
    # Keep it conservative; you can tune later against cisTEM.
    oxygen_inelastic_to_elastic_ratio = math.sqrt(inelastic_scalar_water / 10.0)

    n_added = 0
    n_outside = 0
    n_edge = 0

    for x_a, y_a, z_a in water_coords:
        slab = int(np.searchsorted(z_edges, z_a, side="right") - 1)
        if slab < 0 or slab >= n_slices:
            n_outside += 1
            continue

        x_pix = x_a / ps + half_x
        y_pix = y_a / ps + half_y

        # cisTEM uses modf(coord + 0.5) / floor(coord + 0.5), not
        # Python/torch round-to-even. This matters because seeded waters sit
        # exactly at +/-0.5 voxel octants.
        ix = int(math.floor(float(x_pix) + 0.5))
        iy = int(math.floor(float(y_pix) + 0.5))

        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            n_edge += 1
            continue

        fx = float(x_pix) - ix
        fy = float(y_pix) - iy

        # z subpixel position inside this slab.
        # Convert z_a relative to slab start to pixel units.
        local_z_pix = (z_a - float(z_edges[slab])) / ps
        iz_int = int(math.floor(float(local_z_pix) + 0.5))
        fz = float(local_z_pix) - iz_int

        sx = cistem_subpixel_index_from_fraction(fx, subpix_n)
        sy = cistem_subpixel_index_from_fraction(fy, subpix_n)
        sz = cistem_subpixel_index_from_fraction(fz, subpix_n)

        template_index = sz * subpix_n * subpix_n + sy * subpix_n + sx
        template = templates[template_index]

        weight = 1.0
        if distance_2d is not None and getattr(cfg, "water_soft_weight", False):
            if 0 <= iy < distance_2d.shape[0] and 0 <= ix < distance_2d.shape[1]:
                d = distance_2d[iy, ix]
                if np.isfinite(d):
                    # Soft hydration-style weighting near protein projection.
                    # This is optional and approximate; hard exclusion is 3D.
                    weight = float(hydration_weight(np.array([d], dtype=np.float32), ps)[0])
                else:
                    weight = 1.0

        add_template_clipped(
            phase_slabs[slab],
            template,
            iy,
            ix,
            scale=weight,
        )
        add_template_clipped(
            amp_slabs[slab],
            template,
            iy,
            ix,
            scale=weight * oxygen_inelastic_to_elastic_ratio,
        )
        n_added += 1

    if getattr(cfg, "verbose", False):
        print(
            "fill_water_potential_python: "
            f"added={n_added}, outside_z={n_outside}, skipped_edge={n_edge}"
        )

    return phase_slabs, amp_slabs

# -----------------------------
# Imaging modes
# -----------------------------
def finalize_detector_image(img: np.ndarray, cfg: SimConfig) -> np.ndarray:
    out = img.astype(np.float32)

    # we always skip dqe.
    if cfg.dqe:
        mean = float(out.mean())
        contrast = out - mean
        contrast = apply_fourier_filter(
            contrast,
            dqe_filter(out.shape, cfg.pixel_size, root=True),
        )
        out = mean + contrast

    if cfg.poisson:
        rng = np.random.default_rng(cfg.seed)
        electrons_per_pixel = cfg.dose_e_per_a2 * (cfg.pixel_size ** 2)

        # intensity 应该先归一到平均值 1，再乘 dose
        mean = float(out.mean())
        norm_intensity = out / (mean + 1e-12)
        counts = rng.poisson(
            np.clip(norm_intensity * electrons_per_pixel, 0, None)
        ).astype(np.float32)

        # 输出 electron counts，而不是再强行归一化
        out = counts

    return out.astype(np.float32)
def edge_mean_2d(img: np.ndarray, width: int = 4) -> float:
    w = min(width, img.shape[0] // 4, img.shape[1] // 4)
    if w <= 0:
        return float(img.mean())
    edges = np.concatenate([
        img[:w, :].ravel(),
        img[-w:, :].ravel(),
        img[:, :w].ravel(),
        img[:, -w:].ravel(),
    ])
    return float(edges.mean())


def center_crop_numpy(arr: np.ndarray, final_box: int) -> np.ndarray:
    """Center-crop a 2D image or stack on the last two axes."""
    final_box = int(final_box)
    if final_box <= 0:
        return arr
    if arr.shape[-2] == final_box and arr.shape[-1] == final_box:
        return arr.astype(np.float32, copy=False)
    ny, nx = arr.shape[-2], arr.shape[-1]
    if final_box > ny or final_box > nx:
        raise ValueError(f"Cannot crop {ny}x{nx} array to {final_box}x{final_box}")
    y0 = (ny - final_box) // 2
    x0 = (nx - final_box) // 2
    return arr[..., y0:y0 + final_box, x0:x0 + final_box].astype(np.float32, copy=True)


def make_cistem_work_config(cfg: SimConfig) -> Tuple[SimConfig, int, int]:
    """Return the internal padded cisTEM-like config, final box, and padding.

    In simulate.cpp, solvent is generated and propagated in a larger solvent/FFT
    image and only later PadToWantedSize() crops to the requested output.  This
    direct-slab Python code keeps cfg.box as the requested output size, then uses
    a larger working box for explicit water so water clipping/tapering occurs in
    the guard band rather than in the final image.
    """
    final_box = int(cfg.box)
    pad = int(max(0, getattr(cfg, "solvent_padding_pix", 0) or 0))

    # Keep historical behavior for non-solvent calculations unless the user is
    # explicitly using the explicit-water path that needs a guard band.
    if not getattr(cfg, "explicit_water", False):
        pad = 0

    if pad <= 0:
        return cfg, final_box, 0

    work_cfg = replace(cfg, box=final_box + 2 * pad)
    return work_cfg, final_box, pad


def _distance_to_edge_numpy(shape: Tuple[int, int]) -> np.ndarray:
    ny, nx = int(shape[0]), int(shape[1])
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return np.minimum.reduce([yy, xx, ny - 1 - yy, nx - 1 - xx]).astype(np.float32)


def rectangular_cosine_taper_mask_numpy(shape: Tuple[int, int], width: int) -> np.ndarray:
    width = int(width)
    if width <= 0:
        return np.ones(shape, dtype=np.float32)
    width = min(width, max(1, shape[0] // 2), max(1, shape[1] // 2))
    dist = _distance_to_edge_numpy(shape)
    t = np.clip(dist / float(width), 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def edge_reference_mean_2d_numpy(img: np.ndarray, width: int) -> float:
    """Mean in a band just inside the taper, closer to simulate.cpp than edge mean."""
    width = int(max(1, width))
    dist = _distance_to_edge_numpy(img.shape)
    band_width = max(2, width // 4)
    band = (dist >= width) & (dist < width + band_width)
    if not np.any(band):
        return float(img.mean())
    return float(img[band].mean())


def taper_edges_cistem_like_numpy(img: np.ndarray, width: int) -> Tuple[np.ndarray, float]:
    width = int(width)
    if width <= 0:
        out = img.astype(np.float32, copy=False)
        return out, float(out.mean())
    bg = np.float32(edge_reference_mean_2d_numpy(img, width))
    mask = rectangular_cosine_taper_mask_numpy(img.shape, width)
    out = bg + (img.astype(np.float32, copy=False) - bg) * mask
    out = out.astype(np.float32, copy=False)
    return out, float(out.mean())


def gaussian_lowpass_numpy(img: np.ndarray, cutoff_recip_pix: float) -> np.ndarray:
    cutoff = float(cutoff_recip_pix)
    if cutoff <= 0.0:
        return img.astype(np.float32, copy=False)
    _, _, k2 = frequency_grid(img.shape, pixel_size=1.0)
    freq = np.sqrt(k2).astype(np.float32)
    filt = np.exp(-0.5 * (freq / cutoff) ** 2).astype(np.float32)
    return np.fft.ifft2(np.fft.fft2(img.astype(np.float32, copy=False)) * filt).real.astype(np.float32)


def apply_cistem_edge_pipeline_numpy(
    phase_slabs: List[np.ndarray],
    amp_slabs: List[np.ndarray],
    cfg: SimConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Approximate simulate.cpp's post-water taper_edges + sampled-potential mask.

    The sampled mask is formed from the post-water phase slabs before tapering,
    then each slab is tapered to an interior mean and the outside/uncertain region
    is filled back with that mean.  This prevents water-template clipping at the
    solvent box boundary from becoming a visible edge artifact after FFT/CTF.
    """
    if not getattr(cfg, "explicit_water", False) or getattr(cfg, "disable_cistem_edge_pipeline", False):
        return phase_slabs, amp_slabs
    if len(phase_slabs) == 0:
        return phase_slabs, amp_slabs

    sampled = np.sum(np.stack(phase_slabs, axis=0), axis=0).astype(np.float32)
    width = int(max(0, getattr(cfg, "edge_taper_width_pix", 24)))

    phase_means: List[float] = []
    amp_means: List[float] = []
    for i in range(len(phase_slabs)):
        phase_slabs[i], m = taper_edges_cistem_like_numpy(phase_slabs[i], width)
        phase_means.append(float(m))
        amp_slabs[i], ma = taper_edges_cistem_like_numpy(amp_slabs[i], width)
        amp_means.append(float(ma))

    erode_pix = int(max(0, getattr(cfg, "sampled_mask_erode_pix", 7)))
    lowpass = float(getattr(cfg, "sampled_mask_lowpass", 0.05))
    if erode_pix > 0 or lowpass > 0.0:
        mask = (sampled > 1.0e-3).astype(np.float32)
        if erode_pix > 0:
            try:
                from scipy import ndimage  # type: ignore
                structure = np.ones((2 * erode_pix + 1, 2 * erode_pix + 1), dtype=bool)
                mask = ndimage.binary_erosion(mask > 0.5, structure=structure, border_value=0).astype(np.float32)
            except Exception:
                # Safe fallback if scipy.ndimage is unavailable.
                pass
        if lowpass > 0.0:
            mask = np.clip(gaussian_lowpass_numpy(mask, lowpass), 0.0, 1.0).astype(np.float32)
        comp = (1.0 - mask).astype(np.float32)
        for i in range(len(phase_slabs)):
            phase_slabs[i] = (phase_slabs[i] * mask + np.float32(phase_means[i]) * comp).astype(np.float32)
            amp_slabs[i] = (amp_slabs[i] * mask + np.float32(amp_means[i]) * comp).astype(np.float32)

    return phase_slabs, amp_slabs
def simulate_projection_from_slabs(phase_slabs: Sequence[np.ndarray], cfg: SimConfig) -> np.ndarray:
    proj = np.sum(np.stack(phase_slabs, axis=0), axis=0).astype(np.float32)
    ctf = ctf_2d(
        proj.shape,
        cfg.pixel_size,
        cfg.kv,
        cfg.cs_mm,
        cfg.defocus_u,
        cfg.defocus_v,
        cfg.defocus_angle_deg,
        cfg.amplitude_contrast,
        cfg.phase_shift_rad,
    )
    contrast = apply_fourier_filter(proj, ctf)
    img = 1.0 + contrast
    return finalize_detector_image(img, cfg)


def propagate_slabs_cistem_like(
    phase_slabs,
    amp_slabs,
    dz_list,
    cfg: SimConfig,
) -> np.ndarray:
    wave = np.ones_like(phase_slabs[0], dtype=np.complex64)

    for i, (phase, amp, dz) in enumerate(zip(phase_slabs, amp_slabs, dz_list)):
        phase0 = phase.astype(np.float32)

        # 推荐先减掉 phase 的边缘均值，避免 slab 常数背景造成全局相位漂移。
        # 如果你想严格测试原始 phase，也可以先注释掉这一行。
        phase0 = phase0 - edge_mean_2d(phase0)

        amp0 = amp.astype(np.float32)

        transmission = np.exp(-amp0 + 1j * phase0).astype(np.complex64)
        wave *= transmission

        # 这一段就是 slab 间 Fresnel propagation，必须保留
        prop = fresnel_propagator(
            phase0.shape,
            cfg.pixel_size,
            cfg.kv,
            dz,
        )
        wave = np.fft.ifft2(np.fft.fft2(wave) * prop).astype(np.complex64)

    # 所有 slab 之后再经过 objective lens / CTF
    lens = cistem_complex_lens_transfer(
        wave.shape,
        cfg.pixel_size,
        cfg.kv,
        cfg.cs_mm,
        cfg.defocus_u,
        cfg.defocus_v,
        cfg.defocus_angle_deg,
        cfg.phase_shift_rad,
    )

    image_wave = np.fft.ifft2(np.fft.fft2(wave) * lens).astype(np.complex64)
    img = image_wave.real ** 2 + image_wave.imag ** 2

    return finalize_detector_image(img.astype(np.float32), cfg)

# -----------------------------
# I/O and CLI
# -----------------------------

def save_image_or_volume(path: str | Path, data: np.ndarray, pixel_size: float) -> None:
    path = Path(path)
    if path.suffix.lower() == ".npy" or mrcfile is None:
        np.save(path if path.suffix.lower() == ".npy" else path.with_suffix(".npy"), data.astype(np.float32))
        return
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = pixel_size

def prepare_phase_amp_slabs(
    vol: np.ndarray,
    atoms: List[Atom],
    cfg: SimConfig
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    phase_slabs, amp_slabs, dz_list = volume_to_slabs(vol, cfg.n_slices, cfg.pixel_size)

    # Radiation damage on protein/projected specimen only.
    # This is the recommended default.
    if getattr(cfg, "radiation_damage", False) and getattr(cfg, "radiation_damage_where", "protein") in {"protein", "all"}:
        if cfg.verbose:
            if getattr(cfg, "radiation_damage_where", "protein") == "all" and cfg.explicit_water:
                print("Applying cisTEM-safe radiation-damage filter before explicit water; water is not post-filtered.")
            else:
                print("Applying radiation-damage exposure filter to protein slabs...")
        phase_slabs = apply_radiation_damage_to_slabs(phase_slabs, cfg)

    if cfg.explicit_water:
        if cfg.solvent:
            raise ValueError(
                "Do not use --solvent and --explicit-water together. "
                "--solvent is mean solvent; --explicit-water is explicit solvent. "
                "Using both double-counts water."
            )

        if cfg.verbose:
            print("Generating explicit waters with 3D protein exclusion...")

        water_coords = generate_explicit_water_coords(
            cfg,
            atoms=atoms,
            exclude_from_atoms=True,
        )

        if cfg.verbose:
            print("Precomputing projected water templates...")
        templates = precompute_projected_water_templates(cfg)

        if cfg.water_soft_weight:
            distance_2d = nearest_atom_distance_2d(atoms, cfg, max_r_a=9.0)
        else:
            distance_2d = None

        phase_slabs, amp_slabs = fill_water_potential_python(
            phase_slabs,
            amp_slabs,
            water_coords,
            distance_2d,
            dz_list,
            cfg,
            templates,
        )
        phase_slabs, amp_slabs = apply_cistem_edge_pipeline_numpy(phase_slabs, amp_slabs, cfg)

    # Do not Fourier-filter protein+explicit-water after water insertion.
    # simulate.cpp applies the 2D exposure filter before fill_water_potential();
    # filtering the already-discretized water field is the source of grid artifacts.

    return phase_slabs, amp_slabs, dz_list

def simulate_movie_from_volume(
    vol: np.ndarray,
    atoms: List[Atom],
    cfg: SimConfig,
) -> np.ndarray:
    """
    Per-frame movie simulation.

    For each frame:
        protein base slabs
        -> radiation damage filter for [D0, D1]
        -> optional water shaking
        -> add explicit water
        -> projection or multislice
        -> optional Poisson
        -> accumulate

    If cfg.save_frames is True:
        return stack with shape (n_frames, ny, nx)
    else:
        return summed or averaged image with shape (ny, nx)
    """
    n_frames = max(1, int(getattr(cfg, "number_of_frames", 1)))

    dose_per_frame = getattr(cfg, "dose_per_frame_e_per_a2", None)
    if dose_per_frame is None:
        dose_per_frame = float(cfg.dose_e_per_a2) / float(n_frames)
    else:
        dose_per_frame = float(dose_per_frame)

    pre = float(getattr(cfg, "pre_exposure_e_per_a2", 0.0))

    if cfg.verbose:
        print(
            f"Per-frame simulation: frames={n_frames}, "
            f"dose_per_frame={dose_per_frame:.4f} e-/A^2, "
            f"pre_exposure={pre:.4f} e-/A^2"
        )

    # Base protein slabs only. No radiation damage and no explicit water yet.
    base_phase_slabs, base_amp_slabs, dz_list = volume_to_slabs(
        vol,
        cfg.n_slices,
        cfg.pixel_size,
    )

    # Generate water once. Do not regenerate per frame.
    water_cache = prepare_water_cache(atoms, cfg)

    frames: List[np.ndarray] = []

    # Important: Poisson dose should be per-frame dose inside each frame.
    original_total_dose = float(cfg.dose_e_per_a2)

    for iframe in range(n_frames):
        d0 = pre + iframe * dose_per_frame
        d1 = d0 + dose_per_frame

        if cfg.verbose:
            print(
                f"Frame {iframe + 1}/{n_frames}: "
                f"exposure {d0:.3f} -> {d1:.3f} e-/A^2"
            )

        phase_slabs, amp_slabs = prepare_frame_slabs_from_base(
            base_phase_slabs,
            base_amp_slabs,
            dz_list,
            cfg,
            water_cache,
            exposure_start_e_per_a2=d0,
            exposure_end_e_per_a2=d1,
            dose_per_frame_e_per_a2=dose_per_frame,
            iframe=iframe,
        )

        # For Poisson, each frame should use dose_per_frame, not total dose.
        cfg.dose_e_per_a2 = dose_per_frame

        if cfg.mode == "projection":
            img = simulate_projection_from_slabs(phase_slabs, cfg)
        elif cfg.mode == "multislice":
            img = propagate_slabs_cistem_like(
                phase_slabs,
                amp_slabs,
                dz_list,
                cfg,
            )
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

        frames.append(img.astype(np.float32, copy=False))

    cfg.dose_e_per_a2 = original_total_dose

    stack = np.stack(frames, axis=0).astype(np.float32)

    if getattr(cfg, "save_frames", False):
        return stack

    summed = stack.sum(axis=0).astype(np.float32)

    if getattr(cfg, "normalize_frame_sum", True):
        summed /= float(n_frames)

    return summed



# -----------------------------
# PyTorch/GPU implementation
# -----------------------------

def torch_device_from_cfg(cfg: SimConfig) -> torch.device:
    dev = getattr(cfg, "device", "cuda")
    if dev == "cuda" and not torch.cuda.is_available():
        if getattr(cfg, "verbose", False):
            print("CUDA is not available; falling back to CPU torch.")
        dev = "cpu"
    return torch.device(dev)


def np_to_torch(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def torch_to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def scattering_tensors(device: torch.device):
    a = torch.as_tensor(SCATTERING_A, device=device, dtype=torch.float32)
    b = torch.as_tensor(SCATTERING_B, device=device, dtype=torch.float32)
    z = torch.as_tensor(ATOMIC_NUMBER, device=device, dtype=torch.float32)
    return a, b, z



class TorchSimCache:
    """Small per-run cache for fixed-shape Fourier grids and filters.

    The tensors here are deterministic functions of shape, pixel size and
    microscope parameters.  Reusing them avoids rebuilding grids/CTFs/Fresnel
    propagators in every frame and slab.
    """

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.full_grids: Dict[Tuple[Tuple[int, int], float], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.rfft_grids: Dict[Tuple[Tuple[int, int], float], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.dose_ne_rfft: Dict[Tuple[Tuple[int, int], float, float], torch.Tensor] = {}
        self.dose_filters_rfft: Dict[Tuple[Tuple[int, int], float, float, float, float, bool], torch.Tensor] = {}
        self.ctf_rfft_filters: Dict[Tuple[Tuple[int, int], float, float, float, float, float, float, float, float, float], torch.Tensor] = {}
        self.ctf_full_filters: Dict[Tuple[Tuple[int, int], float, float, float, float, float, float, float, float, float], torch.Tensor] = {}
        self.lens_full_filters: Dict[Tuple[Tuple[int, int], float, float, float, float, float, float, float], torch.Tensor] = {}
        self.fresnel_full_filters: Dict[Tuple[Tuple[int, int], float, float, float], torch.Tensor] = {}
        self.dqe_rfft_filters: Dict[Tuple[Tuple[int, int], float, bool], torch.Tensor] = {}
        self.dqe_full_filters: Dict[Tuple[Tuple[int, int], float, bool], torch.Tensor] = {}
        self.gaussian_lowpass_rfft_filters: Dict[Tuple[Tuple[int, int], float], torch.Tensor] = {}
        self.edge_distance_maps: Dict[Tuple[int, int], torch.Tensor] = {}
        self.taper_masks: Dict[Tuple[Tuple[int, int], int], torch.Tensor] = {}
        self.edge_band_masks: Dict[Tuple[Tuple[int, int], int], Tuple[torch.Tensor, int]] = {}
        self.inelastic_lorentzian_rfft_filters: Dict[Tuple[Tuple[int, int], float], torch.Tensor] = {}
        self.radial_bins_rfft_cache: Dict[Tuple[Tuple[int, int], int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def full_grid(self, shape: Tuple[int, int], pixel_size: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(pixel_size))
        cached = self.full_grids.get(key)
        if cached is not None:
            return cached
        ny, nx = shape
        fx = torch.fft.fftfreq(nx, d=float(pixel_size), device=self.device)
        fy = torch.fft.fftfreq(ny, d=float(pixel_size), device=self.device)
        ky, kx = torch.meshgrid(fy, fx, indexing="ij")
        k2 = (kx * kx + ky * ky).to(torch.float32)
        out = (kx.to(torch.float32), ky.to(torch.float32), k2)
        self.full_grids[key] = out
        return out

    def rfft_grid(self, shape: Tuple[int, int], pixel_size: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(pixel_size))
        cached = self.rfft_grids.get(key)
        if cached is not None:
            return cached
        ny, nx = shape
        fx = torch.fft.rfftfreq(nx, d=float(pixel_size), device=self.device)
        fy = torch.fft.fftfreq(ny, d=float(pixel_size), device=self.device)
        ky, kx = torch.meshgrid(fy, fx, indexing="ij")
        k2 = (kx * kx + ky * ky).to(torch.float32)
        out = (kx.to(torch.float32), ky.to(torch.float32), k2)
        self.rfft_grids[key] = out
        return out

    def critical_dose_rfft(self, shape: Tuple[int, int], pixel_size: float, kv: float) -> torch.Tensor:
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(pixel_size), float(kv))
        cached = self.dose_ne_rfft.get(key)
        if cached is not None:
            return cached
        _, _, k2 = self.rfft_grid(shape, pixel_size)
        freq = torch.sqrt(k2).to(torch.float32)
        ne = critical_exposure_grant_grigorieff_torch(freq, kv=float(kv)).to(torch.float32)
        self.dose_ne_rfft[key] = ne
        return ne

    def dose_filter_rfft(
        self,
        shape: Tuple[int, int],
        pixel_size: float,
        kv: float,
        exposure_start_e_per_a2: float,
        exposure_end_e_per_a2: float,
        average_over_interval: bool = False,
    ) -> torch.Tensor:
        shape = (int(shape[0]), int(shape[1]))
        d0 = float(exposure_start_e_per_a2)
        d1 = float(exposure_end_e_per_a2)
        if d1 < d0:
            raise ValueError("exposure_end_e_per_a2 must be >= exposure_start_e_per_a2")
        key = (shape, float(pixel_size), float(kv), d0, d1, bool(average_over_interval))
        cached = self.dose_filters_rfft.get(key)
        if cached is not None:
            return cached
        if d1 == d0:
            filt = torch.ones((shape[0], shape[1] // 2 + 1), device=self.device, dtype=torch.float32)
            self.dose_filters_rfft[key] = filt
            return filt
        ne = self.critical_dose_rfft(shape, pixel_size, kv)
        if average_over_interval:
            denom = max(d1, 1.0e-12)
            filt = (2.0 * ne / denom) * (torch.exp(-0.5 * d0 / ne) - torch.exp(-0.5 * d1 / ne))
        else:
            filt = torch.exp(-0.5 * d1 / ne)
        filt = filt.to(torch.float32)
        filt[0, 0] = 1.0
        self.dose_filters_rfft[key] = filt
        return filt

    def ctf_rfft(
        self,
        shape: Tuple[int, int],
        pixel_size: float,
        kv: float,
        cs_mm: float,
        defocus_u: float,
        defocus_v: Optional[float] = None,
        defocus_angle_deg: float = 0.0,
        amplitude_contrast: float = 0.07,
        phase_shift_rad: float = 0.0,
    ) -> torch.Tensor:
        if defocus_v is None:
            defocus_v = defocus_u
        shape = (int(shape[0]), int(shape[1]))
        key = (
            shape,
            float(pixel_size),
            float(kv),
            float(cs_mm),
            float(defocus_u),
            float(defocus_v),
            float(defocus_angle_deg),
            float(amplitude_contrast),
            float(phase_shift_rad),
            1.0,
        )
        cached = self.ctf_rfft_filters.get(key)
        if cached is not None:
            return cached
        lam = electron_wavelength_angstrom(float(kv))
        cs_a = float(cs_mm) * 1e7
        kx, ky, k2 = self.rfft_grid(shape, pixel_size)
        theta = torch.atan2(ky, kx) - math.radians(float(defocus_angle_deg))
        defocus = 0.5 * (float(defocus_u) + float(defocus_v)) + 0.5 * (float(defocus_u) - float(defocus_v)) * torch.cos(2.0 * theta)
        chi = math.pi * lam * defocus * k2 - 0.5 * math.pi * cs_a * (lam ** 3) * (k2 ** 2) + float(phase_shift_rad)
        amp = float(amplitude_contrast)
        filt = (-(max(0.0, 1.0 - amp * amp) ** 0.5) * torch.sin(chi) - amp * torch.cos(chi)).to(torch.float32)
        self.ctf_rfft_filters[key] = filt
        return filt

    def ctf_full(
        self,
        shape: Tuple[int, int],
        pixel_size: float,
        kv: float,
        cs_mm: float,
        defocus_u: float,
        defocus_v: Optional[float] = None,
        defocus_angle_deg: float = 0.0,
        amplitude_contrast: float = 0.07,
        phase_shift_rad: float = 0.0,
    ) -> torch.Tensor:
        if defocus_v is None:
            defocus_v = defocus_u
        shape = (int(shape[0]), int(shape[1]))
        key = (
            shape,
            float(pixel_size),
            float(kv),
            float(cs_mm),
            float(defocus_u),
            float(defocus_v),
            float(defocus_angle_deg),
            float(amplitude_contrast),
            float(phase_shift_rad),
            0.0,
        )
        cached = self.ctf_full_filters.get(key)
        if cached is not None:
            return cached
        lam = electron_wavelength_angstrom(float(kv))
        cs_a = float(cs_mm) * 1e7
        kx, ky, k2 = self.full_grid(shape, pixel_size)
        theta = torch.atan2(ky, kx) - math.radians(float(defocus_angle_deg))
        defocus = 0.5 * (float(defocus_u) + float(defocus_v)) + 0.5 * (float(defocus_u) - float(defocus_v)) * torch.cos(2.0 * theta)
        chi = math.pi * lam * defocus * k2 - 0.5 * math.pi * cs_a * (lam ** 3) * (k2 ** 2) + float(phase_shift_rad)
        amp = float(amplitude_contrast)
        filt = (-(max(0.0, 1.0 - amp * amp) ** 0.5) * torch.sin(chi) - amp * torch.cos(chi)).to(torch.float32)
        self.ctf_full_filters[key] = filt
        return filt

    def lens_full(
        self,
        shape: Tuple[int, int],
        pixel_size: float,
        kv: float,
        cs_mm: float,
        defocus_u: float,
        defocus_v: Optional[float],
        defocus_angle_deg: float,
        phase_shift_rad: float,
    ) -> torch.Tensor:
        dv = defocus_u if defocus_v is None else defocus_v
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(pixel_size), float(kv), float(cs_mm), float(defocus_u), float(dv), float(defocus_angle_deg), float(phase_shift_rad))
        cached = self.lens_full_filters.get(key)
        if cached is not None:
            return cached
        h_real = self.ctf_full(shape, pixel_size, kv, cs_mm, defocus_u, dv, defocus_angle_deg, 1.0, phase_shift_rad)
        h_imag = self.ctf_full(shape, pixel_size, kv, cs_mm, defocus_u, dv, defocus_angle_deg, 0.0, phase_shift_rad)
        lens = torch.complex(h_real, h_imag).to(torch.complex64)
        self.lens_full_filters[key] = lens
        return lens

    def fresnel_full(self, shape: Tuple[int, int], pixel_size: float, kv: float, dz_angstrom: float) -> torch.Tensor:
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(pixel_size), float(kv), float(dz_angstrom))
        cached = self.fresnel_full_filters.get(key)
        if cached is not None:
            return cached
        lam = electron_wavelength_angstrom(float(kv))
        _, _, k2 = self.full_grid(shape, pixel_size)
        phase = math.pi * lam * float(dz_angstrom) * k2
        prop = torch.exp(1j * phase).to(torch.complex64)
        self.fresnel_full_filters[key] = prop
        return prop

    def dqe_rfft(self, shape: Tuple[int, int], pixel_size: float, root: bool = True) -> torch.Tensor:
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(pixel_size), bool(root))
        cached = self.dqe_rfft_filters.get(key)
        if cached is not None:
            return cached
        _, _, k2 = self.rfft_grid(shape, pixel_size)
        freq = torch.sqrt(k2).to(torch.float32)
        out = torch.zeros_like(freq, dtype=torch.float32)
        for a, b, c in zip(DQE_A, DQE_B, DQE_C):
            out = out + float(a) * torch.exp(-((freq - float(b)) ** 2) / (2.0 * float(c) * float(c)))
        out = torch.clamp(out, min=0.0)
        out = out / torch.clamp(out.max(), min=1.0e-12)
        if root:
            out = torch.sqrt(out)
        out = out.to(torch.float32)
        self.dqe_rfft_filters[key] = out
        return out

    def dqe_full(self, shape: Tuple[int, int], pixel_size: float, root: bool = True) -> torch.Tensor:
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(pixel_size), bool(root))
        cached = self.dqe_full_filters.get(key)
        if cached is not None:
            return cached
        _, _, k2 = self.full_grid(shape, pixel_size)
        freq = torch.sqrt(k2).to(torch.float32)
        out = torch.zeros_like(freq, dtype=torch.float32)
        for a, b, c in zip(DQE_A, DQE_B, DQE_C):
            out = out + float(a) * torch.exp(-((freq - float(b)) ** 2) / (2.0 * float(c) * float(c)))
        out = torch.clamp(out, min=0.0)
        out = out / torch.clamp(out.max(), min=1.0e-12)
        if root:
            out = torch.sqrt(out)
        out = out.to(torch.float32)
        self.dqe_full_filters[key] = out
        return out

    def gaussian_lowpass_rfft(self, shape: Tuple[int, int], cutoff_recip_pix: float) -> torch.Tensor:
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(cutoff_recip_pix))
        cached = self.gaussian_lowpass_rfft_filters.get(key)
        if cached is not None:
            return cached
        cutoff = float(cutoff_recip_pix)
        if cutoff <= 0.0:
            filt = torch.ones((shape[0], shape[1] // 2 + 1), device=self.device, dtype=torch.float32)
        else:
            _, _, k2 = self.rfft_grid(shape, pixel_size=1.0)
            freq = torch.sqrt(k2).to(torch.float32)
            filt = torch.exp(-0.5 * (freq / cutoff) ** 2).to(torch.float32)
        self.gaussian_lowpass_rfft_filters[key] = filt
        return filt

    def edge_distance(self, shape: Tuple[int, int]) -> torch.Tensor:
        shape = (int(shape[0]), int(shape[1]))
        cached = self.edge_distance_maps.get(shape)
        if cached is not None:
            return cached
        ny, nx = shape
        y = torch.arange(ny, device=self.device, dtype=torch.float32)
        x = torch.arange(nx, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.minimum(torch.minimum(yy, xx), torch.minimum((ny - 1) - yy, (nx - 1) - xx)).to(torch.float32)
        self.edge_distance_maps[shape] = dist
        return dist

    def taper_mask(self, shape: Tuple[int, int], width: int) -> torch.Tensor:
        shape = (int(shape[0]), int(shape[1]))
        width = int(width)
        key = (shape, width)
        cached = self.taper_masks.get(key)
        if cached is not None:
            return cached
        if width <= 0:
            mask = torch.ones(shape, dtype=torch.float32, device=self.device)
        else:
            width = min(width, max(1, shape[0] // 2), max(1, shape[1] // 2))
            dist = self.edge_distance(shape)
            t = torch.clamp(dist / float(width), 0.0, 1.0)
            mask = (0.5 - 0.5 * torch.cos(math.pi * t)).to(torch.float32)
        self.taper_masks[key] = mask
        return mask

    def edge_band_mask(self, shape: Tuple[int, int], width: int) -> Tuple[torch.Tensor, int]:
        shape = (int(shape[0]), int(shape[1]))
        width = int(max(1, width))
        key = (shape, width)
        cached = self.edge_band_masks.get(key)
        if cached is not None:
            return cached
        band_width = max(2, width // 4)
        dist = self.edge_distance(shape)
        band = ((dist >= float(width)) & (dist < float(width + band_width))).contiguous()
        # One-time metadata sync during cache creation. Avoid repeated syncs in the frame loop.
        count = int(band.sum().detach().cpu().item())
        out = (band, count)
        self.edge_band_masks[key] = out
        return out


def get_torch_sim_cache(cfg: SimConfig, device: torch.device) -> TorchSimCache:
    device = torch.device(device)
    cache = getattr(cfg, "_torch_sim_cache", None)
    if cache is None or getattr(cache, "device", None) != device:
        cache = TorchSimCache(device)
        setattr(cfg, "_torch_sim_cache", cache)
    return cache


def frequency_grid_rfft_torch(shape: Tuple[int, int], pixel_size: float, device: torch.device):
    cache = TorchSimCache(device)
    return cache.rfft_grid(shape, pixel_size)


def as_slab_stack_torch(slabs: Sequence[torch.Tensor]) -> torch.Tensor:
    if isinstance(slabs, torch.Tensor):
        if slabs.ndim == 2:
            return slabs[None, ...].to(torch.float32)
        return slabs.to(torch.float32)
    return torch.stack([s.to(torch.float32) for s in slabs], dim=0)


def slab_stack_to_list_torch(stack: torch.Tensor) -> List[torch.Tensor]:
    return [s.contiguous() for s in torch.unbind(stack.to(torch.float32), dim=0)]


def edge_mean_2d_stack_torch(stack: torch.Tensor, width: int = 4) -> torch.Tensor:
    x = as_slab_stack_torch(stack)
    w = min(int(width), x.shape[-2] // 4, x.shape[-1] // 4)
    if w <= 0:
        return x.mean(dim=(-2, -1))
    edges = torch.cat([
        x[..., :w, :].reshape(x.shape[0], -1),
        x[..., -w:, :].reshape(x.shape[0], -1),
        x[..., :, :w].reshape(x.shape[0], -1),
        x[..., :, -w:].reshape(x.shape[0], -1),
    ], dim=1)
    return edges.mean(dim=1).to(torch.float32)


def apply_fourier_filter_real_rfft_torch(img: torch.Tensor, filt_rfft: torch.Tensor) -> torch.Tensor:
    x = img.to(torch.float32)
    return torch.fft.irfft2(
        torch.fft.rfft2(x, dim=(-2, -1)) * filt_rfft,
        s=tuple(x.shape[-2:]),
        dim=(-2, -1),
    ).to(torch.float32)



def frequency_grid_torch(shape: Tuple[int, int], pixel_size: float, device: torch.device):
    ny, nx = shape
    fx = torch.fft.fftfreq(nx, d=float(pixel_size), device=device)
    fy = torch.fft.fftfreq(ny, d=float(pixel_size), device=device)
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    k2 = kx * kx + ky * ky
    return kx, ky, k2


def edge_mean_2d_torch(img: torch.Tensor, width: int = 4) -> torch.Tensor:
    w = min(width, img.shape[-2] // 4, img.shape[-1] // 4)
    if w <= 0:
        return img.mean()
    edges = torch.cat([
        img[:w, :].reshape(-1),
        img[-w:, :].reshape(-1),
        img[:, :w].reshape(-1),
        img[:, -w:].reshape(-1),
    ])
    return edges.mean()


def center_crop_torch(arr: torch.Tensor, final_box: int) -> torch.Tensor:
    """Center-crop a 2D tensor or stack on the last two axes."""
    final_box = int(final_box)
    if final_box <= 0:
        return arr
    if arr.shape[-2] == final_box and arr.shape[-1] == final_box:
        return arr.to(torch.float32)
    ny, nx = int(arr.shape[-2]), int(arr.shape[-1])
    if final_box > ny or final_box > nx:
        raise ValueError(f"Cannot crop {ny}x{nx} tensor to {final_box}x{final_box}")
    y0 = (ny - final_box) // 2
    x0 = (nx - final_box) // 2
    return arr[..., y0:y0 + final_box, x0:x0 + final_box].contiguous().to(torch.float32)


def _distance_to_edge_torch(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    ny, nx = int(shape[0]), int(shape[1])
    y = torch.arange(ny, device=device, dtype=torch.float32)
    x = torch.arange(nx, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.minimum(torch.minimum(yy, xx), torch.minimum((ny - 1) - yy, (nx - 1) - xx))


def rectangular_cosine_taper_mask_torch(shape: Tuple[int, int], width: int, device: torch.device) -> torch.Tensor:
    cache = TorchSimCache(device)
    return cache.taper_mask(shape, int(width))


def edge_reference_mean_2d_torch(img: torch.Tensor, width: int) -> torch.Tensor:
    width = int(max(1, width))
    cache = get_torch_sim_cache(SimConfig(device=str(img.device)), img.device)
    band, count = cache.edge_band_mask(tuple(img.shape), width)
    if count <= 0:
        return img.mean()
    return img[band].mean()


def taper_edges_cistem_like_torch(img: torch.Tensor, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    width = int(width)
    x = img.to(torch.float32)
    if width <= 0:
        return x, x.mean()
    bg = edge_reference_mean_2d_torch(x, width).to(torch.float32)
    mask = rectangular_cosine_taper_mask_torch(tuple(x.shape), width, x.device)
    out = bg + (x - bg) * mask
    out = out.to(torch.float32)
    return out, out.mean()


def gaussian_lowpass_torch(img: torch.Tensor, cutoff_recip_pix: float) -> torch.Tensor:
    cutoff = float(cutoff_recip_pix)
    x = img.to(torch.float32)
    if cutoff <= 0.0:
        return x
    cache = get_torch_sim_cache(SimConfig(device=str(x.device)), x.device)
    filt = cache.gaussian_lowpass_rfft(tuple(x.shape[-2:]), cutoff)
    return apply_fourier_filter_real_rfft_torch(x, filt)


def erode_mask_square_torch(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    x = mask.to(torch.float32)
    if radius <= 0:
        return x
    import torch.nn.functional as F
    inv = (1.0 - x)[None, None, :, :]
    eroded = 1.0 - F.max_pool2d(inv, kernel_size=2 * radius + 1, stride=1, padding=radius)[0, 0]
    return torch.clamp(eroded, 0.0, 1.0).to(torch.float32)




def critical_exposure_grant_grigorieff_torch(
    freq_a_inv: torch.Tensor,
    kv: float = 300.0,
) -> torch.Tensor:
    f = freq_a_inv.to(torch.float32)
    ne = torch.empty_like(f)
    positive = f > 1.0e-6
    ne[positive] = 0.24499 * torch.pow(f[positive], -1.6649) + 2.8141
    ne[~positive] = 1.0e9
    ne = ne * float(electron_dose_voltage_scaling(kv))
    return ne


def dose_filter_interval_torch(
    shape: Tuple[int, int],
    pixel_size: float,
    exposure_start_e_per_a2: float,
    exposure_end_e_per_a2: float,
    device: torch.device,
    average_over_interval: bool = False,
    kv: float = 300.0,
) -> torch.Tensor:
    cache = get_torch_sim_cache(SimConfig(device=str(device), pixel_size=float(pixel_size), kv=float(kv)), device)
    return cache.dose_filter_rfft(
        shape,
        pixel_size,
        kv,
        exposure_start_e_per_a2,
        exposure_end_e_per_a2,
        average_over_interval=average_over_interval,
    )


def apply_exposure_filter_2d_torch(
    img: torch.Tensor,
    pixel_size: float,
    exposure_start_e_per_a2: float,
    exposure_end_e_per_a2: float,
    modify_signal: int = 0,
    subtract_edge_mean: bool = True,
    kv: float = 300.0,
    cfg: Optional[SimConfig] = None,
) -> torch.Tensor:
    x = img.to(torch.float32)
    if x.ndim == 2:
        stack = x[None, :, :]
        squeeze = True
    else:
        stack = x
        squeeze = False
    if subtract_edge_mean:
        bg = edge_mean_2d_stack_torch(stack)
        work = stack - bg[:, None, None]
    else:
        bg = torch.zeros((stack.shape[0],), device=stack.device, dtype=torch.float32)
        work = stack
    cache = get_torch_sim_cache(cfg if cfg is not None else SimConfig(device=str(stack.device), pixel_size=float(pixel_size), kv=float(kv)), stack.device)
    filt = cache.dose_filter_rfft(
        tuple(work.shape[-2:]),
        pixel_size,
        kv,
        exposure_start_e_per_a2,
        exposure_end_e_per_a2,
        average_over_interval=False,
    )
    if modify_signal == 1:
        filt = 1.0 - (1.0 - filt) / (1.0 + filt)
    elif modify_signal == 2:
        filt = torch.sqrt(torch.clamp(filt, min=0.0))
    out = apply_fourier_filter_real_rfft_torch(work, filt) + bg[:, None, None]
    return out[0].to(torch.float32) if squeeze else out.to(torch.float32)


def apply_radiation_damage_to_slabs_torch(
    phase_slabs: List[torch.Tensor],
    cfg: SimConfig,
    exposure_start_e_per_a2: Optional[float] = None,
    exposure_end_e_per_a2: Optional[float] = None,
) -> List[torch.Tensor]:
    if exposure_start_e_per_a2 is None:
        exposure_start_e_per_a2 = float(getattr(cfg, "pre_exposure_e_per_a2", 0.0))
    if exposure_end_e_per_a2 is None:
        exposure_end_e_per_a2 = exposure_start_e_per_a2 + float(cfg.dose_e_per_a2)
    if len(phase_slabs) == 0:
        return phase_slabs
    stack = as_slab_stack_torch(phase_slabs)
    filtered = apply_exposure_filter_2d_torch(
        stack,
        cfg.pixel_size,
        exposure_start_e_per_a2,
        exposure_end_e_per_a2,
        modify_signal=int(getattr(cfg, "exposure_filter_modify_signal", 0)),
        subtract_edge_mean=True,
        kv=float(getattr(cfg, "kv", 300.0)),
        cfg=cfg,
    )
    return slab_stack_to_list_torch(filtered)


def ctf_2d_torch(
    shape: Tuple[int, int], pixel_size: float, kv: float, cs_mm: float,
    defocus_u: float, defocus_v: Optional[float] = None, defocus_angle_deg: float = 0.0,
    amplitude_contrast: float = 0.07, phase_shift_rad: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = get_torch_sim_cache(SimConfig(device=str(device), pixel_size=float(pixel_size), kv=float(kv)), device)
    return cache.ctf_full(shape, pixel_size, kv, cs_mm, defocus_u, defocus_v, defocus_angle_deg, amplitude_contrast, phase_shift_rad)


def ctf_2d_rfft_torch(
    shape: Tuple[int, int], pixel_size: float, kv: float, cs_mm: float,
    defocus_u: float, defocus_v: Optional[float] = None, defocus_angle_deg: float = 0.0,
    amplitude_contrast: float = 0.07, phase_shift_rad: float = 0.0,
    device: Optional[torch.device] = None,
    cfg: Optional[SimConfig] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = get_torch_sim_cache(cfg if cfg is not None else SimConfig(device=str(device), pixel_size=float(pixel_size), kv=float(kv)), device)
    return cache.ctf_rfft(shape, pixel_size, kv, cs_mm, defocus_u, defocus_v, defocus_angle_deg, amplitude_contrast, phase_shift_rad)


def cistem_complex_lens_transfer_torch(
    shape: Tuple[int, int], pixel_size: float, kv: float, cs_mm: float,
    defocus_u: float, defocus_v: Optional[float] = None, defocus_angle_deg: float = 0.0,
    phase_shift_rad: float = 0.0, device: Optional[torch.device] = None,
    cfg: Optional[SimConfig] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = get_torch_sim_cache(cfg if cfg is not None else SimConfig(device=str(device), pixel_size=float(pixel_size), kv=float(kv)), device)
    return cache.lens_full(shape, pixel_size, kv, cs_mm, defocus_u, defocus_v, defocus_angle_deg, phase_shift_rad)


def fresnel_propagator_torch(shape: Tuple[int, int], pixel_size: float, kv: float, dz_angstrom: float, device: torch.device, cfg: Optional[SimConfig] = None) -> torch.Tensor:
    cache = get_torch_sim_cache(cfg if cfg is not None else SimConfig(device=str(device), pixel_size=float(pixel_size), kv=float(kv)), device)
    return cache.fresnel_full(shape, pixel_size, kv, dz_angstrom)


def dqe_filter_torch(shape: Tuple[int, int], pixel_size: float, device: torch.device, root: bool = True, cfg: Optional[SimConfig] = None) -> torch.Tensor:
    cache = get_torch_sim_cache(cfg if cfg is not None else SimConfig(device=str(device), pixel_size=float(pixel_size)), device)
    return cache.dqe_full(shape, pixel_size, root=root)


def dqe_filter_rfft_torch(shape: Tuple[int, int], pixel_size: float, device: torch.device, root: bool = True, cfg: Optional[SimConfig] = None) -> torch.Tensor:
    cache = get_torch_sim_cache(cfg if cfg is not None else SimConfig(device=str(device), pixel_size=float(pixel_size)), device)
    return cache.dqe_rfft(shape, pixel_size, root=root)


def apply_fourier_filter_torch(img: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
    x = img.to(torch.float32)
    # If caller supplied a real-FFT half-plane filter, keep the operation in rfft space.
    if (not torch.is_complex(x)) and filt.shape[-2] == x.shape[-2] and filt.shape[-1] == x.shape[-1] // 2 + 1:
        return apply_fourier_filter_real_rfft_torch(x, filt)
    return torch.fft.ifft2(torch.fft.fft2(x) * filt).real.to(torch.float32)


def voxel_integrated_potential_torch(
    x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor,
    atom_index: int, bfactor: float, lead_term: float, device: torch.device,
) -> torch.Tensor:
    a, b, _ = scattering_tensors(device)
    out = torch.zeros((z1.numel(), y1.numel(), x1.numel()), device=device, dtype=torch.float32)
    for i in range(5):
        b_total = b[atom_index, i] + float(bfactor)
        if float(b_total.detach().cpu()) <= 0:
            continue
        bplus = torch.sqrt(torch.tensor(4.0 * math.pi * math.pi, device=device, dtype=torch.float32) / b_total)
        dx = torch.erf(bplus * x2) - torch.erf(bplus * x1)
        dy = torch.erf(bplus * y2) - torch.erf(bplus * y1)
        dz = torch.erf(bplus * z2) - torch.erf(bplus * z1)
        out = out + (a[atom_index, i] * float(lead_term)) * dz[:, None, None] * dy[None, :, None] * dx[None, None, :]
    return out.to(torch.float32)


def make_scattering_volume_torch(atoms: List[Atom], cfg: SimConfig) -> torch.Tensor:
    device = torch_device_from_cfg(cfg)
    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    vol = torch.zeros((n, n, n), device=device, dtype=torch.float32)
    lam = electron_wavelength_angstrom(cfg.kv)
    lead_term = cfg.bond_scaling * lam / 8.0 / (ps * ps)
    half = n / 2.0
    for i_atom, atom in enumerate(atoms):
        ai = ATOM_INDEX[atom.element]
        x, y, z = atom.xyz / ps + half
        ix, iy, iz = int(round(float(x))), int(round(float(y))), int(round(float(z)))
        bf = complete_bfactor(atom.bfactor, cfg.bfactor_scaling, cfg.min_bfactor)
        r = atom_neighborhood_radius(ps, bf)
        x0, x1i = max(0, ix - r), min(n, ix + r + 1)
        y0, y1i = max(0, iy - r), min(n, iy + r + 1)
        z0, z1i = max(0, iz - r), min(n, iz + r + 1)
        if x0 >= x1i or y0 >= y1i or z0 >= z1i:
            continue
        xs = torch.arange(x0, x1i, device=device, dtype=torch.float32)
        ys = torch.arange(y0, y1i, device=device, dtype=torch.float32)
        zs = torch.arange(z0, z1i, device=device, dtype=torch.float32)
        atom_xyz = atom.xyz.astype(np.float32)
        x1v = ((xs - half) * ps - float(atom_xyz[0])) - 0.5 * ps
        y1v = ((ys - half) * ps - float(atom_xyz[1])) - 0.5 * ps
        z1v = ((zs - half) * ps - float(atom_xyz[2])) - 0.5 * ps
        pot = voxel_integrated_potential_torch(x1v, x1v + ps, y1v, y1v + ps, z1v, z1v + ps, ai, bf, lead_term, device)
        vol[z0:z1i, y0:y1i, x0:x1i] += pot * float(atom.occupancy)
        if cfg.verbose and (i_atom + 1) % 10000 == 0:
            print(f"  torch potential atoms: {i_atom + 1}/{len(atoms)}")
    return vol

def voxel_integrated_projected_potential_torch(
    x1: torch.Tensor,
    x2: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    atom_index: int,
    bfactor: float,
    lead_term: float,
    device: torch.device,
) -> torch.Tensor:
    """2D slab-integrated version of voxel_integrated_potential_torch.

    This computes sum_z potential directly, without materializing a local 3D block.
    It is mathematically equivalent to calculating the same voxel-integrated 3D
    contribution and then summing over the selected z voxels for one slab.
    """
    a, b, _ = scattering_tensors(device)
    out = torch.zeros((y1.numel(), x1.numel()), device=device, dtype=torch.float32)
    for i in range(5):
        b_total = b[atom_index, i] + float(bfactor)
        if float(b_total.detach().cpu()) <= 0:
            continue
        bplus = torch.sqrt(torch.tensor(4.0 * math.pi * math.pi, device=device, dtype=torch.float32) / b_total)
        dx = torch.erf(bplus * x2) - torch.erf(bplus * x1)
        dy = torch.erf(bplus * y2) - torch.erf(bplus * y1)
        dz = torch.erf(bplus * z2) - torch.erf(bplus * z1)
        zsum = dz.sum()
        out = out + (a[atom_index, i] * float(lead_term)) * zsum * dy[:, None] * dx[None, :]
    return out.to(torch.float32)
    
def _cached_atom_template_key(
    atom_index: int,
    sx: int,
    sy: int,
    sz: int,
) -> Tuple[int, int, int, int]:
    return int(atom_index), int(sx), int(sy), int(sz)


def precompute_atom_3d_template_cache_numpy(
    cfg: SimConfig,
    elements: Optional[Sequence[str]] = None,
    subpix_n: int = 5,
    template_radius_pix: int = 7,
) -> Dict[Tuple[int, int, int, int], np.ndarray]:
    """
    Precompute local 3D voxel-integrated atom templates.

    Cache key:
        (atom_index, sx, sy, sz)

    Template shape:
        (2R+1, 2R+1, 2R+1), ordered as z, y, x.

    This assumes cfg.bfactor_scaling == 0, so all atoms of the same element share
    the same effective B-factor:
        bf = 0.25 * cfg.min_bfactor

    The subpixel convention matches round-to-nearest voxel center:
        atom coordinate = integer voxel center + fractional offset
        frac in approximately [-0.5, 0.5)
    """
    if subpix_n <= 0:
        raise ValueError("subpix_n must be positive")
    if template_radius_pix <= 0:
        raise ValueError("template_radius_pix must be positive")

    if elements is None:
        # Protein-heavy default. Add others here if your PDB uses them.
        elements = ("C", "N", "O", "S", "P")

    ps = float(cfg.pixel_size)
    lam = electron_wavelength_angstrom(float(cfg.kv))
    lead_term = float(cfg.bond_scaling) * lam / 8.0 / (ps * ps)

    # bfactor_scaling == 0 path.
    bf = complete_bfactor(
        atom_b=0.0,
        bfactor_scaling=0.0,
        min_bfactor=float(cfg.min_bfactor),
    )

    r = int(template_radius_pix)
    grid = np.arange(-r, r + 1, dtype=np.float64)

    cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    center = (subpix_n - 1) / 2.0

    for elem in elements:
        elem_u = elem.upper()
        if elem_u not in ATOM_INDEX:
            continue

        ai = int(ATOM_INDEX[elem_u])

        for sz in range(subpix_n):
            fz = (sz - center) / float(subpix_n)

            z1 = ((grid - fz) * ps) - 0.5 * ps
            z2 = z1 + ps

            for sy in range(subpix_n):
                fy = (sy - center) / float(subpix_n)

                y1 = ((grid - fy) * ps) - 0.5 * ps
                y2 = y1 + ps

                for sx in range(subpix_n):
                    fx = (sx - center) / float(subpix_n)

                    x1 = ((grid - fx) * ps) - 0.5 * ps
                    x2 = x1 + ps

                    tmpl = voxel_integrated_potential(
                        x1,
                        x2,
                        y1,
                        y2,
                        z1,
                        z2,
                        ai,
                        bf,
                        lead_term,
                    ).astype(np.float32, copy=False)

                    cache[_cached_atom_template_key(ai, sx, sy, sz)] = tmpl

    return cache


def _subpixel_bin_from_fraction_numpy(frac: float, subpix_n: int) -> int:
    """
    Map fractional offset relative to rounded integer pixel center to subpixel bin.

    frac should usually be in [-0.5, 0.5]. We use floor rather than round so
    bins partition the interval evenly.
    """
    idx = int(np.floor((float(frac) + 0.5) * float(subpix_n)))
    if idx < 0:
        return 0
    if idx >= subpix_n:
        return subpix_n - 1
    return idx



def make_phase_amp_slabs_direct_from_atoms_numpy_cached_legacy(
    atoms: List[Atom],
    cfg: SimConfig,
    subpix_n: int = 9,
    template_radius_pix: int = 9,
    elements_for_cache: Optional[Sequence[str]] = None,
    fallback_if_needed: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Cached NumPy direct atom -> phase_slabs path.

    Intended replacement for:
        make_phase_amp_slabs_direct_from_atoms_numpy(atoms, cfg)

    It is fastest and most appropriate when:
        cfg.bfactor_scaling == 0
        main elements are C/N/O/S/P

    If cfg.bfactor_scaling != 0, this function falls back to the non-cached
    direct NumPy version, because atom.bfactor then changes the template shape.

    Parameters
    ----------
    subpix_n
        Number of subpixel bins along x/y/z. Start with 5. Try 7 if you need
        better agreement with the non-cached direct version.
    template_radius_pix
        Local template cutoff radius in pixels. You said your previous multislice
        code used about 7 pixels, so default is 7.
    elements_for_cache
        Optional list of elements to cache. Default caches C/N/O/S/P plus any
        supported elements actually found in atoms.
    fallback_if_needed
        If True, unsupported elements or nonzero bfactor_scaling use the old
        direct function. If False, unsupported elements raise an error.
    """
    # If B-factor varies per atom, this simple cache is no longer exact.
    if abs(float(cfg.bfactor_scaling)) > 1.0e-12:
        if fallback_if_needed:
            if getattr(cfg, "verbose", False):
                print(
                    "Cached direct slabs disabled because cfg.bfactor_scaling != 0; "
                    "falling back to make_phase_amp_slabs_direct_from_atoms_numpy."
                )
            return make_phase_amp_slabs_direct_from_atoms_numpy(atoms, cfg)
        raise ValueError(
            "Cached direct slabs currently assumes cfg.bfactor_scaling == 0. "
            "Use fallback or add B-factor binning."
        )

    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    n_slices = max(1, int(cfg.n_slices))
    half = n / 2.0

    # Same slab partition as np.array_split(vol, n_slices, axis=0), but no vol.
    z_chunks = np.array_split(np.arange(n, dtype=np.int32), n_slices)
    slab_z_starts = np.array([int(c[0]) for c in z_chunks], dtype=np.int32)
    slab_z_ends = np.array([int(c[-1]) + 1 for c in z_chunks], dtype=np.int32)
    dz_list = [float(len(c) * ps) for c in z_chunks]

    phase_slabs = [np.zeros((n, n), dtype=np.float32) for _ in range(n_slices)]
    amp_slabs = [np.zeros((n, n), dtype=np.float32) for _ in range(n_slices)]

    # Cache only elements that are actually present, plus common protein elements.
    present_elements = sorted({a.element.upper() for a in atoms if a.element.upper() in ATOM_INDEX})

    if elements_for_cache is None:
        base_elements = {"C", "N", "O", "S", "P"}
        base_elements.update(present_elements)
        elements_for_cache = tuple(sorted(base_elements))

    cache = precompute_atom_3d_template_cache_numpy(
        cfg,
        elements=elements_for_cache,
        subpix_n=int(subpix_n),
        template_radius_pix=int(template_radius_pix),
    )

    r = int(template_radius_pix)
    full_template_size = 2 * r + 1

    for i_atom, atom in enumerate(atoms):
        elem = atom.element.upper()
        if elem not in ATOM_INDEX:
            if fallback_if_needed:
                # Extremely conservative: if unsupported element appears, fall back
                # for the entire run to preserve behavior.
                if getattr(cfg, "verbose", False):
                    print(
                        f"Cached direct slabs found unsupported element {elem}; "
                        "falling back to non-cached direct function."
                    )
                return make_phase_amp_slabs_direct_from_atoms_numpy(atoms, cfg)
            raise ValueError(f"Unsupported element for cached slabs: {elem}")

        ai = int(ATOM_INDEX[elem])

        # Pixel coordinate in volume frame.
        x_pix, y_pix, z_pix = atom.xyz / ps + half

        ix = int(round(float(x_pix)))
        iy = int(round(float(y_pix)))
        iz = int(round(float(z_pix)))

        # Fraction relative to the rounded voxel center.
        fx = float(x_pix) - ix
        fy = float(y_pix) - iy
        fz = float(z_pix) - iz

        sx = _subpixel_bin_from_fraction_numpy(fx, int(subpix_n))
        sy = _subpixel_bin_from_fraction_numpy(fy, int(subpix_n))
        sz = _subpixel_bin_from_fraction_numpy(fz, int(subpix_n))

        key = _cached_atom_template_key(ai, sx, sy, sz)
        tmpl3d = cache.get(key)
        if tmpl3d is None:
            if fallback_if_needed:
                if getattr(cfg, "verbose", False):
                    print(
                        f"Cached direct slabs missing template for element {elem}; "
                        "falling back to non-cached direct function."
                    )
                return make_phase_amp_slabs_direct_from_atoms_numpy(atoms, cfg)
            raise KeyError(f"Missing cached template key: {key}")

        # Template nominally covers global voxel indices:
        #   x: ix-r ... ix+r
        #   y: iy-r ... iy+r
        #   z: iz-r ... iz+r
        x0 = max(0, ix - r)
        x1i = min(n, ix + r + 1)
        y0 = max(0, iy - r)
        y1i = min(n, iy + r + 1)
        z0 = max(0, iz - r)
        z1i = min(n, iz + r + 1)

        if x0 >= x1i or y0 >= y1i or z0 >= z1i:
            continue

        # Corresponding template slices after clipping at box boundary.
        tx0 = x0 - (ix - r)
        tx1 = tx0 + (x1i - x0)
        ty0 = y0 - (iy - r)
        ty1 = ty0 + (y1i - y0)
        tz0_global = z0 - (iz - r)
        tz1_global = tz0_global + (z1i - z0)

        # Safety check; should not trigger.
        if (
            tx0 < 0 or ty0 < 0 or tz0_global < 0
            or tx1 > full_template_size
            or ty1 > full_template_size
            or tz1_global > full_template_size
        ):
            if fallback_if_needed:
                return make_phase_amp_slabs_direct_from_atoms_numpy(atoms, cfg)
            raise RuntimeError("Template clipping index out of range.")

        # Only loop over slabs intersecting this atom's z support.
        # slab_z_ends are exclusive. Need slabs with:
        #   slab_end > z0 and slab_start < z1i
        s0 = int(np.searchsorted(slab_z_ends, z0, side="right"))
        s1 = int(np.searchsorted(slab_z_starts, z1i - 1, side="right"))

        if s0 >= s1:
            continue

        scale = np.float32(atom.occupancy)

        for s in range(s0, s1):
            zz0 = max(z0, int(slab_z_starts[s]))
            zz1 = min(z1i, int(slab_z_ends[s]))

            if zz0 >= zz1:
                continue

            # Convert global z voxel interval to template z interval.
            tz0 = tz0_global + (zz0 - z0)
            tz1 = tz0 + (zz1 - zz0)

            # Sum template over z inside this slab.
            # tmpl3d order is z, y, x.
            pot2d = tmpl3d[tz0:tz1, ty0:ty1, tx0:tx1].sum(axis=0, dtype=np.float32)

            phase_slabs[s][y0:y1i, x0:x1i] += pot2d * scale

        if getattr(cfg, "verbose", False) and (i_atom + 1) % 10000 == 0:
            print(f"  cached direct slab atoms: {i_atom + 1}/{len(atoms)}")

    return phase_slabs, amp_slabs, dz_list


def _precompute_cached_atom_xy_kernels_torch(
    cfg: SimConfig,
    element_ids: Sequence[int],
    subpix_n: int,
    template_radius_pix: int,
    device: torch.device,
) -> torch.Tensor:
    """Return xy kernels for the cached atom grouped-conv slab builder.

    Shape is [E, subpix_x, subpix_y, gaussian, K, K].  The kernels contain only
    dx*dy.  The per-atom per-slab weights include occupancy, scattering_A,
    lead_term, and the cached z integral, which preserves the separable cached
    template math while avoiding one kernel group per z-overlap interval.
    """
    r = int(template_radius_pix)
    k = 2 * r + 1
    ps = float(cfg.pixel_size)
    bf = complete_bfactor(0.0, 0.0, float(cfg.min_bfactor))
    grid = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    center = (int(subpix_n) - 1) / 2.0

    kernels = torch.empty(
        (len(element_ids), int(subpix_n), int(subpix_n), 5, k, k),
        device=device,
        dtype=torch.float32,
    )

    for epos, ai in enumerate(element_ids):
        for sx in range(int(subpix_n)):
            fx = (float(sx) - center) / float(subpix_n)
            x1 = (grid - float(fx)) * ps - 0.5 * ps
            x2 = x1 + ps
            for sy in range(int(subpix_n)):
                fy = (float(sy) - center) / float(subpix_n)
                y1 = (grid - float(fy)) * ps - 0.5 * ps
                y2 = y1 + ps
                for ig in range(5):
                    b_total = float(SCATTERING_B[int(ai), ig] + bf)
                    if b_total <= 0.0:
                        kernels[epos, sx, sy, ig].zero_()
                        continue
                    bplus = math.sqrt(4.0 * math.pi * math.pi / b_total)
                    dx = torch.erf(float(bplus) * x2) - torch.erf(float(bplus) * x1)
                    dy = torch.erf(float(bplus) * y2) - torch.erf(float(bplus) * y1)
                    kernels[epos, sx, sy, ig] = dy[:, None] * dx[None, :]

    return kernels.contiguous()


def _precompute_cached_atom_z_prefix_numpy(
    cfg: SimConfig,
    element_ids: Sequence[int],
    subpix_n: int,
    template_radius_pix: int,
) -> np.ndarray:
    """Prefix sums of the cached 1D z integrals.

    Shape is [E, subpix_z, gaussian, K+1].  A slab-specific z contribution is
    prefix[..., tz1] - prefix[..., tz0].
    """
    r = int(template_radius_pix)
    k = 2 * r + 1
    ps = float(cfg.pixel_size)
    bf = complete_bfactor(0.0, 0.0, float(cfg.min_bfactor))
    grid = np.arange(-r, r + 1, dtype=np.float64)
    center = (int(subpix_n) - 1) / 2.0

    prefix = np.zeros((len(element_ids), int(subpix_n), 5, k + 1), dtype=np.float32)
    for epos, ai in enumerate(element_ids):
        for sz in range(int(subpix_n)):
            fz = (float(sz) - center) / float(subpix_n)
            z1 = (grid - float(fz)) * ps - 0.5 * ps
            z2 = z1 + ps
            for ig in range(5):
                b_total = float(SCATTERING_B[int(ai), ig] + bf)
                if b_total <= 0.0:
                    continue
                bplus = math.sqrt(4.0 * math.pi * math.pi / b_total)
                dz = (erf(bplus * z2) - erf(bplus * z1)).astype(np.float32, copy=False)
                prefix[epos, sz, ig, 1:] = np.cumsum(dz, dtype=np.float32)
    return prefix


def _select_cached_atom_elements(
    atoms: List[Atom],
    elements_for_cache: Optional[Sequence[str]],
) -> Tuple[Tuple[str, ...], Tuple[int, ...], Dict[int, int]]:
    present = sorted({a.element.upper() for a in atoms if a.element.upper() in ATOM_INDEX})
    if elements_for_cache is None:
        elems = set(present)
        elems.update({"C", "N", "O", "S", "P"})
    else:
        elems = {str(e).upper() for e in elements_for_cache}
        elems.update(present)
    elems = tuple(sorted(e for e in elems if e in ATOM_INDEX))
    element_ids = tuple(int(ATOM_INDEX[e]) for e in elems)
    ai_to_epos = {ai: i for i, ai in enumerate(element_ids)}
    return elems, element_ids, ai_to_epos


def make_phase_amp_slabs_direct_from_atoms_torch_cached_grouped(
    atoms: List[Atom],
    cfg: SimConfig,
    subpix_n: int = 9,
    template_radius_pix: int = 9,
    elements_for_cache: Optional[Sequence[str]] = None,
    fallback_if_needed: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
    """Cached atom -> phase slabs using grouped impulse maps + conv2d.

    This is the GPU replacement for the old cached NumPy template-splat path.
    It keeps the same cached-template assumptions: bfactor_scaling must be zero,
    and the atom template is quantized by element and subpixel bin.  The 3D
    cached template is used in separable form: x/y are handled by grouped conv2d,
    while the slab-specific z integral becomes an impulse-map weight.
    """
    if abs(float(cfg.bfactor_scaling)) > 1.0e-12:
        if fallback_if_needed:
            if getattr(cfg, "verbose", False):
                print(
                    "Torch cached atom slabs disabled because cfg.bfactor_scaling != 0; "
                    "falling back to make_phase_amp_slabs_direct_from_atoms_numpy."
                )
            phase_np, amp_np, dz_list = make_phase_amp_slabs_direct_from_atoms_numpy(atoms, cfg)
            return numpy_slabs_to_torch(phase_np, amp_np, device=torch_device_from_cfg(cfg)) + (dz_list,)
        raise ValueError("Torch cached atom slabs currently assumes cfg.bfactor_scaling == 0.")

    if int(subpix_n) <= 0 or int(subpix_n) % 2 != 1:
        raise ValueError("atom cached subpix_n must be a positive odd integer")
    if int(template_radius_pix) <= 0:
        raise ValueError("atom cached template_radius_pix must be positive")

    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    n_slices = max(1, int(cfg.n_slices))
    half = n / 2.0
    r = int(template_radius_pix)
    k = 2 * r + 1
    hpad = n + 2 * r
    wpad = n + 2 * r
    device = torch_device_from_cfg(cfg)

    z_chunks = np.array_split(np.arange(n, dtype=np.int32), n_slices)
    slab_z_starts = np.array([int(c[0]) for c in z_chunks], dtype=np.int32)
    slab_z_ends = np.array([int(c[-1]) + 1 for c in z_chunks], dtype=np.int32)
    dz_list = [float(len(c) * ps) for c in z_chunks]

    _, element_ids, ai_to_epos = _select_cached_atom_elements(atoms, elements_for_cache)
    if len(element_ids) == 0:
        phase_stack = torch.zeros((n_slices, n, n), device=device, dtype=torch.float32)
        amp_stack = torch.zeros_like(phase_stack)
        return slab_stack_to_list_torch(phase_stack), slab_stack_to_list_torch(amp_stack), dz_list

    if getattr(cfg, "verbose", False):
        print(
            f"Using torch grouped-conv cached atom slabs: atoms={len(atoms)}, "
            f"box={n}, slices={n_slices}, subpix={subpix_n}, radius={r}, device={device}"
        )

    xy_kernels = _precompute_cached_atom_xy_kernels_torch(
        cfg, element_ids, int(subpix_n), r, device
    )
    z_prefix = _precompute_cached_atom_z_prefix_numpy(cfg, element_ids, int(subpix_n), r)

    lam = electron_wavelength_angstrom(float(cfg.kv))
    lead_term = float(cfg.bond_scaling) * lam / 8.0 / (ps * ps)

    group_ids: List[int] = []
    center_x: List[int] = []
    center_y: List[int] = []
    weights: List[float] = []

    # Encode group = (((slab * n_elem + elem_pos) * subpix + sx) * subpix + sy) * 5 + gaussian.
    n_elem = len(element_ids)
    n_skipped_element = 0
    n_skipped_xy = 0
    n_skipped_z = 0

    for i_atom, atom in enumerate(atoms):
        elem = atom.element.upper()
        if elem not in ATOM_INDEX:
            if fallback_if_needed:
                if getattr(cfg, "verbose", False):
                    print(f"Unsupported element {elem}; falling back to non-cached NumPy slabs.")
                phase_np, amp_np, dz_fallback = make_phase_amp_slabs_direct_from_atoms_numpy(atoms, cfg)
                return numpy_slabs_to_torch(phase_np, amp_np, device=device) + (dz_fallback,)
            raise ValueError(f"Unsupported element for cached slabs: {elem}")

        ai = int(ATOM_INDEX[elem])
        epos = ai_to_epos.get(ai)
        if epos is None:
            n_skipped_element += 1
            continue

        x_pix, y_pix, z_pix = atom.xyz / ps + half
        ix = int(round(float(x_pix)))
        iy = int(round(float(y_pix)))
        iz = int(round(float(z_pix)))

        # If the full template does not overlap the target box in x/y, skip it.
        if ix < -r or ix >= n + r or iy < -r or iy >= n + r:
            n_skipped_xy += 1
            continue

        fx = float(x_pix) - ix
        fy = float(y_pix) - iy
        fz = float(z_pix) - iz
        sx = _subpixel_bin_from_fraction_numpy(fx, int(subpix_n))
        sy = _subpixel_bin_from_fraction_numpy(fy, int(subpix_n))
        sz = _subpixel_bin_from_fraction_numpy(fz, int(subpix_n))

        z0_raw = iz - r
        z1_raw = iz + r + 1
        z0 = max(0, z0_raw)
        z1 = min(n, z1_raw)
        if z0 >= z1:
            n_skipped_z += 1
            continue

        s0 = int(np.searchsorted(slab_z_ends, z0, side="right"))
        s1 = int(np.searchsorted(slab_z_starts, z1 - 1, side="right"))
        if s0 >= s1:
            n_skipped_z += 1
            continue

        cx_pad = ix + r
        cy_pad = iy + r
        occ = float(atom.occupancy)

        for s in range(s0, s1):
            zz0 = max(z0, int(slab_z_starts[s]))
            zz1 = min(z1, int(slab_z_ends[s]))
            if zz0 >= zz1:
                continue
            tz0 = int(zz0 - z0_raw)
            tz1 = int(tz0 + (zz1 - zz0))
            if tz0 < 0 or tz1 > k or tz0 >= tz1:
                if fallback_if_needed:
                    if getattr(cfg, "verbose", False):
                        print("Cached atom z clipping went out of range; falling back to non-cached NumPy slabs.")
                    phase_np, amp_np, dz_fallback = make_phase_amp_slabs_direct_from_atoms_numpy(atoms, cfg)
                    return numpy_slabs_to_torch(phase_np, amp_np, device=device) + (dz_fallback,)
                raise RuntimeError("Cached atom z clipping index out of range")

            for ig in range(5):
                zsum = float(z_prefix[epos, sz, ig, tz1] - z_prefix[epos, sz, ig, tz0])
                if zsum == 0.0:
                    continue
                weight = occ * float(SCATTERING_A[ai, ig]) * float(lead_term) * zsum
                if weight == 0.0:
                    continue
                group = (((int(s) * n_elem + int(epos)) * int(subpix_n) + int(sx)) * int(subpix_n) + int(sy)) * 5 + int(ig)
                group_ids.append(group)
                center_x.append(cx_pad)
                center_y.append(cy_pad)
                weights.append(weight)

        if getattr(cfg, "verbose", False) and (i_atom + 1) % 100000 == 0:
            print(f"  grouped cached atom metadata: {i_atom + 1}/{len(atoms)}")

    phase_stack = torch.zeros((n_slices, n, n), device=device, dtype=torch.float32)
    amp_stack = torch.zeros_like(phase_stack)

    if len(group_ids) == 0:
        if getattr(cfg, "verbose", False):
            print(
                "Torch cached atom slabs: no atom contributions "
                f"(skipped element={n_skipped_element}, xy={n_skipped_xy}, z={n_skipped_z})."
            )
        return slab_stack_to_list_torch(phase_stack), slab_stack_to_list_torch(amp_stack), dz_list

    groups = torch.as_tensor(group_ids, device=device, dtype=torch.long)
    cx = torch.as_tensor(center_x, device=device, dtype=torch.long)
    cy = torch.as_tensor(center_y, device=device, dtype=torch.long)
    vals = torch.as_tensor(weights, device=device, dtype=torch.float32)

    order = torch.argsort(groups, stable=True)
    groups_sorted = groups[order]
    unique_groups, counts = torch.unique_consecutive(groups_sorted, return_counts=True)

    groups_cpu = [int(x) for x in unique_groups.detach().cpu().tolist()]
    counts_cpu = [int(x) for x in counts.detach().cpu().tolist()]
    starts_cpu = [0]
    for c in counts_cpu:
        starts_cpu.append(starts_cpu[-1] + int(c))

    cx_sorted = cx[order].contiguous()
    cy_sorted = cy[order].contiguous()
    vals_sorted = vals[order].contiguous()

    import torch.nn.functional as F

    group_chunk = int(max(1, getattr(cfg, "atom_template_chunk_size", 16)))
    hwpad = int(hpad * wpad)
    n_groups = len(groups_cpu)

    if getattr(cfg, "verbose", False):
        print(
            f"Torch cached atom slabs: contributions={len(group_ids)}, groups={n_groups}, "
            f"chunk={group_chunk}, padded_impulse={hpad}x{wpad}"
        )

    for g0 in range(0, n_groups, group_chunk):
        g1 = min(n_groups, g0 + group_chunk)
        local_groups = groups_cpu[g0:g1]
        local_counts_cpu = counts_cpu[g0:g1]
        data_start = starts_cpu[g0]
        data_end = starts_cpu[g1]
        g_count = g1 - g0
        if g_count <= 0 or data_end <= data_start:
            continue

        local_counts = torch.as_tensor(local_counts_cpu, device=device, dtype=torch.long)
        local_channels = torch.repeat_interleave(torch.arange(g_count, device=device, dtype=torch.long), local_counts)
        cy_local = cy_sorted[data_start:data_end]
        cx_local = cx_sorted[data_start:data_end]
        vals_local = vals_sorted[data_start:data_end]

        impulse = torch.zeros((1, g_count, hpad, wpad), device=device, dtype=torch.float32)
        flat_idx = local_channels * hwpad + cy_local * wpad + cx_local
        impulse.view(-1).scatter_add_(0, flat_idx, vals_local)

        slab_indices: List[int] = []
        epos_indices: List[int] = []
        sx_indices: List[int] = []
        sy_indices: List[int] = []
        gaussian_indices: List[int] = []
        for gid in local_groups:
            tmp = int(gid)
            ig = tmp % 5
            tmp //= 5
            sy = tmp % int(subpix_n)
            tmp //= int(subpix_n)
            sx = tmp % int(subpix_n)
            tmp //= int(subpix_n)
            epos = tmp % n_elem
            slab = tmp // n_elem
            slab_indices.append(int(slab))
            epos_indices.append(int(epos))
            sx_indices.append(int(sx))
            sy_indices.append(int(sy))
            gaussian_indices.append(int(ig))

        epos_t = torch.as_tensor(epos_indices, device=device, dtype=torch.long)
        sx_t = torch.as_tensor(sx_indices, device=device, dtype=torch.long)
        sy_t = torch.as_tensor(sy_indices, device=device, dtype=torch.long)
        ig_t = torch.as_tensor(gaussian_indices, device=device, dtype=torch.long)
        kernels = xy_kernels[epos_t, sx_t, sy_t, ig_t]
        kernels_conv = torch.flip(kernels, dims=(-2, -1))[:, None, :, :].contiguous()

        conv_maps = F.conv2d(
            impulse,
            kernels_conv,
            bias=None,
            stride=1,
            padding=r,
            dilation=1,
            groups=g_count,
        )[0, :, r:r + n, r:r + n].to(torch.float32)

        for j, s in enumerate(slab_indices):
            if 0 <= s < n_slices:
                phase_stack[s].add_(conv_maps[j])

    return slab_stack_to_list_torch(phase_stack.contiguous()), slab_stack_to_list_torch(amp_stack.contiguous()), dz_list


def make_phase_amp_slabs_direct_from_atoms_numpy_cached(
    atoms: List[Atom],
    cfg: SimConfig,
    subpix_n: int = 9,
    template_radius_pix: int = 9,
    elements_for_cache: Optional[Sequence[str]] = None,
    fallback_if_needed: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
    """Cached atom direct-slab path.

    Historically this function used a CPU/NumPy 3D template cache.  In this
    optimized version, --use-cache-atom defaults to a torch grouped-convolution
    backend that is mathematically equivalent to cached template splatting up to
    fp32 accumulation order.  Use --atom-cache-backend numpy to force the legacy
    CPU cached implementation.
    """
    backend = str(getattr(cfg, "atom_cache_backend", "torch-convolution")).lower()
    subpix_n = int(getattr(cfg, "atom_cache_subpix_n", subpix_n))
    template_radius_pix = int(getattr(cfg, "atom_cache_radius_pix", template_radius_pix))

    if backend in {"numpy", "cpu", "legacy"}:
        return make_phase_amp_slabs_direct_from_atoms_numpy_cached_legacy(
            atoms,
            cfg,
            subpix_n=subpix_n,
            template_radius_pix=template_radius_pix,
            elements_for_cache=elements_for_cache,
            fallback_if_needed=fallback_if_needed,
        )
    if backend not in {"torch", "torch-convolution", "convolution", "conv", "gpu"}:
        raise ValueError(f"Unknown atom_cache_backend: {backend}")

    return make_phase_amp_slabs_direct_from_atoms_torch_cached_grouped(
        atoms,
        cfg,
        subpix_n=subpix_n,
        template_radius_pix=template_radius_pix,
        elements_for_cache=elements_for_cache,
        fallback_if_needed=fallback_if_needed,
    )


def direct_slab_z_bounds(n: int, n_slices: int) -> Tuple[List[int], List[int], List[float]]:
    """Match torch.tensor_split/np.array_split boundaries along z without creating a volume."""
    n_slices = max(1, int(n_slices))
    q, r = divmod(int(n), n_slices)
    starts: List[int] = []
    ends: List[int] = []
    z0 = 0
    for s in range(n_slices):
        size = q + (1 if s < r else 0)
        starts.append(z0)
        z0 += size
        ends.append(z0)
    return starts, ends, []


def make_phase_amp_slabs_direct_from_atoms_torch(atoms: List[Atom], cfg: SimConfig):
    """Generate protein phase slabs directly from atoms, skipping the full 3D volume.

    This replaces the slow path:
        make_scattering_volume_torch -> volume_to_slabs_torch

    The original 3D potential is separable in x/y/z. Since multislice needs only
    z-summed projected potential per slab, the z summation is performed immediately
    for each atom/slab intersection.
    """
    device = torch_device_from_cfg(cfg)
    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    n_slices = max(1, int(cfg.n_slices))
    half = n / 2.0

    slab_starts, slab_ends, _ = direct_slab_z_bounds(n, n_slices)
    dz_list = [float((e - s) * ps) for s, e in zip(slab_starts, slab_ends)]

    phase_slabs = [torch.zeros((n, n), device=device, dtype=torch.float32) for _ in range(n_slices)]
    amp_slabs = [torch.zeros_like(phase_slabs[0], dtype=torch.float32) for _ in range(n_slices)]

    lam = electron_wavelength_angstrom(cfg.kv)
    lead_term = cfg.bond_scaling * lam / 8.0 / (ps * ps)

    for i_atom, atom in enumerate(atoms):
        ai = ATOM_INDEX[atom.element]
        x, y, z = atom.xyz / ps + half
        ix, iy, iz = int(round(float(x))), int(round(float(y))), int(round(float(z)))
        bf = complete_bfactor(atom.bfactor, cfg.bfactor_scaling, cfg.min_bfactor)
        r = atom_neighborhood_radius(ps, bf)

        x0, x1i = max(0, ix - r), min(n, ix + r + 1)
        y0, y1i = max(0, iy - r), min(n, iy + r + 1)
        z0, z1i = max(0, iz - r), min(n, iz + r + 1)
        if x0 >= x1i or y0 >= y1i or z0 >= z1i:
            continue

        xs = torch.arange(x0, x1i, device=device, dtype=torch.float32)
        ys = torch.arange(y0, y1i, device=device, dtype=torch.float32)
        atom_xyz = atom.xyz.astype(np.float32)
        x1v = ((xs - half) * ps - float(atom_xyz[0])) - 0.5 * ps
        y1v = ((ys - half) * ps - float(atom_xyz[1])) - 0.5 * ps

        # Only the slabs intersecting this atom's local z support need work.
        for s, (sz0, sz1) in enumerate(zip(slab_starts, slab_ends)):
            zz0 = max(z0, sz0)
            zz1 = min(z1i, sz1)
            if zz0 >= zz1:
                continue
            zs = torch.arange(zz0, zz1, device=device, dtype=torch.float32)
            z1v = ((zs - half) * ps - float(atom_xyz[2])) - 0.5 * ps
            pot2d = voxel_integrated_projected_potential_torch(
                x1v, x1v + ps,
                y1v, y1v + ps,
                z1v, z1v + ps,
                ai, bf, lead_term, device,
            )
            phase_slabs[s][y0:y1i, x0:x1i] += pot2d * float(atom.occupancy)

        if cfg.verbose and (i_atom + 1) % 10000 == 0:
            print(f"  direct slab protein atoms: {i_atom + 1}/{len(atoms)}")

    return phase_slabs, amp_slabs, dz_list
def voxel_integrated_projected_potential_numpy(
    x1: np.ndarray,
    x2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    atom_index: int,
    bfactor: float,
    lead_term: float,
) -> np.ndarray:
    """
    Direct projected slab potential for one atom.

    This is equivalent to:
        voxel_integrated_potential(...).sum(axis=0)

    but avoids building the local 3D block. The original 3D integral is separable:
        potential(z,y,x) = dz[:,None,None] * dy[None,:,None] * dx[None,None,:]

    Therefore, summing over z inside one slab gives:
        projected(y,x) = sum(dz) * dy[:,None] * dx[None,:]
    """
    out = np.zeros((y1.size, x1.size), dtype=np.float32)

    if x1.size == 0 or y1.size == 0 or z1.size == 0:
        return out

    for i in range(5):
        b_total = SCATTERING_B[atom_index, i] + bfactor
        if b_total <= 0:
            continue

        bplus = math.sqrt(4.0 * math.pi * math.pi / b_total)

        dx = (erf(bplus * x2) - erf(bplus * x1)).astype(np.float32, copy=False)
        dy = (erf(bplus * y2) - erf(bplus * y1)).astype(np.float32, copy=False)
        dz = (erf(bplus * z2) - erf(bplus * z1)).astype(np.float32, copy=False)

        dz_sum = np.float32(dz.sum(dtype=np.float64))
        if dz_sum == 0.0:
            continue

        out += (
            np.float32(SCATTERING_A[atom_index, i] * lead_term)
            * dz_sum
            * dy[:, None]
            * dx[None, :]
        )

    return out


def make_phase_amp_slabs_direct_from_atoms_numpy(
    atoms: List[Atom],
    cfg: SimConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    CPU/NumPy direct atom -> phase_slabs path.

    This replaces:
        vol = make_scattering_volume(...)
        phase_slabs, amp_slabs, dz_list = volume_to_slabs(vol, ...)

    It does NOT generate the full 3D volume. It directly integrates atom
    contributions into z-slabs.

    Notes
    -----
    - amp_slabs are initialized as zeros, same as volume_to_slabs().
    - This is intended for the protein base slabs. Explicit water can still be
      added later by the existing water path.
    - For consistency with the original 3D volume path, z slabs are defined by
      splitting integer z pixel indices with np.array_split(np.arange(n), n_slices).
    """
    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    n_slices = max(1, int(cfg.n_slices))
    half = n / 2.0

    lam = electron_wavelength_angstrom(float(cfg.kv))
    lead_term = float(cfg.bond_scaling) * lam / 8.0 / (ps * ps)

    # Same slab partition as np.array_split(vol, n_slices, axis=0),
    # but without allocating vol.
    z_chunks = np.array_split(np.arange(n, dtype=np.int32), n_slices)
    slab_z_starts = np.array([int(c[0]) for c in z_chunks], dtype=np.int32)
    slab_z_ends = np.array([int(c[-1]) + 1 for c in z_chunks], dtype=np.int32)

    dz_list = [float(len(c) * ps) for c in z_chunks]

    phase_slabs = [
        np.zeros((n, n), dtype=np.float32)
        for _ in range(n_slices)
    ]
    amp_slabs = [
        np.zeros((n, n), dtype=np.float32)
        for _ in range(n_slices)
    ]

    for i_atom, atom in enumerate(atoms):
        ai = ATOM_INDEX[atom.element]

        x_pix, y_pix, z_pix = atom.xyz / ps + half
        ix = int(round(float(x_pix)))
        iy = int(round(float(y_pix)))
        iz = int(round(float(z_pix)))

        bf = complete_bfactor(
            float(atom.bfactor),
            float(cfg.bfactor_scaling),
            float(cfg.min_bfactor),
        )

        r = atom_neighborhood_radius(ps, bf)

        x0 = max(0, ix - r)
        x1i = min(n, ix + r + 1)
        y0 = max(0, iy - r)
        y1i = min(n, iy + r + 1)
        z0 = max(0, iz - r)
        z1i = min(n, iz + r + 1)

        if x0 >= x1i or y0 >= y1i or z0 >= z1i:
            continue

        # Only loop over slabs intersecting this atom's local z support.
        # slab_z_ends are exclusive. Need slabs with:
        #     slab_end > z0 and slab_start < z1i
        s0 = int(np.searchsorted(slab_z_ends, z0, side="right"))
        s1 = int(np.searchsorted(slab_z_starts, z1i - 1, side="right"))

        if s0 >= s1:
            continue

        xs = np.arange(x0, x1i, dtype=np.float64)
        ys = np.arange(y0, y1i, dtype=np.float64)

        x1_arr = ((xs - half) * ps - atom.xyz[0]) - 0.5 * ps
        x2_arr = x1_arr + ps
        y1_arr = ((ys - half) * ps - atom.xyz[1]) - 0.5 * ps
        y2_arr = y1_arr + ps

        scale = np.float32(atom.occupancy)

        for s in range(s0, s1):
            zz0 = max(z0, int(slab_z_starts[s]))
            zz1 = min(z1i, int(slab_z_ends[s]))

            if zz0 >= zz1:
                continue

            zs = np.arange(zz0, zz1, dtype=np.float64)
            z1_arr = ((zs - half) * ps - atom.xyz[2]) - 0.5 * ps
            z2_arr = z1_arr + ps

            pot2d = voxel_integrated_projected_potential_numpy(
                x1_arr,
                x2_arr,
                y1_arr,
                y2_arr,
                z1_arr,
                z2_arr,
                ai,
                bf,
                lead_term,
            )

            phase_slabs[s][y0:y1i, x0:x1i] += pot2d * scale

        if cfg.verbose and (i_atom + 1) % 10000 == 0:
            print(f"  direct slab atoms: {i_atom + 1}/{len(atoms)}")

    return phase_slabs, amp_slabs, dz_list
def numpy_slabs_to_torch(
    phase_slabs: List[np.ndarray],
    amp_slabs: List[np.ndarray],
    device: str | torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    device = torch.device(device)
    phase_t = [
        torch.as_tensor(s, dtype=torch.float32, device=device)
        for s in phase_slabs
    ]
    amp_t = [
        torch.as_tensor(s, dtype=torch.float32, device=device)
        for s in amp_slabs
    ]
    return phase_t, amp_t
    
def prepare_phase_amp_slabs_direct_torch(atoms: List[Atom], cfg: SimConfig):
    """Prepare slabs for one image without constructing a 3D protein volume."""
#    phase_slabs, amp_slabs, dz_list = make_phase_amp_slabs_direct_from_atoms_torch(atoms, cfg)
    if getattr(cfg, "use_cache_atom", False):
        phase_slabs_any, amp_slabs_any, dz_list = make_phase_amp_slabs_direct_from_atoms_numpy_cached(atoms, cfg)
    else:
        phase_slabs_any, amp_slabs_any, dz_list = make_phase_amp_slabs_direct_from_atoms_numpy(atoms, cfg)
    phase_slabs, amp_slabs = numpy_slabs_to_torch(phase_slabs_any, amp_slabs_any, device=cfg.device if hasattr(cfg, "device") else "cuda")
    if getattr(cfg, "radiation_damage", False) and getattr(cfg, "radiation_damage_where", "protein") in {"protein", "all"}:
        if cfg.verbose:
            if getattr(cfg, "radiation_damage_where", "protein") == "all" and cfg.explicit_water:
                print("Applying cisTEM-safe torch radiation-damage filter before explicit water; water is not post-filtered.")
            else:
                print("Applying torch radiation-damage exposure filter to direct protein slabs...")
        phase_slabs = apply_radiation_damage_to_slabs_torch(phase_slabs, cfg)

    if cfg.explicit_water:
        wc = prepare_water_cache_torch(atoms, cfg)
        if wc is not None:
            phase_slabs, amp_slabs = fill_water_potential_torch(
                phase_slabs, amp_slabs, wc.water_coords, wc.distance_2d, dz_list, cfg, wc.templates
            )
            phase_slabs, amp_slabs = apply_cistem_edge_pipeline_torch(phase_slabs, amp_slabs, cfg)

    # No post-water Fourier exposure filter; see prepare_phase_amp_slabs().

    return phase_slabs, amp_slabs, dz_list




def hydration_weight_torch(radius_a: torch.Tensor, pixel_size: float) -> torch.Tensor:
    v = torch.as_tensor(HYDRATION_RADIUS_VALS, device=radius_a.device, dtype=torch.float32)
    shifted = radius_a + float(PUSH_BACK_BY)
    return (
        0.5
        + 0.5 * torch.erf((shifted - (v[2] + float(HYDRATION_RADIUS_EXTRA_SHIFT) * pixel_size)) / (math.sqrt(2.0) * v[5]))
        + v[0] * torch.exp(-((shifted - (v[3] + float(HYDRATION_RADIUS_EXTRA_SHIFT) * pixel_size)) ** 2) / (2.0 * v[6] ** 2))
        + v[1] * torch.exp(-((shifted - (v[4] + float(HYDRATION_RADIUS_EXTRA_SHIFT) * pixel_size)) ** 2) / (2.0 * v[7] ** 2))
    ).to(torch.float32)


def add_mean_hydration_shell_torch(vol: torch.Tensor, atoms: List[Atom], cfg: SimConfig) -> torch.Tensor:
    device = vol.device
    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    half = n / 2.0
    dist2 = torch.full_like(vol, float("inf"), dtype=torch.float32, device=device)
    r_pix = int(math.ceil(9.0 / ps))
    for i_atom, atom in enumerate(atoms):
        x, y, z = atom.xyz / ps + half
        ix, iy, iz = int(round(float(x))), int(round(float(y))), int(round(float(z)))
        x0, x1i = max(0, ix - r_pix), min(n, ix + r_pix + 1)
        y0, y1i = max(0, iy - r_pix), min(n, iy + r_pix + 1)
        z0, z1i = max(0, iz - r_pix), min(n, iz + r_pix + 1)
        if x0 >= x1i or y0 >= y1i or z0 >= z1i:
            continue
        xs = torch.arange(x0, x1i, dtype=torch.float32, device=device)
        ys = torch.arange(y0, y1i, dtype=torch.float32, device=device)
        zs = torch.arange(z0, z1i, dtype=torch.float32, device=device)
        atom_xyz = atom.xyz.astype(np.float32)
        dx2 = ((xs - half) * ps - float(atom_xyz[0])) ** 2
        dy2 = ((ys - half) * ps - float(atom_xyz[1])) ** 2
        dz2 = ((zs - half) * ps - float(atom_xyz[2])) ** 2
        d2 = dz2[:, None, None] + dy2[None, :, None] + dx2[None, None, :]
        block = dist2[z0:z1i, y0:y1i, x0:x1i]
        dist2[z0:z1i, y0:y1i, x0:x1i] = torch.minimum(block, d2)
        if cfg.verbose and (i_atom + 1) % 10000 == 0:
            print(f"  torch mean hydration distance atoms: {i_atom + 1}/{len(atoms)}")
    finite = torch.isfinite(dist2)
    r = torch.zeros_like(dist2)
    r[finite] = torch.sqrt(dist2[finite])
    shell = torch.empty_like(dist2)
    shell[finite] = hydration_weight_torch(r[finite], ps)
    shell[~finite] = 0.0 if cfg.solvent_shell_only else 1.0
    oxygen_scale = float(SCATTERING_A[ATOM_INDEX["O"]].sum()) * WATER_DENSITY_PER_A3 * (ps ** 3)
    return vol + float(cfg.solvent_weight) * oxygen_scale * shell


def volume_to_slabs_torch(vol: torch.Tensor, n_slices: int, pixel_size: float):
    chunks = torch.tensor_split(vol, max(1, int(n_slices)), dim=0)
    phase_slabs = [c.sum(dim=0).to(torch.float32) for c in chunks]
    amp_slabs = [torch.zeros_like(phase_slabs[0], dtype=torch.float32) for _ in phase_slabs]
    dz_list = [float(c.shape[0] * pixel_size) for c in chunks]
    return phase_slabs, amp_slabs, dz_list


def nearest_atom_distance_2d_torch(atoms: List[Atom], cfg: SimConfig, max_r_a: float = 9.0) -> torch.Tensor:
    device = torch_device_from_cfg(cfg)
    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    half = n / 2.0
    dist2 = torch.full((n, n), float("inf"), dtype=torch.float32, device=device)
    r_pix = int(math.ceil(max_r_a / ps))
    for i_atom, atom in enumerate(atoms):
        x, y = atom.xyz[:2] / ps + half
        ix, iy = int(round(float(x))), int(round(float(y)))
        x0, x1i = max(0, ix - r_pix), min(n, ix + r_pix + 1)
        y0, y1i = max(0, iy - r_pix), min(n, iy + r_pix + 1)
        if x0 >= x1i or y0 >= y1i:
            continue
        xs = torch.arange(x0, x1i, dtype=torch.float32, device=device)
        ys = torch.arange(y0, y1i, dtype=torch.float32, device=device)
        atom_xyz = atom.xyz.astype(np.float32)
        dx2 = ((xs - half) * ps - float(atom_xyz[0])) ** 2
        dy2 = ((ys - half) * ps - float(atom_xyz[1])) ** 2
        d2 = dy2[:, None] + dx2[None, :]
        block = dist2[y0:y1i, x0:x1i]
        dist2[y0:y1i, x0:x1i] = torch.minimum(block, d2)
    dist = torch.full_like(dist2, float("inf"))
    finite = torch.isfinite(dist2)
    dist[finite] = torch.sqrt(dist2[finite])
    return dist


def precompute_projected_water_templates_torch(cfg: SimConfig) -> torch.Tensor:
    device = torch_device_from_cfg(cfg)
    subpix_n = int(cfg.water_subpix_n)
    if subpix_n <= 0:
        raise ValueError("water_subpix_n must be positive")
    radius_pix = int(cfg.water_template_radius_pix)
    ai = ATOM_INDEX["O"]
    ps = float(cfg.pixel_size)
    lam = electron_wavelength_angstrom(cfg.kv)
    lead_term = cfg.bond_scaling * lam / 8.0 / (ps * ps)
    bf = cistem_water_template_bfactor(cfg)
    xs = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.float32)
    ys = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.float32)
    zs = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.float32)
    templates = []
    offsets = cistem_subpixel_offsets(subpix_n)
    for sz, dz in enumerate(offsets):
        for sy, dy in enumerate(offsets):
            for sx, dx in enumerate(offsets):
                dx = float(dx)
                dy = float(dy)
                dz = float(dz)
                x1v = (xs - dx) * ps - 0.5 * ps
                y1v = (ys - dy) * ps - 0.5 * ps
                z1v = (zs - dz) * ps - 0.5 * ps
                pot3d = voxel_integrated_potential_torch(x1v, x1v + ps, y1v, y1v + ps, z1v, z1v + ps, ai, bf, lead_term, device)
                templates.append(pot3d.sum(dim=0).to(torch.float32))
    return torch.stack(templates, dim=0)






def fill_water_potential_torch_scatter_backend(
    phase_slabs: List[torch.Tensor], amp_slabs: List[torch.Tensor], water_coords: np.ndarray,
    distance_2d: Optional[torch.Tensor], dz_list: List[float], cfg: SimConfig, templates: torch.Tensor,
    inelastic_scalar_water: float = 0.0725, chunk_size: int = 50000,
):
    if water_coords is None or len(water_coords) == 0:
        return phase_slabs, amp_slabs
    device = phase_slabs[0].device
    coords = torch.as_tensor(water_coords, device=device, dtype=torch.float32)
    n_slices = len(phase_slabs)
    ny, nx = phase_slabs[0].shape
    ps = float(cfg.pixel_size)
    n_templates = int(templates.shape[0])
    subpix_n = int(round(n_templates ** (1.0 / 3.0)))
    if subpix_n ** 3 != n_templates:
        raise ValueError(f"Number of water templates must be subpix_n^3, got {templates.shape[0]}")
    if subpix_n % 2 != 1:
        raise ValueError("cisTEM water subpixel grid must be a positive odd integer, e.g. 5")
    radius_pix = int(templates.shape[-1] // 2)

    # Avoid a GPU->CPU sync here: dz_list is already a Python list.
    total_z_a = float(sum(float(x) for x in dz_list))
    dz_arr = torch.as_tensor(dz_list, device=device, dtype=torch.float32).contiguous()
    z_edges = torch.empty(n_slices + 1, device=device, dtype=torch.float32)
    z_edges[0] = -0.5 * total_z_a
    z_edges[1:] = z_edges[0] + torch.cumsum(dz_arr, dim=0)
    z_edges = z_edges.contiguous()

    x_a = coords[:, 0].contiguous()
    y_a = coords[:, 1].contiguous()
    z_a = coords[:, 2].contiguous()

    slab_all = torch.searchsorted(z_edges, z_a, right=True) - 1
    valid_z = (slab_all >= 0) & (slab_all < n_slices)

    slab = slab_all[valid_z].long().contiguous()
    x_valid = x_a[valid_z].contiguous()
    y_valid = y_a[valid_z].contiguous()
    z_valid = z_a[valid_z].contiguous()

    x_pix = x_valid / ps + nx / 2.0
    y_pix = y_valid / ps + ny / 2.0

    # Match simulate.cpp's modf(coord + 0.5) convention; do not use round-to-even.
    ix_center = torch.floor(x_pix + 0.5).long()
    iy_center = torch.floor(y_pix + 0.5).long()
    valid_xy = (ix_center >= 0) & (ix_center < nx) & (iy_center >= 0) & (iy_center < ny)

    slab = slab[valid_xy].contiguous()
    z_valid = z_valid[valid_xy].contiguous()
    x_pix = x_pix[valid_xy].contiguous()
    y_pix = y_pix[valid_xy].contiguous()
    ix = ix_center[valid_xy].contiguous()
    iy = iy_center[valid_xy].contiguous()

    if ix.numel() == 0:
        if getattr(cfg, "verbose", False):
            n_outside = int((~valid_z).sum().detach().cpu().item())
            n_edge = int((~valid_xy).sum().detach().cpu().item())
            print(f"fill_water_potential_torch: added=0, outside_z={n_outside}, skipped_edge={n_edge}")
        return phase_slabs, amp_slabs

    fx = x_pix - ix.to(torch.float32)
    fy = y_pix - iy.to(torch.float32)
    local_z_pix = (z_valid - z_edges[slab]) / ps
    iz_int = torch.floor(local_z_pix + 0.5).long()
    fz = local_z_pix - iz_int.to(torch.float32)

    subpix_half = (subpix_n - 1) // 2
    sx = torch.clamp(torch.trunc(fx * subpix_n).long() + subpix_half, 0, subpix_n - 1)
    sy = torch.clamp(torch.trunc(fy * subpix_n).long() + subpix_half, 0, subpix_n - 1)
    sz = torch.clamp(torch.trunc(fz * subpix_n).long() + subpix_half, 0, subpix_n - 1)
    template_index = (sz * subpix_n * subpix_n + sy * subpix_n + sx).long()

    weights = torch.ones_like(fx, dtype=torch.float32)
    if distance_2d is not None and getattr(cfg, "water_soft_weight", False):
        d = distance_2d[iy, ix]
        finite = torch.isfinite(d)
        weights[finite] = hydration_weight_torch(d[finite], ps)

    ratio = math.sqrt(float(inelastic_scalar_water) / 10.0)
    yoff = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.long)
    xoff = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.long)
    yy, xx = torch.meshgrid(yoff, xoff, indexing="ij")

    # Composite group sorting replaces many per-slab/per-template tensor.any(),
    # unique().tolist() and nonzero() synchronizations with one small CPU transfer
    # of the group table per fill.
    group = (slab * n_templates + template_index).long()
    order = torch.argsort(group, stable=True)
    group_sorted = group[order]
    unique_groups, counts = torch.unique_consecutive(group_sorted, return_counts=True)
    groups_cpu = unique_groups.detach().cpu().tolist()
    counts_cpu = counts.detach().cpu().tolist()

    ix_sorted = ix[order]
    iy_sorted = iy[order]
    weights_sorted = weights[order]

    start = 0
    n_added = 0
    for group_id, count in zip(groups_cpu, counts_cpu):
        end = start + int(count)
        s = int(group_id) // n_templates
        t = int(group_id) - s * n_templates
        if 0 <= s < n_slices and 0 <= t < n_templates:
            phase_flat = phase_slabs[s].reshape(-1)
            amp_flat = amp_slabs[s].reshape(-1)
            template = templates[t].to(torch.float32)
            for st in range(start, end, chunk_size):
                en = min(end, st + int(chunk_size))
                cy = iy_sorted[st:en]
                cx = ix_sorted[st:en]
                gy = cy[:, None, None] + yy[None, :, :]
                gx = cx[:, None, None] + xx[None, :, :]
                valid_pix = (gy >= 0) & (gy < ny) & (gx >= 0) & (gx < nx)
                flat_idx = (gy[valid_pix] * nx + gx[valid_pix]).reshape(-1)
                vals = (weights_sorted[st:en, None, None] * template[None, :, :])[valid_pix].reshape(-1)
                phase_flat.scatter_add_(0, flat_idx, vals)
                amp_flat.scatter_add_(0, flat_idx, vals * ratio)
            n_added += int(count)
        start = end

    if getattr(cfg, "verbose", False):
        n_outside = int((~valid_z).sum().detach().cpu().item())
        n_edge = int((~valid_xy).sum().detach().cpu().item())
        print(f"fill_water_potential_torch: added={n_added}, outside_z={n_outside}, skipped_edge={n_edge}")
    return phase_slabs, amp_slabs








def prepare_phase_amp_slabs_torch(vol: torch.Tensor, atoms: List[Atom], cfg: SimConfig):
    phase_slabs, amp_slabs, dz_list = volume_to_slabs_torch(vol, cfg.n_slices, cfg.pixel_size)
    if getattr(cfg, "radiation_damage", False) and getattr(cfg, "radiation_damage_where", "protein") in {"protein", "all"}:
        if cfg.verbose:
            if getattr(cfg, "radiation_damage_where", "protein") == "all" and cfg.explicit_water:
                print("Applying cisTEM-safe torch radiation-damage filter before explicit water; water is not post-filtered.")
            else:
                print("Applying torch radiation-damage exposure filter to protein slabs...")
        phase_slabs = apply_radiation_damage_to_slabs_torch(phase_slabs, cfg)
    if cfg.explicit_water:
        if cfg.solvent:
            raise ValueError("Do not use --solvent and --explicit-water together.")
        wc = prepare_water_cache_torch(atoms, cfg)
        if wc is not None:
            phase_slabs, amp_slabs = fill_water_potential_torch(phase_slabs, amp_slabs, wc.water_coords, wc.distance_2d, dz_list, cfg, wc.templates)
            phase_slabs, amp_slabs = apply_cistem_edge_pipeline_torch(phase_slabs, amp_slabs, cfg)
    # No post-water Fourier exposure filter; see prepare_phase_amp_slabs().
    return phase_slabs, amp_slabs, dz_list







# -----------------------------------------------------------------------------
# Stage-2/3/4 GPU-utilization implementations
# -----------------------------------------------------------------------------
# The first optimization pass above already cached Fourier filters, used rfft2 for
# real-valued filters, batched slabs for the dose filter, reduced GPU/CPU sync in
# the scatter path, and streamed frame accumulation.  The override block below is
# kept near the end of the file because it depends on helper functions defined above.

_fill_water_potential_torch_scatter_backend = fill_water_potential_torch_scatter_backend


@dataclass
class TorchWaterCache:
    templates: torch.Tensor
    water_coords: torch.Tensor
    distance_2d: Optional[torch.Tensor]
    generator: Optional[torch.Generator] = None
    static_phase_contrib: Optional[torch.Tensor] = None
    static_amp_contrib: Optional[torch.Tensor] = None


def _make_torch_generator_for_device(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    try:
        gen = torch.Generator(device=device)
    except TypeError:
        gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen




@dataclass
class TorchAtomCellIndex:
    atom_xyz_sorted: torch.Tensor
    starts: torch.Tensor
    counts: torch.Tensor
    dims: Tuple[int, int, int]
    half_box_a: float
    cell_size_a: float


def _atoms_xyz_tensor_torch(atoms: List[Atom], device: torch.device) -> torch.Tensor:
    if len(atoms) == 0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
    arr = np.asarray([a.xyz for a in atoms], dtype=np.float32)
    return torch.as_tensor(arr, device=device, dtype=torch.float32).contiguous()


def _water_octant_offsets_torch(device: torch.device) -> torch.Tensor:
    """Return cisTEM-style half-voxel octant offsets in x/y/z order."""
    return torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [-0.5,  0.5, -0.5],
            [ 0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [-0.5,  0.5,  0.5],
            [ 0.5,  0.5,  0.5],
        ],
        device=device,
        dtype=torch.float32,
    )


def _build_atom_cell_index_torch(
    atoms: List[Atom],
    cfg: SimConfig,
    device: torch.device,
    exclude_radius_a: float,
) -> Optional[TorchAtomCellIndex]:
    """Build a uniform-grid atom index for exact GPU radius queries.

    Cell size is guaranteed to be at least the exclusion radius, so any atom
    within cutoff of a water site must be in the same cell or one of the 26
    adjacent cells.  The dense count/start arrays are much smaller than an
    octant-resolution 3D mask and are built once per simulation.
    """
    if len(atoms) == 0:
        return None

    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    box_a = float(n) * ps
    half = 0.5 * box_a
    r = float(exclude_radius_a)

    atom_xyz = _atoms_xyz_tensor_torch(atoms, device)
    if atom_xyz.numel() == 0:
        return None

    # Atoms farther than r outside the water box cannot exclude any water.
    in_range = torch.all((atom_xyz >= (-half - r)) & (atom_xyz < (half + r)), dim=1)
    atom_xyz = atom_xyz[in_range].contiguous()
    if atom_xyz.numel() == 0:
        return None

    requested_cell = getattr(cfg, "water_filter_cell_size_a", None)
    if requested_cell is None or float(requested_cell) <= 0.0:
        # 4 A cells keep the dense cell table modest while still requiring only
        # the 27-neighbor search because cell_size >= cutoff.  The box_a/256
        # term prevents huge cell tables for very large working boxes.
        cell_size = max(r, 4.0 * ps, box_a / 256.0)
    else:
        cell_size = max(r, float(requested_cell))

    dim = max(1, int(math.ceil(box_a / cell_size)))
    n_cells = int(dim * dim * dim)
    max_cells = 256 ** 3
    if n_cells > max_cells:
        dim = 256
        cell_size = max(r, box_a / float(dim))
        n_cells = int(dim * dim * dim)

    shifted = (atom_xyz + half) / float(cell_size)
    cx = torch.clamp(torch.floor(shifted[:, 0]).long(), 0, dim - 1)
    cy = torch.clamp(torch.floor(shifted[:, 1]).long(), 0, dim - 1)
    cz = torch.clamp(torch.floor(shifted[:, 2]).long(), 0, dim - 1)
    cell_id = (cz * (dim * dim) + cy * dim + cx).long().contiguous()

    order = torch.argsort(cell_id, stable=True)
    cell_sorted = cell_id[order]
    atom_sorted = atom_xyz[order].contiguous()
    counts = torch.bincount(cell_sorted, minlength=n_cells).to(torch.long)
    starts = torch.cumsum(counts, dim=0) - counts

    return TorchAtomCellIndex(
        atom_xyz_sorted=atom_sorted,
        starts=starts.contiguous(),
        counts=counts.contiguous(),
        dims=(dim, dim, dim),
        half_box_a=half,
        cell_size_a=float(cell_size),
    )


def _filter_waters_by_atom_distance_torch_grid_with_index(
    water_coords: torch.Tensor,
    atom_index: Optional[TorchAtomCellIndex],
    exclude_radius_a: float,
    chunk_size: int = 250000,
) -> torch.Tensor:
    """Remove waters closer than exclude_radius_a to any atom on the torch device."""
    coords = torch.as_tensor(water_coords, dtype=torch.float32, device=water_coords.device).contiguous()
    if coords.numel() == 0 or atom_index is None or atom_index.atom_xyz_sorted.numel() == 0:
        return coords

    device = coords.device
    r2 = float(exclude_radius_a) * float(exclude_radius_a)
    atom_xyz = atom_index.atom_xyz_sorted
    starts = atom_index.starts
    counts = atom_index.counts
    nx, ny, nz = atom_index.dims
    half = float(atom_index.half_box_a)
    cell_size = float(atom_index.cell_size_a)
    xy_cells = nx * ny
    neighbor_offsets = tuple((dx, dy, dz) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))

    kept_chunks: List[torch.Tensor] = []
    n_total = int(coords.shape[0])
    chunk_size = max(1, int(chunk_size))

    for st in range(0, n_total, chunk_size):
        en = min(n_total, st + chunk_size)
        w = coords[st:en].contiguous()
        m = int(w.shape[0])
        if m == 0:
            continue

        shifted = (w + half) / cell_size
        base_x = torch.floor(shifted[:, 0]).long()
        base_y = torch.floor(shifted[:, 1]).long()
        base_z = torch.floor(shifted[:, 2]).long()
        too_close = torch.zeros((m,), device=device, dtype=torch.bool)

        for dx, dy, dz in neighbor_offsets:
            cx = base_x + int(dx)
            cy = base_y + int(dy)
            cz = base_z + int(dz)
            valid_cell = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny) & (cz >= 0) & (cz < nz)
            # This bool check is one tiny synchronization per neighbor/chunk;
            # the large pair construction and distance work remain on the GPU.
            if not bool(valid_cell.any()):
                continue

            local_water = torch.nonzero(valid_cell, as_tuple=False).flatten()
            cell_id = (cz[local_water] * xy_cells + cy[local_water] * nx + cx[local_water]).long()
            cell_counts = counts.index_select(0, cell_id)
            has_atoms = cell_counts > 0
            if not bool(has_atoms.any()):
                continue

            local_water = local_water[has_atoms].contiguous()
            cell_id = cell_id[has_atoms].contiguous()
            cell_counts = cell_counts[has_atoms].contiguous()
            total_pairs = int(cell_counts.sum().detach().cpu().item())
            if total_pairs <= 0:
                continue

            cell_starts = starts.index_select(0, cell_id)
            water_rep = torch.repeat_interleave(local_water, cell_counts, output_size=total_pairs)
            start_rep = torch.repeat_interleave(cell_starts, cell_counts, output_size=total_pairs)
            block_offsets = torch.repeat_interleave(
                torch.cumsum(cell_counts, dim=0) - cell_counts,
                cell_counts,
                output_size=total_pairs,
            )
            rel = torch.arange(total_pairs, device=device, dtype=torch.long) - block_offsets
            atom_rep = start_rep + rel

            d = w.index_select(0, water_rep) - atom_xyz.index_select(0, atom_rep)
            hit = torch.sum(d * d, dim=1) < r2
            if bool(hit.any()):
                too_close[water_rep[hit]] = True

        kept_chunks.append(w[~too_close].contiguous())

    if len(kept_chunks) == 0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
    return torch.cat(kept_chunks, dim=0).contiguous()


def filter_waters_by_atom_distance_torch_grid(
    water_coords: torch.Tensor,
    atoms: List[Atom],
    cfg: SimConfig,
    exclude_radius_a: float = 2.5,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    device = water_coords.device if isinstance(water_coords, torch.Tensor) else torch_device_from_cfg(cfg)
    atom_index = _build_atom_cell_index_torch(atoms, cfg, device, exclude_radius_a=float(exclude_radius_a))
    return _filter_waters_by_atom_distance_torch_grid_with_index(
        torch.as_tensor(water_coords, device=device, dtype=torch.float32).contiguous(),
        atom_index,
        exclude_radius_a=float(exclude_radius_a),
        chunk_size=int(chunk_size or getattr(cfg, "water_filter_chunk_size", 250000)),
    )


def _coords_from_flat_octant_indices_torch(flat: torch.Tensor, n: int, ps: float, z_offset: int = 0) -> torch.Tensor:
    device = flat.device
    flat = flat.to(torch.long)
    oct_id = torch.remainder(flat, 8).long()
    voxel = torch.div(flat, 8, rounding_mode="floor")
    ix = torch.remainder(voxel, n).to(torch.float32)
    tmp = torch.div(voxel, n, rounding_mode="floor")
    iy = torch.remainder(tmp, n).to(torch.float32)
    iz = torch.div(tmp, n, rounding_mode="floor").to(torch.float32) + float(z_offset)
    offs = _water_octant_offsets_torch(device).index_select(0, oct_id)
    coords_pix = torch.stack((ix, iy, iz), dim=1) + offs
    return ((coords_pix - float(n) / 2.0) * float(ps)).to(torch.float32).contiguous()


def generate_explicit_water_coords_torch(
    cfg: SimConfig,
    atoms: Optional[List[Atom]] = None,
    exclude_from_atoms: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate cisTEM-like explicit water coordinates directly on torch device.

    The seeding step keeps the same Bernoulli-per-octant model as the legacy
    NumPy implementation and cisTEM SeedWaters3d.  Protein exclusion is handled
    by an exact distance test using a GPU uniform-grid atom index.
    """
    if device is None:
        device = torch_device_from_cfg(cfg)
    else:
        device = torch.device(device)

    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    if n <= 0 or ps <= 0.0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)

    gen = _make_torch_generator_for_device(device, cfg.seed)
    box_a = float(n) * ps
    water_density_per_a3 = 0.94 * 0.6022140857 / 18.01528
    expected_n_water = water_density_per_a3 * (box_a ** 3) * float(cfg.water_density_scale)
    total_octants = int(n) * int(n) * int(n) * 8
    p_water_octant = expected_n_water / float(total_octants)
    p_water_octant = float(min(1.0, max(0.0, p_water_octant)))

    if cfg.verbose:
        print(f"Expected waters: {expected_n_water:.3e}")
        print(f"Water probability per voxel octant: {p_water_octant:.6g}")
        print(f"Generating explicit waters on {device} with torch RNG + grid exclusion...")

    atom_index = None
    if exclude_from_atoms and atoms is not None and len(atoms) > 0 and float(cfg.water_exclude_below_a) > 0.0:
        atom_index = _build_atom_cell_index_torch(
            atoms,
            cfg,
            device,
            exclude_radius_a=float(cfg.water_exclude_below_a),
        )

    filter_chunk = int(max(1, getattr(cfg, "water_filter_chunk_size", 250000)))

    def _maybe_filter(c: torch.Tensor) -> torch.Tensor:
        if c.numel() == 0 or atom_index is None:
            return c.reshape(-1, 3).to(torch.float32).contiguous()
        return _filter_waters_by_atom_distance_torch_grid_with_index(
            c.contiguous(),
            atom_index,
            exclude_radius_a=float(cfg.water_exclude_below_a),
            chunk_size=filter_chunk,
        )

    if p_water_octant <= 0.0 or total_octants <= 0:
        water_coords = torch.empty((0, 3), device=device, dtype=torch.float32)
    elif cfg.water_max_count is not None and int(cfg.water_max_count) > 0:
        # Debug path matching the old behavior: random octants, then exclusion,
        # then downsample if exclusion left too many.
        target = int(cfg.water_max_count)
        oversample = 3 if atom_index is not None else 1
        n_candidates = min(total_octants, max(target * oversample, target))
        flat = torch.randint(
            low=0,
            high=total_octants,
            size=(int(n_candidates),),
            device=device,
            dtype=torch.long,
            generator=gen,
        )
        water_coords = _coords_from_flat_octant_indices_torch(flat, n, ps)
        water_coords = _maybe_filter(water_coords)
    else:
        z_chunk = int(max(1, getattr(cfg, "water_seed_z_chunk", 16)))
        max_octants = int(max(8, getattr(cfg, "water_seed_max_octants_per_chunk", 0) or 0))
        if max_octants > 0:
            z_chunk = min(z_chunk, max(1, max_octants // max(1, n * n * 8)))
        z_chunk = min(n, max(1, z_chunk))
        offsets = _water_octant_offsets_torch(device)
        chunks: List[torch.Tensor] = []
        n_before = 0
        n_after = 0

        for z0 in range(0, n, z_chunk):
            z1 = min(n, z0 + z_chunk)
            zc = int(z1 - z0)
            occ = torch.rand((zc, n, n, 8), device=device, dtype=torch.float32, generator=gen) < p_water_octant
            idx = torch.nonzero(occ, as_tuple=False)
            del occ
            if idx.numel() == 0:
                continue
            n_before += int(idx.shape[0])

            iz = idx[:, 0].to(torch.float32) + float(z0)
            iy = idx[:, 1].to(torch.float32)
            ix = idx[:, 2].to(torch.float32)
            io = idx[:, 3].long()
            coords_pix = torch.stack((ix, iy, iz), dim=1) + offsets.index_select(0, io)
            coords = ((coords_pix - float(n) / 2.0) * ps).to(torch.float32).contiguous()
            del idx, coords_pix, iz, iy, ix, io

            coords = _maybe_filter(coords)
            if coords.numel() > 0:
                n_after += int(coords.shape[0])
                chunks.append(coords)

        if len(chunks) == 0:
            water_coords = torch.empty((0, 3), device=device, dtype=torch.float32)
        else:
            water_coords = torch.cat(chunks, dim=0).contiguous()

        if cfg.verbose:
            print(f"Generated waters before protein exclusion: {n_before}")
            print(f"Remaining waters after protein exclusion: {n_after}")

    if cfg.water_max_count is not None and int(cfg.water_max_count) > 0 and int(water_coords.shape[0]) > int(cfg.water_max_count):
        target = int(cfg.water_max_count)
        perm = torch.randperm(int(water_coords.shape[0]), device=device, generator=gen)[:target]
        water_coords = water_coords.index_select(0, perm).contiguous()
        if cfg.verbose:
            print(f"Downsampled waters to water_max_count: {int(water_coords.shape[0])}")

    if cfg.verbose:
        print(f"Final waters: {int(water_coords.shape[0])}")

    return water_coords.to(device=device, dtype=torch.float32).contiguous()


def prepare_water_cache_torch(atoms: List[Atom], cfg: SimConfig) -> Optional[TorchWaterCache]:
    """Generate waters once, keep coordinates on the torch device, and precompute templates.

    The default path now performs stochastic octant seeding and protein exclusion
    directly on the selected torch device.  The legacy NumPy/cKDTree generator is
    still available with --water-generation-backend numpy for comparison.
    After initialization, water coordinates remain resident on the GPU;
    --shake-waters updates them in place frame by frame.
    """
    if not cfg.explicit_water:
        return None
    if cfg.solvent:
        raise ValueError("Do not use --solvent and --explicit-water together.")

    device = torch_device_from_cfg(cfg)
    backend = str(getattr(cfg, "water_generation_backend", "torch")).lower()

    if backend == "numpy":
        if cfg.verbose:
            print("Generating explicit waters with legacy NumPy RNG + CPU KDTree exclusion...")
        water_coords_np = generate_explicit_water_coords(cfg, atoms=atoms, exclude_from_atoms=True)
        water_coords = torch.as_tensor(water_coords_np, device=device, dtype=torch.float32).contiguous()
    elif backend == "torch":
        water_coords = generate_explicit_water_coords_torch(
            cfg,
            atoms=atoms,
            exclude_from_atoms=True,
            device=device,
        ).to(device=device, dtype=torch.float32).contiguous()
    else:
        raise ValueError(f"Unknown water_generation_backend: {backend}")

    if cfg.verbose:
        print(f"Keeping {int(water_coords.shape[0])} water coordinates resident on {device}.")
        print("Precomputing projected water templates on torch device...")
    templates = precompute_projected_water_templates_torch(cfg).to(device=device, dtype=torch.float32).contiguous()
    distance_2d = nearest_atom_distance_2d_torch(atoms, cfg, max_r_a=9.0) if cfg.water_soft_weight else None

    # Cumulative water shaking uses one generator stream rather than reseeding
    # every frame.  This changes the exact random sequence relative to the older
    # CPU implementation, but it preserves the intended statistics and eliminates
    # per-frame CPU random-number generation and H2D transfers.
    base_seed = None if cfg.seed is None else int(cfg.seed) + 1000003
    gen = _make_torch_generator_for_device(device, base_seed)

    return TorchWaterCache(
        templates=templates,
        water_coords=water_coords,
        distance_2d=distance_2d,
        generator=gen,
    )


def shake_waters_3d_torch_inplace(
    water_cache: TorchWaterCache,
    cfg: SimConfig,
    dose_per_frame_e_per_a2: float,
) -> torch.Tensor:
    """cisTEM-like cumulative water shaking directly on the torch device."""
    coords = water_cache.water_coords
    if coords.numel() == 0:
        return coords

    sigma_a = 1.5 * float(dose_per_frame_e_per_a2)
    if sigma_a != 0.0:
        if water_cache.generator is None:
            noise = torch.randn(coords.shape, device=coords.device, dtype=coords.dtype)
        else:
            noise = torch.randn(coords.shape, device=coords.device, dtype=coords.dtype, generator=water_cache.generator)
        coords.add_(noise, alpha=float(sigma_a))

    box_a = float(cfg.box) * float(cfg.pixel_size)
    if box_a > 0.0:
        half = 0.5 * box_a
        coords.add_(half)
        coords.remainder_(box_a)
        coords.sub_(half)

    # Static non-shaken contribution is invalid once coordinates move.
    water_cache.static_phase_contrib = None
    water_cache.static_amp_contrib = None
    return coords


def _water_coordinate_bins_torch(
    water_coords: torch.Tensor,
    distance_2d: Optional[torch.Tensor],
    dz_list: List[float],
    cfg: SimConfig,
    templates: torch.Tensor,
    shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return grouped water splat metadata on device.

    Outputs are:
        group, ix, iy, weights, valid_z, valid_xy, z_edges
    where group = slab * n_templates + template_index.
    """
    device = templates.device
    coords = torch.as_tensor(water_coords, device=device, dtype=torch.float32).contiguous()
    n_templates = int(templates.shape[0])
    subpix_n = int(round(n_templates ** (1.0 / 3.0)))
    if subpix_n ** 3 != n_templates:
        raise ValueError(f"Number of water templates must be subpix_n^3, got {templates.shape[0]}")
    if subpix_n % 2 != 1:
        raise ValueError("cisTEM water subpixel grid must be a positive odd integer, e.g. 5")

    ny, nx = int(shape[0]), int(shape[1])
    n_slices = len(dz_list)
    ps = float(cfg.pixel_size)

    total_z_a = float(sum(float(x) for x in dz_list))
    dz_arr = torch.as_tensor(dz_list, device=device, dtype=torch.float32).contiguous()
    z_edges = torch.empty(n_slices + 1, device=device, dtype=torch.float32)
    z_edges[0] = -0.5 * total_z_a
    z_edges[1:] = z_edges[0] + torch.cumsum(dz_arr, dim=0)
    z_edges = z_edges.contiguous()

    if coords.numel() == 0:
        empty_l = torch.empty((0,), device=device, dtype=torch.long)
        empty_f = torch.empty((0,), device=device, dtype=torch.float32)
        empty_b = torch.empty((0,), device=device, dtype=torch.bool)
        return empty_l, empty_l, empty_l, empty_f, empty_b, empty_b, z_edges

    x_a = coords[:, 0].contiguous()
    y_a = coords[:, 1].contiguous()
    z_a = coords[:, 2].contiguous()

    slab_all = torch.searchsorted(z_edges, z_a, right=True) - 1
    valid_z = (slab_all >= 0) & (slab_all < n_slices)

    slab = slab_all[valid_z].long().contiguous()
    x_valid = x_a[valid_z].contiguous()
    y_valid = y_a[valid_z].contiguous()
    z_valid = z_a[valid_z].contiguous()

    x_pix = x_valid / ps + nx / 2.0
    y_pix = y_valid / ps + ny / 2.0

    # Match cisTEM's modf(coord + 0.5) / myroundint behavior and avoid
    # torch.round's banker-rounding on half-pixel water lattice coordinates.
    ix_center = torch.floor(x_pix + 0.5).long()
    iy_center = torch.floor(y_pix + 0.5).long()
    valid_xy = (ix_center >= 0) & (ix_center < nx) & (iy_center >= 0) & (iy_center < ny)

    slab = slab[valid_xy].contiguous()
    z_valid = z_valid[valid_xy].contiguous()
    x_pix = x_pix[valid_xy].contiguous()
    y_pix = y_pix[valid_xy].contiguous()
    ix = ix_center[valid_xy].contiguous()
    iy = iy_center[valid_xy].contiguous()

    if ix.numel() == 0:
        weights = torch.empty_like(x_pix, dtype=torch.float32)
        return slab, ix, iy, weights, valid_z, valid_xy, z_edges

    fx = x_pix - ix.to(torch.float32)
    fy = y_pix - iy.to(torch.float32)
    local_z_pix = (z_valid - z_edges[slab]) / ps
    iz_int = torch.floor(local_z_pix + 0.5).long()
    fz = local_z_pix - iz_int.to(torch.float32)

    subpix_half = (subpix_n - 1) // 2
    sx = torch.clamp(torch.trunc(fx * subpix_n).long() + subpix_half, 0, subpix_n - 1)
    sy = torch.clamp(torch.trunc(fy * subpix_n).long() + subpix_half, 0, subpix_n - 1)
    sz = torch.clamp(torch.trunc(fz * subpix_n).long() + subpix_half, 0, subpix_n - 1)
    template_index = (sz * subpix_n * subpix_n + sy * subpix_n + sx).long()

    weights = torch.ones_like(fx, dtype=torch.float32)
    if distance_2d is not None and getattr(cfg, "water_soft_weight", False):
        d = distance_2d[iy, ix]
        finite = torch.isfinite(d)
        weights[finite] = hydration_weight_torch(d[finite], ps)

    group = (slab * n_templates + template_index).long().contiguous()
    return group, ix, iy, weights.contiguous(), valid_z, valid_xy, z_edges


def fill_water_potential_torch(
    phase_slabs,
    amp_slabs,
    water_coords,
    distance_2d: Optional[torch.Tensor],
    dz_list: List[float],
    cfg: SimConfig,
    templates: torch.Tensor,
    inelastic_scalar_water: float = 0.0725,
    chunk_size: int = 50000,
):
    """Add explicit water using either grouped conv2d impulse maps or legacy scatter.

    The convolution backend is the stage-2 default.  For each slab/template group
    it builds an impulse image containing water-center weights, then applies the
    flipped projected-water template with grouped conv2d.  This is mathematically
    equivalent to template splatting with edge clipping up to fp32 accumulation
    order, but it turns millions of tiny atomic scatter updates into larger dense
    GPU kernels.
    """
    method = str(getattr(cfg, "water_splat_method", "convolution")).lower()
    if method == "scatter":
        return _fill_water_potential_torch_scatter_backend(
            phase_slabs,
            amp_slabs,
            water_coords,
            distance_2d,
            dz_list,
            cfg,
            templates,
            inelastic_scalar_water=inelastic_scalar_water,
            chunk_size=chunk_size,
        )
    if method not in {"convolution", "conv", "impulse"}:
        raise ValueError(f"Unknown water_splat_method: {method}")

    input_was_tensor = isinstance(phase_slabs, torch.Tensor)
    phase_stack = as_slab_stack_torch(phase_slabs)
    amp_stack = as_slab_stack_torch(amp_slabs)
    device = phase_stack.device

    coords = torch.as_tensor(water_coords, device=device, dtype=torch.float32)
    if coords.numel() == 0 or phase_stack.numel() == 0:
        return (phase_stack, amp_stack) if input_was_tensor else (slab_stack_to_list_torch(phase_stack), slab_stack_to_list_torch(amp_stack))

    templates = templates.to(device=device, dtype=torch.float32).contiguous()
    n_slices, ny, nx = int(phase_stack.shape[0]), int(phase_stack.shape[-2]), int(phase_stack.shape[-1])
    n_templates = int(templates.shape[0])
    radius_pix = int(templates.shape[-1] // 2)

    group, ix, iy, weights, valid_z, valid_xy, _ = _water_coordinate_bins_torch(
        coords,
        distance_2d,
        dz_list,
        cfg,
        templates,
        (ny, nx),
    )

    if group.numel() == 0:
        if getattr(cfg, "verbose", False):
            n_outside = int((~valid_z).sum().detach().cpu().item()) if valid_z.numel() else 0
            n_edge = int((~valid_xy).sum().detach().cpu().item()) if valid_xy.numel() else 0
            print(f"fill_water_potential_torch(conv): added=0, outside_z={n_outside}, skipped_edge={n_edge}")
        return (phase_stack, amp_stack) if input_was_tensor else (slab_stack_to_list_torch(phase_stack), slab_stack_to_list_torch(amp_stack))

    order = torch.argsort(group, stable=True)
    group_sorted = group[order]
    unique_groups, counts = torch.unique_consecutive(group_sorted, return_counts=True)

    # One small synchronization per fill to drive the Python loop over at most
    # n_slices * n_templates groups.  The water-center and convolution work stays
    # on the GPU.
    groups_cpu = [int(x) for x in unique_groups.detach().cpu().tolist()]
    counts_cpu = [int(x) for x in counts.detach().cpu().tolist()]
    starts_cpu = [0]
    for c in counts_cpu:
        starts_cpu.append(starts_cpu[-1] + int(c))

    ix_sorted = ix[order].contiguous()
    iy_sorted = iy[order].contiguous()
    weights_sorted = weights[order].contiguous()

    import torch.nn.functional as F

    # conv2d is cross-correlation, so flip templates to obtain splatting
    # semantics: output[center + offset] += template[radius + offset].
    templates_conv = torch.flip(templates, dims=(-2, -1)).contiguous()
    group_chunk = int(max(1, getattr(cfg, "water_template_chunk_size", 16)))
    ratio = math.sqrt(float(inelastic_scalar_water) / 10.0)
    hw = int(ny * nx)

    n_groups = len(groups_cpu)
    n_added = 0
    for g0 in range(0, n_groups, group_chunk):
        g1 = min(n_groups, g0 + group_chunk)
        local_groups = groups_cpu[g0:g1]
        local_counts_cpu = counts_cpu[g0:g1]
        data_start = starts_cpu[g0]
        data_end = starts_cpu[g1]
        g_count = g1 - g0
        if g_count <= 0 or data_end <= data_start:
            continue

        local_counts = torch.as_tensor(local_counts_cpu, device=device, dtype=torch.long)
        local_channels = torch.repeat_interleave(
            torch.arange(g_count, device=device, dtype=torch.long),
            local_counts,
        )
        cy = iy_sorted[data_start:data_end]
        cx = ix_sorted[data_start:data_end]
        vals = weights_sorted[data_start:data_end]

        impulse = torch.zeros((1, g_count, ny, nx), device=device, dtype=torch.float32)
        flat_idx = local_channels * hw + cy * nx + cx
        impulse.view(-1).scatter_add_(0, flat_idx, vals)

        t_indices = torch.as_tensor([gid % n_templates for gid in local_groups], device=device, dtype=torch.long)
        kernels = templates_conv.index_select(0, t_indices)[:, None, :, :].contiguous()
        conv_maps = F.conv2d(
            impulse,
            kernels,
            bias=None,
            stride=1,
            padding=radius_pix,
            dilation=1,
            groups=g_count,
        )[0].to(torch.float32)

        # Accumulate group maps into their target slabs.  g_count is intentionally
        # small, so this loop is not a bottleneck; the expensive work is the
        # grouped convolution above.
        for j, gid in enumerate(local_groups):
            s = int(gid) // n_templates
            if 0 <= s < n_slices:
                phase_stack[s].add_(conv_maps[j])
                amp_stack[s].add_(conv_maps[j], alpha=float(ratio))
        n_added += int(data_end - data_start)

    if getattr(cfg, "verbose", False):
        n_outside = int((~valid_z).sum().detach().cpu().item()) if valid_z.numel() else 0
        n_edge = int((~valid_xy).sum().detach().cpu().item()) if valid_xy.numel() else 0
        print(
            f"fill_water_potential_torch(conv): added={n_added}, "
            f"groups={n_groups}, outside_z={n_outside}, skipped_edge={n_edge}"
        )

    return (phase_stack, amp_stack) if input_was_tensor else (slab_stack_to_list_torch(phase_stack), slab_stack_to_list_torch(amp_stack))


def apply_cistem_edge_pipeline_torch(phase_slabs, amp_slabs, cfg: SimConfig):
    """Batch-aware post-water taper_edges + sampled-potential mask."""
    input_was_list = not isinstance(phase_slabs, torch.Tensor)
    phase = as_slab_stack_torch(phase_slabs) if input_was_list else phase_slabs.to(torch.float32)
    amp = as_slab_stack_torch(amp_slabs) if input_was_list else amp_slabs.to(torch.float32)

    if not getattr(cfg, "explicit_water", False) or getattr(cfg, "disable_cistem_edge_pipeline", False):
        return (slab_stack_to_list_torch(phase), slab_stack_to_list_torch(amp)) if input_was_list else (phase, amp)
    if phase.numel() == 0:
        return (slab_stack_to_list_torch(phase), slab_stack_to_list_torch(amp)) if input_was_list else (phase, amp)

    squeeze_b = False
    if phase.ndim == 3:
        phase = phase.unsqueeze(0)
        amp = amp.unsqueeze(0)
        squeeze_b = True
    elif phase.ndim != 4:
        raise ValueError("phase_slabs must be [S,H,W] or [B,S,H,W]")

    device = phase.device
    cache = get_torch_sim_cache(cfg, device)
    shape = tuple(phase.shape[-2:])
    sampled = phase.sum(dim=1).to(torch.float32)  # [B,H,W]

    width = int(max(0, getattr(cfg, "edge_taper_width_pix", 24)))
    if width > 0:
        band, count = cache.edge_band_mask(shape, width)
        if count > 0:
            phase_bg = phase[:, :, band].mean(dim=2).to(torch.float32)  # [B,S]
            amp_bg = amp[:, :, band].mean(dim=2).to(torch.float32)
        else:
            phase_bg = phase.mean(dim=(-2, -1)).to(torch.float32)
            amp_bg = amp.mean(dim=(-2, -1)).to(torch.float32)
        taper = cache.taper_mask(shape, width)
        phase = phase_bg[:, :, None, None] + (phase - phase_bg[:, :, None, None]) * taper[None, None, :, :]
        amp = amp_bg[:, :, None, None] + (amp - amp_bg[:, :, None, None]) * taper[None, None, :, :]

    phase_means = phase.mean(dim=(-2, -1)).to(torch.float32)  # [B,S]
    amp_means = amp.mean(dim=(-2, -1)).to(torch.float32)

    erode_pix = int(max(0, getattr(cfg, "sampled_mask_erode_pix", 7)))
    lowpass = float(getattr(cfg, "sampled_mask_lowpass", 0.05))
    if erode_pix > 0 or lowpass > 0.0:
        mask = (sampled > 1.0e-3).to(torch.float32)
        if erode_pix > 0:
            import torch.nn.functional as F
            inv = (1.0 - mask)[:, None, :, :]
            mask = 1.0 - F.max_pool2d(inv, kernel_size=2 * erode_pix + 1, stride=1, padding=erode_pix)[:, 0]
            mask = torch.clamp(mask, 0.0, 1.0)
        if lowpass > 0.0:
            filt = cache.gaussian_lowpass_rfft(tuple(mask.shape[-2:]), lowpass)
            mask = torch.clamp(apply_fourier_filter_real_rfft_torch(mask, filt), 0.0, 1.0)
        comp = (1.0 - mask).to(torch.float32)
        phase = phase * mask[:, None, :, :] + phase_means[:, :, None, None] * comp[:, None, :, :]
        amp = amp * mask[:, None, :, :] + amp_means[:, :, None, None] * comp[:, None, :, :]

    if squeeze_b:
        phase = phase[0].contiguous()
        amp = amp[0].contiguous()
    else:
        phase = phase.contiguous()
        amp = amp.contiguous()

    if input_was_list:
        return slab_stack_to_list_torch(phase), slab_stack_to_list_torch(amp)
    return phase, amp


def dose_filter_rfft_batch_torch(
    shape: Tuple[int, int],
    cfg: SimConfig,
    exposure_start: torch.Tensor,
    exposure_end: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return [B,H,W//2+1] cisTEM-style end-dose filters."""
    cache = get_torch_sim_cache(cfg, device)
    ne = cache.critical_dose_rfft(shape, float(cfg.pixel_size), float(getattr(cfg, "kv", 300.0)))
    d0 = exposure_start.to(device=device, dtype=torch.float32).reshape(-1, 1, 1)
    d1 = exposure_end.to(device=device, dtype=torch.float32).reshape(-1, 1, 1)
    filt = torch.exp(-0.5 * d1 / ne[None, :, :]).to(torch.float32)
    # The d0 input is retained for API clarity and future cumulative/interval
    # filters.  simulate.cpp's 2D CalculateDoseFilterAs1DArray uses dose_finish.
    _ = d0
    filt[:, 0, 0] = 1.0
    modify_signal = int(getattr(cfg, "exposure_filter_modify_signal", 0))
    if modify_signal == 1:
        filt = 1.0 - (1.0 - filt) / (1.0 + filt)
    elif modify_signal == 2:
        filt = torch.sqrt(torch.clamp(filt, min=0.0))
    return filt.to(torch.float32)


def apply_radiation_damage_to_frame_batch_torch(
    base_phase_stack: torch.Tensor,
    cfg: SimConfig,
    exposure_starts: Sequence[float],
    exposure_ends: Sequence[float],
) -> torch.Tensor:
    """Apply per-frame, per-slab radiation damage as one batched rFFT operation."""
    base = as_slab_stack_torch(base_phase_stack)
    device = base.device
    b = len(exposure_starts)
    if b <= 0:
        raise ValueError("empty exposure batch")
    if not getattr(cfg, "radiation_damage", False):
        return base.unsqueeze(0).expand(b, -1, -1, -1).clone().to(torch.float32)

    stack = base.unsqueeze(0).expand(b, -1, -1, -1).clone().to(torch.float32)
    bg = edge_mean_2d_stack_torch(base).to(torch.float32)  # [S]
    work = stack - bg[None, :, None, None]
    exp_start_t = torch.as_tensor(exposure_starts, device=device, dtype=torch.float32)
    exp_end_t = torch.as_tensor(exposure_ends, device=device, dtype=torch.float32)
    filt = dose_filter_rfft_batch_torch(tuple(work.shape[-2:]), cfg, exp_start_t, exp_end_t, device)
    out = torch.fft.irfft2(
        torch.fft.rfft2(work, dim=(-2, -1)) * filt[:, None, :, :],
        s=tuple(work.shape[-2:]),
        dim=(-2, -1),
    ).to(torch.float32)
    out = out + bg[None, :, None, None]
    return out.contiguous()


def _prepare_frame_batch_slabs_from_base_torch(
    base_phase_stack: torch.Tensor,
    base_amp_stack: torch.Tensor,
    dz_list: List[float],
    cfg: SimConfig,
    water_cache: Optional[TorchWaterCache],
    frame_indices: Sequence[int],
    dose_per_frame_e_per_a2: float,
    pre_exposure_e_per_a2: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    frame_indices = [int(x) for x in frame_indices]
    starts = [float(pre_exposure_e_per_a2) + i * float(dose_per_frame_e_per_a2) for i in frame_indices]
    ends = [d0 + float(dose_per_frame_e_per_a2) for d0 in starts]
    b = len(frame_indices)

    phase_batch = apply_radiation_damage_to_frame_batch_torch(
        base_phase_stack,
        cfg,
        starts,
        ends,
    )
    amp_batch = as_slab_stack_torch(base_amp_stack).unsqueeze(0).expand(b, -1, -1, -1).clone().to(torch.float32)

    if water_cache is not None:
        if getattr(cfg, "shake_waters", False):
            # Cumulative in-place shake, then fill the corresponding frame in the batch.
            for j, iframe in enumerate(frame_indices):
                if bool(getattr(cfg, "gpu_water_shake", True)):
                    shake_waters_3d_torch_inplace(water_cache, cfg, dose_per_frame_e_per_a2)
                    coords = water_cache.water_coords
                else:
                    # Fallback for debugging only; this reintroduces CPU random numbers and transfers.
                    coords_np = torch_to_numpy(water_cache.water_coords)
                    seed = None if cfg.seed is None else int(cfg.seed) + int(iframe) + 1000003
                    rng = np.random.default_rng(seed)
                    coords_np = shake_waters_3d(coords_np, cfg, dose_per_frame_e_per_a2, rng)
                    water_cache.water_coords = torch.as_tensor(coords_np, device=base_phase_stack.device, dtype=torch.float32).contiguous()
                    coords = water_cache.water_coords
                fill_water_potential_torch(
                    phase_batch[j],
                    amp_batch[j],
                    coords,
                    water_cache.distance_2d,
                    dz_list,
                    cfg,
                    water_cache.templates,
                )
        else:
            # Static water: compute the water contribution once and reuse for all
            # frames in the batch and subsequent batches.
            if water_cache.static_phase_contrib is None or water_cache.static_amp_contrib is None:
                water_phase = torch.zeros_like(base_phase_stack, dtype=torch.float32)
                water_amp = torch.zeros_like(base_amp_stack, dtype=torch.float32)
                fill_water_potential_torch(
                    water_phase,
                    water_amp,
                    water_cache.water_coords,
                    water_cache.distance_2d,
                    dz_list,
                    cfg,
                    water_cache.templates,
                )
                water_cache.static_phase_contrib = water_phase.contiguous()
                water_cache.static_amp_contrib = water_amp.contiguous()
            phase_batch = phase_batch + water_cache.static_phase_contrib[None, :, :, :]
            amp_batch = amp_batch + water_cache.static_amp_contrib[None, :, :, :]

        phase_batch, amp_batch = apply_cistem_edge_pipeline_torch(phase_batch, amp_batch, cfg)

    return phase_batch.contiguous(), amp_batch.contiguous()


def prepare_frame_slabs_from_base_torch(
    base_phase_slabs: List[torch.Tensor],
    base_amp_slabs: List[torch.Tensor],
    dz_list: List[float],
    cfg: SimConfig,
    water_cache: Optional[TorchWaterCache],
    exposure_start_e_per_a2: float,
    exposure_end_e_per_a2: float,
    dose_per_frame_e_per_a2: float,
    iframe: int,
):
    # Compatibility wrapper for older single-frame callers.
    phase_batch, amp_batch = _prepare_frame_batch_slabs_from_base_torch(
        as_slab_stack_torch(base_phase_slabs),
        as_slab_stack_torch(base_amp_slabs),
        dz_list,
        cfg,
        water_cache,
        [int(iframe)],
        dose_per_frame_e_per_a2,
        float(exposure_start_e_per_a2) - int(iframe) * float(dose_per_frame_e_per_a2),
    )
    return slab_stack_to_list_torch(phase_batch[0]), slab_stack_to_list_torch(amp_batch[0])


def finalize_detector_image_torch(img: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    """Detector finalization that treats [B,H,W] as B independent images."""
    out = img.to(torch.float32)
    spatial_dims = (-2, -1)
    if cfg.dqe:
        mean = out.mean(dim=spatial_dims, keepdim=True)
        contrast = out - mean
        filt = dqe_filter_rfft_torch(tuple(out.shape[-2:]), cfg.pixel_size, out.device, root=True, cfg=cfg)
        contrast = apply_fourier_filter_real_rfft_torch(contrast, filt)
        out = mean + contrast
    if cfg.poisson:
        if cfg.seed is not None:
            torch.manual_seed(int(cfg.seed))
            if out.device.type == "cuda":
                torch.cuda.manual_seed_all(int(cfg.seed))
        electrons_per_pixel = float(cfg.dose_e_per_a2) * (float(cfg.pixel_size) ** 2)
        mean = out.mean(dim=spatial_dims, keepdim=True)
        norm_intensity = out / (mean + 1e-12)
        out = torch.poisson(torch.clamp(norm_intensity * electrons_per_pixel, min=0.0)).to(torch.float32)
    return out.to(torch.float32)


def simulate_projection_from_slab_stack_batch_torch(phase_stack: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    squeeze = False
    phase = phase_stack.to(torch.float32)
    if phase.ndim == 3:
        phase = phase.unsqueeze(0)
        squeeze = True
    elif phase.ndim != 4:
        raise ValueError("phase_stack must be [S,H,W] or [B,S,H,W]")
    proj = phase.sum(dim=1).to(torch.float32)
    ctf = ctf_2d_rfft_torch(
        tuple(proj.shape[-2:]),
        cfg.pixel_size,
        cfg.kv,
        cfg.cs_mm,
        cfg.defocus_u,
        cfg.defocus_v,
        cfg.defocus_angle_deg,
        cfg.amplitude_contrast,
        cfg.phase_shift_rad,
        proj.device,
        cfg=cfg,
    )
    contrast = apply_fourier_filter_real_rfft_torch(proj, ctf)
    img = finalize_detector_image_torch(1.0 + contrast, cfg)
    return img[0].contiguous() if squeeze else img.contiguous()


def simulate_projection_from_slabs_torch(phase_slabs: Sequence[torch.Tensor], cfg: SimConfig) -> torch.Tensor:
    return simulate_projection_from_slab_stack_batch_torch(as_slab_stack_torch(phase_slabs), cfg)



def _cistem_inelastic_voltage_scale(kv: float) -> float:
    """Voltage-dependent inelastic scaling used in WaveFunctionPropagator."""
    kv = float(kv)
    if 299.0 < kv < 301.0:
        return 1.158
    if 199.0 < kv < 201.0:
        return 1.081
    # The cisTEM code applies no extra scaling at 100 kV, where the empirical
    # inelastic/elastic ratios were calibrated.
    return 1.0


def _cistem_radial_bin_index_full_torch(
    shape: Tuple[int, int],
    device: torch.device,
    cfg: SimConfig,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return full-FFT radial bin indices in reciprocal-pixel units.

    cisTEM builds a Curve from 0 to 0.5*sqrt(2) with roughly
    (nx/2+1)*sqrt(2)+1 samples before applying the whitening curve to the
    inelastic amplitude grating.  This helper caches the nearest-bin map and
    bin counts for a given image shape.
    """
    cache = get_torch_sim_cache(cfg, device)
    store = getattr(cache, "radial_bin_full_maps", None)
    if store is None:
        store = {}
        setattr(cache, "radial_bin_full_maps", store)
    shape = (int(shape[0]), int(shape[1]))
    key = shape
    cached = store.get(key)
    if cached is not None:
        return cached

    ny, nx = shape
    # Reciprocal-pixel grid, not reciprocal-Angstrom grid.
    fx = torch.fft.fftfreq(nx, d=1.0, device=device)
    fy = torch.fft.fftfreq(ny, d=1.0, device=device)
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    freq_pix = torch.sqrt(kx * kx + ky * ky).to(torch.float32)
    n_bins = int((max(nx, ny) / 2.0 + 1.0) * math.sqrt(2.0) + 1.0)
    n_bins = max(8, n_bins)
    step = (0.5 * math.sqrt(2.0)) / float(max(1, n_bins - 1))
    bin_idx = torch.clamp(torch.round(freq_pix / step).long(), 0, n_bins - 1).contiguous()
    counts = torch.bincount(bin_idx.reshape(-1), minlength=n_bins).to(torch.float32)
    counts = torch.clamp(counts, min=1.0)
    out = (bin_idx, counts, n_bins)
    store[key] = out
    return out


def _cistem_inelastic_lorentzian_filter_full_torch(
    shape: Tuple[int, int],
    pixel_size: float,
    device: torch.device,
    cfg: SimConfig,
) -> torch.Tensor:
    """cisTEM's empirical Lorentzian-like plasmon conversion filter.

    This intentionally preserves the expression as written in
    wave_function_propagator.cpp, including the `q1 * frequency_squared * +q2 *
    frequency` term, which C++ parses as q1*q2*f^3.
    """
    cache = get_torch_sim_cache(cfg, device)
    store = getattr(cache, "inelastic_lorentzian_full_filters", None)
    if store is None:
        store = {}
        setattr(cache, "inelastic_lorentzian_full_filters", store)
    shape = (int(shape[0]), int(shape[1]))
    key = (shape, float(pixel_size))
    cached = store.get(key)
    if cached is not None:
        return cached

    _, _, k2 = cache.full_grid(shape, pixel_size)
    frequency_squared = k2.to(torch.float32)
    frequency = torch.sqrt(torch.clamp(frequency_squared, min=0.0))

    p1 = 0.8235
    p2 = 47.4
    p3 = 1.0
    q1 = 2334.0
    q2 = 39.22
    q3 = 1.001
    numerator = p1 * frequency_squared + p2 * frequency + p3
    denominator = frequency_squared * frequency + q1 * frequency_squared * q2 * frequency + q3
    filt = (numerator / torch.clamp(denominator, min=1.0e-20)).to(torch.float32)
    store[key] = filt
    return filt


def _cistem_whiten_inelastic_fft_full_torch(
    fft_img: torch.Tensor,
    cfg: SimConfig,
) -> torch.Tensor:
    """Apply cisTEM-like radial whitening to full FFTs of amplitude gratings.

    Input shape is [M,H,W] complex.  The output is the same shape.  The exact
    cisTEM Curve interpolation is approximated with nearest radial bins; this
    keeps the important behavior: suppress the very strong smooth/DC inelastic
    background before the Lorentzian conversion filter.
    """
    if fft_img.ndim != 3:
        raise ValueError("fft_img must be [M,H,W]")
    m, ny, nx = fft_img.shape
    device = fft_img.device
    bin_idx, counts, n_bins = _cistem_radial_bin_index_full_torch((ny, nx), device, cfg)
    flat_bins = bin_idx.reshape(-1)
    power = (fft_img.real * fft_img.real + fft_img.imag * fft_img.imag).reshape(m, -1).to(torch.float32)
    sums = torch.zeros((m, n_bins), device=device, dtype=torch.float32)
    sums.scatter_add_(1, flat_bins[None, :].expand(m, -1), power)
    avg = sums / counts[None, :]
    # cisTEM takes sqrt, reciprocal, then normalizes by the maximum value.
    radial_weight = torch.rsqrt(torch.clamp(avg, min=1.0e-30))
    radial_weight = radial_weight / torch.clamp(radial_weight.max(dim=1, keepdim=True).values, min=1.0e-30)
    weight = radial_weight.gather(1, flat_bins[None, :].expand(m, -1)).reshape(m, ny, nx).to(torch.float32)
    return fft_img * weight


def cistem_filter_inelastic_amplitude_batch_torch(
    amp: torch.Tensor,
    cfg: SimConfig,
) -> torch.Tensor:
    """Match WaveFunctionPropagator's amplitude_grating preprocessing.

    cisTEM copies the inelastic potential into amplitude_grating, and for the
    real image path it scales it by voltage, whitens its Fourier amplitude, and
    applies an empirical Lorentzian plasmon filter before forming
    exp(-amplitude) * cos/sin(phase).  This function applies the same operation
    to a batch of per-slab amplitude images.
    """
    if bool(getattr(cfg, "disable_cistem_inelastic_filter", False)):
        return amp.to(torch.float32)

    x = amp.to(torch.float32)
    squeeze = False
    original_shape = x.shape
    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeeze = True
    elif x.ndim != 3:
        raise ValueError("amp must be [H,W] or [M,H,W]")

    m, ny, nx = x.shape
    mean = x.mean(dim=(-2, -1), keepdim=True)
    active = (mean > 0.001).to(torch.float32)

    scaled = x * float(_cistem_inelastic_voltage_scale(float(getattr(cfg, "kv", 300.0))))
    fft_amp = torch.fft.fft2(scaled, dim=(-2, -1))
    fft_amp = _cistem_whiten_inelastic_fft_full_torch(fft_amp, cfg)
    lorentz = _cistem_inelastic_lorentzian_filter_full_torch((ny, nx), float(cfg.pixel_size), x.device, cfg)
    filtered = torch.fft.ifft2(fft_amp * lorentz[None, :, :], dim=(-2, -1)).real.to(torch.float32)

    out = filtered * active + x * (1.0 - active)
    if squeeze:
        return out[0].contiguous()
    return out.reshape(original_shape).contiguous()


def cistem_propagator_distances_from_dz(dz_list: Sequence[float]) -> List[float]:
    """simulate.cpp stores slab propagation distances as negative thicknesses."""
    return [-abs(float(dz)) for dz in dz_list]


def cistem_defocus_offset_batch_torch(
    phase_stack: torch.Tensor,
    propagator_distances: Sequence[float],
    cfg: SimConfig,
) -> torch.Tensor:
    """Compute simulate.cpp's scattering-center defocus offset for [B,S,H,W]."""
    phase = phase_stack.to(torch.float32)
    if phase.ndim == 3:
        phase = phase.unsqueeze(0)
    if bool(getattr(cfg, "disable_cistem_defocus_offset", False)):
        return torch.zeros((phase.shape[0],), device=phase.device, dtype=torch.float32)

    b, s, _, _ = phase.shape
    prop = torch.as_tensor(list(propagator_distances), device=phase.device, dtype=torch.float32)
    if prop.numel() != s:
        raise ValueError("propagator_distances length must match number of slabs")

    # simulate.cpp updates scattering_total_shift[i] with all downstream slab
    # propagation distances, i.e. cumulative sum from i to the specimen exit.
    cumulative_from_slab = torch.flip(torch.cumsum(torch.flip(prop, dims=[0]), dim=0), dims=[0])
    mass = phase.sum(dim=(-2, -1)).to(torch.float32)
    total_mass = mass.sum(dim=1)
    weighted = (mass * cumulative_from_slab[None, :]).sum(dim=1)
    center = weighted / torch.clamp(total_mass, min=1.0e-20)
    center = torch.where(torch.abs(total_mass) > 1.0e-10, center, torch.zeros_like(center))
    # defocus_offset = scattering_center_of_mass - propagator_distance[0]/2
    offset = center - prop[0] / 2.0
    return offset.to(torch.float32)


def cistem_complex_transfer_full_batch_torch(
    shape: Tuple[int, int],
    cfg: SimConfig,
    device: torch.device,
    defocus_offset: torch.Tensor,
) -> torch.Tensor:
    """Complex final CTF transfer used by WaveFunctionPropagator.

    ctf[0] is initialized with amplitude contrast 1 and ctf[1] with amplitude
    contrast 0; the real/imag recombination is equivalent to multiplying by
    ctf_amp1 + i*ctf_amp0.
    """
    cache = get_torch_sim_cache(cfg, device)
    ny, nx = int(shape[0]), int(shape[1])
    kx, ky, k2 = cache.full_grid((ny, nx), float(cfg.pixel_size))
    b = int(defocus_offset.numel())

    if cfg.defocus_v is None:
        defocus_v = float(cfg.defocus_u)
    else:
        defocus_v = float(cfg.defocus_v)

    du = torch.as_tensor(float(cfg.defocus_u), device=device, dtype=torch.float32) - defocus_offset.to(device=device, dtype=torch.float32)
    dv = torch.as_tensor(defocus_v, device=device, dtype=torch.float32) - defocus_offset.to(device=device, dtype=torch.float32)

    theta = torch.atan2(ky, kx) - math.radians(float(cfg.defocus_angle_deg))
    astig = torch.cos(2.0 * theta)[None, :, :]
    defocus = 0.5 * (du[:, None, None] + dv[:, None, None]) + 0.5 * (du[:, None, None] - dv[:, None, None]) * astig
    lam = electron_wavelength_angstrom(float(cfg.kv))
    cs_a = float(cfg.cs_mm) * 1.0e7
    chi = math.pi * lam * defocus * k2[None, :, :] - 0.5 * math.pi * cs_a * (lam ** 3) * (k2[None, :, :] ** 2) + float(cfg.phase_shift_rad)
    h_real = -torch.cos(chi)
    h_imag = -torch.sin(chi)
    return torch.complex(h_real.to(torch.float32), h_imag.to(torch.float32)).to(torch.complex64)


def cistem_objective_aperture_mask_full_torch(
    shape: Tuple[int, int],
    cfg: SimConfig,
    device: torch.device,
) -> torch.Tensor:
    """Approximate WaveFunctionPropagator's objective aperture cosine mask."""
    diameter = float(getattr(cfg, "objective_aperture_diameter_micron", 100.0))
    if diameter <= 0.0:
        return torch.ones(shape, device=device, dtype=torch.float32)

    cache = get_torch_sim_cache(cfg, device)
    store = getattr(cache, "objective_aperture_masks", None)
    if store is None:
        store = {}
        setattr(cache, "objective_aperture_masks", store)
    key = (int(shape[0]), int(shape[1]), float(cfg.pixel_size), float(cfg.kv), diameter, float(getattr(cfg, "objective_aperture_falloff_pix", 14.0)))
    cached = store.get(key)
    if cached is not None:
        return cached

    wavelength = electron_wavelength_angstrom(float(cfg.kv))
    objective_lens_focal_length_mm = 3.5
    # Same expression as ReturnObjectiveApertureResoution().  Treat the result
    # as an Angstrom resolution and convert to reciprocal-pixel cutoff.
    resolution_a = (wavelength * objective_lens_focal_length_mm * 1.0e7) / ((diameter / 2.0) * 1.0e4)
    cutoff_recip_pix = float(cfg.pixel_size) / max(resolution_a, 1.0e-12)

    _, _, k2_pix = cache.full_grid(shape, pixel_size=1.0)
    freq_pix = torch.sqrt(k2_pix).to(torch.float32)
    ny, nx = int(shape[0]), int(shape[1])
    max_freq = 0.5 * math.sqrt(2.0)
    if cutoff_recip_pix >= max_freq:
        mask = torch.ones(shape, device=device, dtype=torch.float32)
    else:
        falloff_pix = max(0.0, float(getattr(cfg, "objective_aperture_falloff_pix", 14.0)))
        falloff = falloff_pix / float(max(nx, ny))
        if falloff <= 0.0:
            mask = (freq_pix <= cutoff_recip_pix).to(torch.float32)
        else:
            inner = max(0.0, cutoff_recip_pix - falloff)
            t = torch.clamp((freq_pix - inner) / max(cutoff_recip_pix - inner, 1.0e-12), 0.0, 1.0)
            mask = torch.where(
                freq_pix <= inner,
                torch.ones_like(freq_pix),
                torch.where(freq_pix >= cutoff_recip_pix, torch.zeros_like(freq_pix), 0.5 + 0.5 * torch.cos(math.pi * t)),
            ).to(torch.float32)
    store[key] = mask
    return mask


def propagate_slab_stack_batch_cistem_like_torch(
    phase_stack: torch.Tensor,
    amp_stack: torch.Tensor,
    dz_list: List[float],
    cfg: SimConfig,
) -> torch.Tensor:
    """cisTEM WaveFunctionPropagator-like multislice propagation.

    This implements the iContrast=1 image path from wave_function_propagator.cpp
    in compact complex PyTorch form.  The four real Image products used by the
    C++ implementation are mathematically equivalent to multiplying a complex
    wave by exp(-A+i*phi), applying the complex Fresnel propagator
    ctf[amp=1]+i*ctf[amp=0], applying the final objective CTF with the same
    real/imag split, then taking |wave|^2.

    Differences deliberately retained for performance/standalone use:
    - CTFFIND amplitude-contrast fitting is not run.
    - Beam-tilt full propagation is not implemented in this direct-slab script.
    - The objective aperture is not applied because its exact microscope
      geometry helper is in cisTEM headers not included here; the default 100 um
      aperture is usually beyond the simulated Nyquist for the use cases here.
    """
    squeeze = False
    phase = phase_stack.to(torch.float32)
    amp = amp_stack.to(torch.float32)
    if phase.ndim == 3:
        phase = phase.unsqueeze(0)
        amp = amp.unsqueeze(0)
        squeeze = True
    elif phase.ndim != 4:
        raise ValueError("phase_stack must be [S,H,W] or [B,S,H,W]")

    device = phase.device
    b, n_slices, ny, nx = phase.shape
    shape = (int(ny), int(nx))
    cache = get_torch_sim_cache(cfg, device)

    # cisTEM uses negative slab thicknesses as propagation distances and then
    # offsets the final CTF to the scattering center of mass.
    prop_distances = cistem_propagator_distances_from_dz(dz_list)
    defocus_offsets = cistem_defocus_offset_batch_torch(phase, prop_distances, cfg)

    # Inelastic amplitude-grating preprocessing from WaveFunctionPropagator.
    amp_for_grating = cistem_filter_inelastic_amplitude_batch_torch(
        amp.reshape(b * n_slices, ny, nx),
        cfg,
    ).reshape(b, n_slices, ny, nx)

    # SetInputWaveFunction initializes real=wave_function_in[0], imag=0.  In the
    # non-expert simulator path this is 1.0; keeping it at 1 preserves the older
    # Python intensity normalization and matches the common simulate.cpp path.
    wave = torch.ones((b, ny, nx), dtype=torch.complex64, device=device)

    for s, prop_dist in enumerate(prop_distances):
        phase0 = phase[:, s, :, :].to(torch.float32)
        if bool(getattr(cfg, "legacy_subtract_phase_edge_mean", False)):
            bg = edge_mean_2d_stack_torch(phase0).to(torch.float32)
            phase0 = phase0 - bg[:, None, None]
        amp0 = amp_for_grating[:, s, :, :].to(torch.float32)

        # phase_grating[0] = exp(-A)*cos(phi), phase_grating[1] = exp(-A)*sin(phi)
        # in the C++ code; this is the same complex transmission.
        transmission = torch.exp(torch.complex(-amp0, phase0)).to(torch.complex64)
        wave = wave * transmission

        # SetFresnelPropagator(0.0f, propagator_distance[iSlab]) followed by
        # ApplyCTF(fresnel_propagator[0/1]).  ctf[0]+i*ctf[1] = -exp(i*chi),
        # where chi = pi*lambda*propagator_distance*k^2 for Cs=0.
        prop = -cache.fresnel_full(shape, cfg.pixel_size, cfg.kv, float(prop_dist))
        wave = torch.fft.ifft2(
            torch.fft.fft2(wave, dim=(-2, -1)) * prop,
            dim=(-2, -1),
        ).to(torch.complex64)

    lens = cistem_complex_transfer_full_batch_torch(shape, cfg, device, defocus_offsets)
    image_wave = torch.fft.ifft2(
        torch.fft.fft2(wave, dim=(-2, -1)) * lens,
        dim=(-2, -1),
    ).to(torch.complex64)

    # WaveFunctionPropagator applies the objective aperture to the detector wave
    # before |wave|^2.  With the default 100 um aperture this mask is normally
    # all ones at cryo-EM pixel sizes, but the option is retained.
    aperture = cistem_objective_aperture_mask_full_torch(shape, cfg, device)
    if not bool(torch.all(aperture == 1.0)):
        image_wave = torch.fft.ifft2(
            torch.fft.fft2(image_wave, dim=(-2, -1)) * aperture,
            dim=(-2, -1),
        ).to(torch.complex64)

    img = image_wave.real * image_wave.real + image_wave.imag * image_wave.imag
    img = finalize_detector_image_torch(img.to(torch.float32), cfg)
    return img[0].contiguous() if squeeze else img.contiguous()


def propagate_slabs_cistem_like_torch(phase_slabs, amp_slabs, dz_list, cfg: SimConfig) -> torch.Tensor:
    return propagate_slab_stack_batch_cistem_like_torch(
        as_slab_stack_torch(phase_slabs),
        as_slab_stack_torch(amp_slabs),
        dz_list,
        cfg,
    )


def simulate_movie_from_direct_slabs_torch(atoms: List[Atom], cfg: SimConfig) -> torch.Tensor:
    """Per-frame movie simulation with batched propagation and GPU-resident waters."""
    n_frames = max(1, int(getattr(cfg, "number_of_frames", 1)))
    dose_per_frame = getattr(cfg, "dose_per_frame_e_per_a2", None)
    dose_per_frame = float(cfg.dose_e_per_a2) / float(n_frames) if dose_per_frame is None else float(dose_per_frame)
    pre = float(getattr(cfg, "pre_exposure_e_per_a2", 0.0))
    frame_batch_size = max(1, int(getattr(cfg, "frame_batch_size", 1)))

    if cfg.verbose:
        print(
            f"Torch direct-slab per-frame simulation: frames={n_frames}, "
            f"dose_per_frame={dose_per_frame:.4f}, pre_exposure={pre:.4f}, "
            f"frame_batch_size={frame_batch_size}, water_splat={getattr(cfg, 'water_splat_method', 'convolution')}"
        )

    if getattr(cfg, "use_cache_atom", False):
        base_phase_slabs_any, base_amp_slabs_any, dz_list = make_phase_amp_slabs_direct_from_atoms_numpy_cached(atoms, cfg)
    else:
        base_phase_slabs_any, base_amp_slabs_any, dz_list = make_phase_amp_slabs_direct_from_atoms_numpy(atoms, cfg)
    base_phase_slabs, base_amp_slabs = numpy_slabs_to_torch(
        base_phase_slabs_any,
        base_amp_slabs_any,
        device=torch_device_from_cfg(cfg),
    )
    base_phase_stack = as_slab_stack_torch(base_phase_slabs).contiguous()
    base_amp_stack = as_slab_stack_torch(base_amp_slabs).contiguous()
    water_cache = prepare_water_cache_torch(atoms, cfg)

    frames: List[torch.Tensor] = []
    summed: Optional[torch.Tensor] = None
    original_total_dose = float(cfg.dose_e_per_a2)
    save_frames = bool(getattr(cfg, "save_frames", False))

    try:
        cfg.dose_e_per_a2 = dose_per_frame
        for batch_start in range(0, n_frames, frame_batch_size):
            batch_end = min(n_frames, batch_start + frame_batch_size)
            frame_indices = list(range(batch_start, batch_end))
            if cfg.verbose:
                d0 = pre + frame_indices[0] * dose_per_frame
                d1 = pre + frame_indices[-1] * dose_per_frame + dose_per_frame
                print(
                    f"Frames {frame_indices[0] + 1}-{frame_indices[-1] + 1}/{n_frames}: "
                    f"exposure {d0:.3f} -> {d1:.3f} e-/A^2"
                )

            phase_batch, amp_batch = _prepare_frame_batch_slabs_from_base_torch(
                base_phase_stack,
                base_amp_stack,
                dz_list,
                cfg,
                water_cache,
                frame_indices,
                dose_per_frame,
                pre,
            )

            if cfg.mode == "projection":
                imgs = simulate_projection_from_slab_stack_batch_torch(phase_batch, cfg)
            elif cfg.mode == "multislice":
                imgs = propagate_slab_stack_batch_cistem_like_torch(phase_batch, amp_batch, dz_list, cfg)
            else:
                raise ValueError(f"Unknown mode: {cfg.mode}")
            imgs = imgs.to(torch.float32)
            if imgs.ndim == 2:
                imgs = imgs.unsqueeze(0)

            if save_frames:
                frames.append(imgs.contiguous())
            else:
                batch_sum = imgs.sum(dim=0).to(torch.float32)
                summed = batch_sum if summed is None else (summed + batch_sum)
    finally:
        cfg.dose_e_per_a2 = original_total_dose

    if save_frames:
        if not frames:
            raise RuntimeError("No frames were generated.")
        return torch.cat(frames, dim=0).to(torch.float32)
    if summed is None:
        raise RuntimeError("No frames were generated.")
    if getattr(cfg, "normalize_frame_sum", True):
        summed = summed / float(n_frames)
    return summed.to(torch.float32)


def simulate_movie_from_volume_torch(vol: torch.Tensor, atoms: List[Atom], cfg: SimConfig) -> torch.Tensor:
    """Volume-based movie path with the same batching backend as direct slabs."""
    base_phase_slabs, base_amp_slabs, dz_list = volume_to_slabs_torch(vol, cfg.n_slices, cfg.pixel_size)
    base_phase_stack = as_slab_stack_torch(base_phase_slabs).contiguous()
    base_amp_stack = as_slab_stack_torch(base_amp_slabs).contiguous()
    n_frames = max(1, int(getattr(cfg, "number_of_frames", 1)))
    dose_per_frame = getattr(cfg, "dose_per_frame_e_per_a2", None)
    dose_per_frame = float(cfg.dose_e_per_a2) / float(n_frames) if dose_per_frame is None else float(dose_per_frame)
    pre = float(getattr(cfg, "pre_exposure_e_per_a2", 0.0))
    frame_batch_size = max(1, int(getattr(cfg, "frame_batch_size", 1)))
    water_cache = prepare_water_cache_torch(atoms, cfg)

    frames: List[torch.Tensor] = []
    summed: Optional[torch.Tensor] = None
    original_total_dose = float(cfg.dose_e_per_a2)
    save_frames = bool(getattr(cfg, "save_frames", False))
    try:
        cfg.dose_e_per_a2 = dose_per_frame
        for batch_start in range(0, n_frames, frame_batch_size):
            batch_end = min(n_frames, batch_start + frame_batch_size)
            frame_indices = list(range(batch_start, batch_end))
            phase_batch, amp_batch = _prepare_frame_batch_slabs_from_base_torch(
                base_phase_stack, base_amp_stack, dz_list, cfg, water_cache, frame_indices, dose_per_frame, pre
            )
            if cfg.mode == "projection":
                imgs = simulate_projection_from_slab_stack_batch_torch(phase_batch, cfg)
            elif cfg.mode == "multislice":
                imgs = propagate_slab_stack_batch_cistem_like_torch(phase_batch, amp_batch, dz_list, cfg)
            else:
                raise ValueError(f"Unknown mode: {cfg.mode}")
            if imgs.ndim == 2:
                imgs = imgs.unsqueeze(0)
            if save_frames:
                frames.append(imgs.contiguous())
            else:
                batch_sum = imgs.sum(dim=0).to(torch.float32)
                summed = batch_sum if summed is None else (summed + batch_sum)
    finally:
        cfg.dose_e_per_a2 = original_total_dose
    if save_frames:
        return torch.cat(frames, dim=0).to(torch.float32)
    if summed is None:
        raise RuntimeError("No frames were generated.")
    if getattr(cfg, "normalize_frame_sum", True):
        summed = summed / float(n_frames)
    return summed.to(torch.float32)

def run(cfg: SimConfig, pdb_path: str | Path, output_path: str | Path) -> np.ndarray:
    """Run simulation using direct atom-to-slab protein potential generation.

    This optimized version intentionally skips --solvent and --save-volume. The main
    target is multislice/projection 2D simulation with optional explicit water.
    """
    if cfg.verbose:
        print("Reading PDB...")
    atoms = read_pdb_atoms(pdb_path, use_hydrogen=cfg.use_hydrogen)
    if cfg.center_by_mass:
        center_atoms(atoms)
    rotate_atoms_euler(atoms, cfg.euler_rot_deg, cfg.euler_tilt_deg, cfg.euler_psi_deg, inverse=cfg.euler_inverse)

    work_cfg, final_box, solvent_pad = make_cistem_work_config(cfg)

    if cfg.verbose:
        print(f"Loaded atoms: {len(atoms)}")
        if cfg.euler_rot_deg or cfg.euler_tilt_deg or cfg.euler_psi_deg:
            direction = "inverse/passive" if cfg.euler_inverse else "active"
            print(
                f"Applied ZYZ Euler rotation: rot={cfg.euler_rot_deg}, "
                f"tilt={cfg.euler_tilt_deg}, psi={cfg.euler_psi_deg} deg ({direction})"
            )
        if solvent_pad > 0:
            print(
                f"Using cisTEM-like solvent guard band: final_box={final_box}, "
                f"working_box={work_cfg.box}, pad={solvent_pad} px"
            )
        print(f"Using direct atom-to-slab protein potential on {torch_device_from_cfg(work_cfg)}")

    if getattr(work_cfg, "per_frame", False):
        img_t = simulate_movie_from_direct_slabs_torch(atoms, work_cfg)
    else:
        phase_slabs, amp_slabs, dz_list = prepare_phase_amp_slabs_direct_torch(atoms, work_cfg)
        if work_cfg.mode == "projection":
            if work_cfg.verbose:
                print("Simulating torch projection from direct slabs...")
            img_t = simulate_projection_from_slabs_torch(phase_slabs, work_cfg)
            #img_t = simulate_raw_projection_from_slabs_torch(phase_slabs, work_cfg)
        elif work_cfg.mode == "multislice":
            if work_cfg.verbose:
                print("Running torch cisTEM-like multislice from direct slabs...")
            img_t = propagate_slabs_cistem_like_torch(phase_slabs, amp_slabs, dz_list, work_cfg)
        else:
            raise ValueError(f"Unknown mode: {work_cfg.mode}")

    # Match simulate.cpp's late PadToWantedSize(): crop only after propagation,
    # detector operations and optional frame summing.
    img_t = center_crop_torch(img_t, final_box)

    img_np = torch_to_numpy(img_t)
    save_image_or_volume(output_path, img_np, cfg.pixel_size)
    if cfg.verbose:
        print(f"Saved {output_path}")
    return img_np

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimized PyTorch/GPU cisTEM simulate scaffold with cisTEM-like explicit water.")
    p.add_argument("pdb", help="Input PDB file")
    p.add_argument("output", help="Output .mrc or .npy image")
    p.add_argument("--box", type=int, default=256, help="Cubic box size in pixels")
    p.add_argument("--pixel-size", type=float, default=1.0, help="Pixel size in Angstrom")
    p.add_argument("--kv", type=float, default=300.0, help="Acceleration voltage in kV")
    p.add_argument("--cs", type=float, default=2.7, help="Spherical aberration in mm")
    p.add_argument("--defocus", type=float, default=15000.0, help="Mean defocus or defocus U in Angstrom")
    p.add_argument("--defocus-v", type=float, default=None, help="Defocus V in Angstrom")
    p.add_argument("--defocus-angle", type=float, default=0.0, help="Astigmatism angle in degrees")
    p.add_argument("--amplitude-contrast", type=float, default=0.07)
    p.add_argument("--phase-shift", type=float, default=0.0, help="Additional phase shift in radians")
    p.add_argument("--dose", type=float, default=30.0, help="Dose in e-/A^2 used when --poisson is set")
    p.add_argument("--mode", choices=["projection", "multislice"], default="projection")
    p.add_argument("--n-slices", type=int, default=16)
    p.add_argument("--min-bfactor", type=float, default=15.0)
    p.add_argument("--bfactor-scaling", type=float, default=0.0)
    p.add_argument("--bond-scaling", type=float, default=BOND_SCALING_DEFAULT)
    p.add_argument("--use-hydrogen", action="store_true")
    p.add_argument("--no-center", action="store_true", help="Do not shift model to center of mass")
    p.add_argument("--rot", type=float, default=0.0, help="Euler Rot angle in degrees, ZYZ convention")
    p.add_argument("--tilt", type=float, default=0.0, help="Euler Tilt angle in degrees, ZYZ convention")
    p.add_argument("--psi", type=float, default=0.0, help="Euler Psi angle in degrees, ZYZ convention")
    p.add_argument("--euler-inverse", action="store_true", help="Apply inverse/passive Euler transform instead of active coordinate rotation")
    p.add_argument("--explicit-water", action="store_true", help="Add simplified explicit projected water templates per slab")
    p.add_argument("--water-density-scale", type=float, default=1.0, help="Scale bulk water number density")
    p.add_argument("--water-max-count", type=int, default=None, help="Cap number of generated waters for testing")
    p.add_argument("--water-template-radius-pix", type=int, default=4)
    p.add_argument("--water-subpix-n", type=int, default=5)
    p.add_argument("--water-exclude-below", type=float, default=2.5, help="Skip waters whose 3D distance to protein is below this value in Angstrom")
    p.add_argument("--water-soft-weight", action="store_true", help="Apply 2D hydration-weight explicit waters near protein. Usually do not use")
    p.add_argument("--water-bfactor", type=float, default=34.0)
    p.add_argument("--solvent-padding-pix", type=int, default=64, help="cisTEM-like explicit-water guard band on each side before final center crop. Use 0 to disable")
    p.add_argument("--edge-taper-width-pix", type=int, default=24, help="Width of the cisTEM-like cosine taper applied inside the padded solvent box")
    p.add_argument("--sampled-mask-erode-pix", type=int, default=7, help="Erosion radius for the cisTEM-like sampled-potential mask")
    p.add_argument("--sampled-mask-lowpass", type=float, default=0.05, help="Gaussian low-pass cutoff in reciprocal pixels for the sampled-potential mask")
    p.add_argument("--disable-cistem-edge-pipeline", action="store_true", help="Disable post-water taper_edges and sampled-potential mask emulation")
    p.add_argument("--objective-aperture", type=float, default=100.0, help="Objective aperture diameter in microns for cisTEM-like multislice. Default matches simulate.cpp non-expert path")
    p.add_argument("--objective-aperture-falloff-pix", type=float, default=14.0, help="Cosine falloff width in Fourier pixels for the objective-aperture mask")
    p.add_argument("--disable-cistem-inelastic-filter", action="store_true", help="Disable cisTEM WaveFunctionPropagator inelastic amplitude filtering/scaling in multislice")
    p.add_argument("--disable-cistem-defocus-offset", action="store_true", help="Disable cisTEM-like scattering-center defocus offset in multislice final CTF")
    p.add_argument("--legacy-subtract-phase-edge-mean", action="store_true", help="Restore the older Python multislice behavior that subtracts the phase edge mean in each slab before transmission")
    p.add_argument("--radiation-damage", action="store_true", help="Apply Grant/Grigorieff-style radiation damage exposure filter to projected slabs")
    p.add_argument("--pre-exposure", type=float, default=0.0, help="Pre-exposure in e-/A^2 before this simulated image/frame")
    p.add_argument("--radiation-damage-where", choices=["protein", "all"], default="protein", help="protein: cisTEM 2D filter before water. all: safe cisTEM-compatible mode; also filters before water and avoids post-water Fourier filtering artifacts")
    p.add_argument("--exposure-filter-modify-signal", type=int, choices=[0, 1, 2], default=0, help="0: multiply filter, 1: 2F/(1+F), 2: sqrt(F)")
    p.add_argument("--per-frame", action="store_true", help="Simulate movie frames separately and sum/average them")
    p.add_argument("--number-of-frames", type=int, default=1)
    p.add_argument("--dose-per-frame", type=float, default=None, help="Dose per frame in e-/A^2. If omitted, use --dose / --number-of-frames")
    p.add_argument("--use-cache-atom", action="store_true", help="Use cached atomic potential. Uses the torch grouped-conv backend by default")
    p.add_argument("--atom-cache-backend", choices=["torch-convolution", "numpy"], default="torch-convolution", help="Backend for --use-cache-atom. torch-convolution keeps atom splatting on the GPU; numpy uses the legacy CPU cached template path")
    p.add_argument("--atom-cache-subpix-n", type=int, default=9, help="Subpixel bins per dimension for cached atom templates. Must be odd for the torch backend")
    p.add_argument("--atom-cache-radius-pix", type=int, default=9, help="Cached atom template radius in pixels")
    p.add_argument("--atom-template-chunk-size", type=int, default=16, help="Number of atom template groups processed per grouped conv2d chunk")
    p.add_argument("--save-frames", action="store_true", help="Save all frames as a stack instead of summed/averaged image")
    p.add_argument("--no-normalize-frame-sum", action="store_true", help="Do not divide summed frames by number of frames")
    p.add_argument("--shake-waters", action="store_true", help="Apply cisTEM-like random water displacement per frame")
    p.add_argument("--frame-batch-size", type=int, default=1, help="Number of movie frames to propagate as one GPU batch. Use 1 for legacy memory behavior; try 2, 4, or 8 to raise GPU utilization")
    p.add_argument("--water-splat-method", choices=["convolution", "scatter"], default="convolution", help="Water splatting backend. convolution builds per-template impulse maps and applies grouped conv2d; scatter keeps the previous grouped scatter_add path")
    p.add_argument("--water-template-chunk-size", type=int, default=16, help="Number of slab/template water groups processed per grouped conv2d chunk. Larger values can improve utilization but use more memory")
    p.add_argument("--disable-gpu-water-shake", action="store_true", help="Keep compatibility fallback for water shaking; by default water coordinates stay on the torch device and are shaken in place there")
    p.add_argument("--water-generation-backend", choices=["torch", "numpy"], default="torch", help="Backend for initial explicit-water seeding and protein exclusion. torch keeps the large stochastic water generation on the selected torch device; numpy is the legacy CPU RNG + cKDTree path")
    p.add_argument("--water-seed-z-chunk", type=int, default=16, help="Number of z slices per torch water-seeding chunk. Larger values improve throughput but use more temporary GPU memory")
    p.add_argument("--water-seed-max-octants-per-chunk", type=int, default=67108864, help="Maximum voxel-octant Bernoulli trials per torch water-seeding chunk. Lower this if seeding uses too much temporary memory")
    p.add_argument("--water-filter-chunk-size", type=int, default=250000, help="Number of candidate waters per GPU protein-exclusion chunk")
    p.add_argument("--water-filter-cell-size", type=float, default=None, help="Uniform-grid cell size in Angstrom for GPU water/protein exclusion. Default chooses a memory-safe value >= exclusion radius")
    p.add_argument("--water-exclusion-atom-chunk-size", type=int, default=1024, help="Legacy compatibility option; ignored by the current grid water-exclusion backend")
    p.add_argument("--water-exclusion-offset-chunk-size", type=int, default=2048, help="Legacy compatibility option; ignored by the current grid water-exclusion backend")
    p.add_argument("--dqe", action="store_true", help="Apply approximate sqrt(DQE) Fourier filter")
    p.add_argument("--poisson", action="store_true", help="Apply Poisson shot noise")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    a = parse_args(argv)
    cfg = SimConfig(
        box=a.box, pixel_size=a.pixel_size, kv=a.kv, cs_mm=a.cs,
        defocus_u=a.defocus, defocus_v=a.defocus_v, defocus_angle_deg=a.defocus_angle,
        amplitude_contrast=a.amplitude_contrast, phase_shift_rad=a.phase_shift,
        dose_e_per_a2=a.dose, n_slices=a.n_slices, min_bfactor=a.min_bfactor,
        bfactor_scaling=a.bfactor_scaling, bond_scaling=a.bond_scaling,
        center_by_mass=not a.no_center, euler_rot_deg=a.rot, euler_tilt_deg=a.tilt,
        euler_psi_deg=a.psi, euler_inverse=a.euler_inverse, use_hydrogen=a.use_hydrogen,
        solvent=False, solvent_weight=1.0, solvent_shell_only=True,
        explicit_water=a.explicit_water, water_density_scale=a.water_density_scale,
        water_max_count=a.water_max_count, water_template_radius_pix=a.water_template_radius_pix,
        water_subpix_n=a.water_subpix_n, water_exclude_below_a=a.water_exclude_below,
        water_soft_weight=a.water_soft_weight, water_bfactor=a.water_bfactor,
        mode=a.mode, poisson=a.poisson, dqe=a.dqe, seed=a.seed, verbose=a.verbose,
        radiation_damage=a.radiation_damage, radiation_damage_where=a.radiation_damage_where,
        pre_exposure_e_per_a2=a.pre_exposure, exposure_filter_modify_signal=a.exposure_filter_modify_signal,
        per_frame=a.per_frame, number_of_frames=a.number_of_frames,
        dose_per_frame_e_per_a2=a.dose_per_frame, save_frames=a.save_frames,use_cache_atom=a.use_cache_atom,
        normalize_frame_sum=not a.no_normalize_frame_sum, shake_waters=a.shake_waters,
        use_torch=True, device=a.device, solvent_padding_pix=a.solvent_padding_pix,
        edge_taper_width_pix=a.edge_taper_width_pix, sampled_mask_erode_pix=a.sampled_mask_erode_pix,
        sampled_mask_lowpass=a.sampled_mask_lowpass,
        disable_cistem_edge_pipeline=a.disable_cistem_edge_pipeline,
        frame_batch_size=max(1, int(a.frame_batch_size)),
        water_splat_method=a.water_splat_method,
        water_template_chunk_size=max(1, int(a.water_template_chunk_size)),
        gpu_water_shake=not a.disable_gpu_water_shake,
        water_generation_backend=a.water_generation_backend,
        water_seed_z_chunk=max(1, int(a.water_seed_z_chunk)),
        water_seed_max_octants_per_chunk=max(8, int(a.water_seed_max_octants_per_chunk)),
        water_filter_chunk_size=max(1, int(a.water_filter_chunk_size)),
        water_filter_cell_size_a=a.water_filter_cell_size,
        water_exclusion_atom_chunk_size=max(1, int(a.water_exclusion_atom_chunk_size)),
        water_exclusion_offset_chunk_size=max(1, int(a.water_exclusion_offset_chunk_size)),
        atom_cache_backend=a.atom_cache_backend,
        atom_cache_subpix_n=max(1, int(a.atom_cache_subpix_n)),
        atom_cache_radius_pix=max(1, int(a.atom_cache_radius_pix)),
        atom_template_chunk_size=max(1, int(a.atom_template_chunk_size)),
        objective_aperture_diameter_micron=float(a.objective_aperture),
        objective_aperture_falloff_pix=float(a.objective_aperture_falloff_pix),
        disable_cistem_inelastic_filter=bool(a.disable_cistem_inelastic_filter),
        disable_cistem_defocus_offset=bool(a.disable_cistem_defocus_offset),
        legacy_subtract_phase_edge_mean=bool(a.legacy_subtract_phase_edge_mean),
    )
    run(cfg, a.pdb, a.output)


if __name__ == "__main__":
    main()
