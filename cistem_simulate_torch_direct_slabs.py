#!/usr/bin/env python3
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
from dataclasses import dataclass
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
def critical_exposure_grant_grigorieff(freq_a_inv: np.ndarray) -> np.ndarray:
    """
    Approximate critical exposure Ne(f) in e-/A^2.

    freq_a_inv is spatial frequency in 1/A.

    This follows the commonly used Grant/Grigorieff-style empirical form:
        Ne(f) = 0.245 * f^(-1.665) + 2.81

    Low-frequency DC is protected by setting Ne to very large.
    """
    f = np.asarray(freq_a_inv, dtype=np.float32)
    ne = np.empty_like(f, dtype=np.float32)

    positive = f > 1.0e-6
    ne[positive] = 0.245 * np.power(f[positive], -1.665) + 2.81
    ne[~positive] = 1.0e9

    return ne.astype(np.float32)


def dose_filter_interval(
    shape: Tuple[int, int],
    pixel_size: float,
    exposure_start_e_per_a2: float,
    exposure_end_e_per_a2: float,
    average_over_interval: bool = True,
) -> np.ndarray:
    """
    Fourier-space radiation-damage filter for one exposure interval.

    If average_over_interval=True, returns the average decay factor over
    [exposure_start, exposure_end], which is appropriate for one integrated frame.

    Signal amplitude is modeled as:
        A(f, D) = exp(-D / (2 * Ne(f)))

    The interval average is:
        mean_A = (1 / ΔD) ∫_D0^D1 exp(-D / (2Ne)) dD
               = (2Ne / ΔD) * [exp(-D0/(2Ne)) - exp(-D1/(2Ne))]

    If average_over_interval=False, uses the end-exposure factor only.
    """
    _, _, k2 = frequency_grid(shape, pixel_size)
    freq = np.sqrt(k2).astype(np.float32)

    ne = critical_exposure_grant_grigorieff(freq)

    d0 = float(exposure_start_e_per_a2)
    d1 = float(exposure_end_e_per_a2)
    if d1 < d0:
        raise ValueError("exposure_end_e_per_a2 must be >= exposure_start_e_per_a2")

    if d1 == d0:
        return np.ones(shape, dtype=np.float32)

    if average_over_interval:
        delta = d1 - d0
        filt = (2.0 * ne / delta) * (
            np.exp(-d0 / (2.0 * ne)) - np.exp(-d1 / (2.0 * ne))
        )
    else:
        filt = np.exp(-d1 / (2.0 * ne))

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
        average_over_interval=True,
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

        phase_slabs, amp_slabs = fill_water_potential_python(
            phase_slabs,
            amp_slabs,
            frame_water_coords,
            water_cache.distance_2d,
            dz_list,
            cfg,
            water_cache.templates,
        )

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
    # cisTEM uses a water B-factor in this section; this is a tunable approximation.
    bf = 0.25 * float(cfg.water_bfactor)

    xs = np.arange(-radius_pix, radius_pix + 1)
    ys = np.arange(-radius_pix, radius_pix + 1)
    zs = np.arange(-radius_pix, radius_pix + 1)

    templates: List[np.ndarray] = []
    center = (subpix_n - 1) / 2.0
    for sz in range(subpix_n):
        for sy in range(subpix_n):
            for sx in range(subpix_n):
                dx = (sx - center) / subpix_n
                dy = (sy - center) / subpix_n
                dz = (sz - center) / subpix_n

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

        ix = int(round(x_pix)) % nx
        iy = int(round(y_pix)) % ny

        # Need enough room for the projected water template.
        # We still support clipping at edges below, so this is permissive.
#        if ix < -radius_pix or ix >= nx + radius_pix or iy < -radius_pix or iy >= ny + radius_pix:
#            n_edge += 1
#            continue

        fx = x_pix - ix
        fy = y_pix - iy

        # z subpixel position inside this slab.
        # Convert z_a relative to slab start to pixel units.
        local_z_pix = (z_a - float(z_edges[slab])) / ps
        iz_int = int(round(local_z_pix))
        fz = local_z_pix - iz_int

        sx = int(np.clip(np.floor((fx + 0.5) * subpix_n), 0, subpix_n - 1))
        sy = int(np.clip(np.floor((fy + 0.5) * subpix_n), 0, subpix_n - 1))
        sz = int(np.clip(np.floor((fz + 0.5) * subpix_n), 0, subpix_n - 1))

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

        add_template_periodic(
            phase_slabs[slab],
            template,
            iy,
            ix,
            scale=weight,
        )
        add_template_periodic(
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
    if getattr(cfg, "radiation_damage", False) and getattr(cfg, "radiation_damage_where", "protein") == "protein":
        if cfg.verbose:
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

    # Optional: damage everything after explicit water.
    # I would not use this as the default, but it is useful for testing.
    if getattr(cfg, "radiation_damage", False) and getattr(cfg, "radiation_damage_where", "protein") == "all":
        if cfg.verbose:
            print("Applying radiation-damage exposure filter to all phase slabs...")
        phase_slabs = apply_radiation_damage_to_slabs(phase_slabs, cfg)

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


def critical_exposure_grant_grigorieff_torch(freq_a_inv: torch.Tensor) -> torch.Tensor:
    f = freq_a_inv.to(torch.float32)
    ne = torch.empty_like(f)
    positive = f > 1.0e-6
    ne[positive] = 0.245 * torch.pow(f[positive], -1.665) + 2.81
    ne[~positive] = 1.0e9
    return ne


def dose_filter_interval_torch(
    shape: Tuple[int, int],
    pixel_size: float,
    exposure_start_e_per_a2: float,
    exposure_end_e_per_a2: float,
    device: torch.device,
    average_over_interval: bool = True,
) -> torch.Tensor:
    _, _, k2 = frequency_grid_torch(shape, pixel_size, device)
    freq = torch.sqrt(k2).to(torch.float32)
    ne = critical_exposure_grant_grigorieff_torch(freq)
    d0 = float(exposure_start_e_per_a2)
    d1 = float(exposure_end_e_per_a2)
    if d1 < d0:
        raise ValueError("exposure_end_e_per_a2 must be >= exposure_start_e_per_a2")
    if d1 == d0:
        return torch.ones(shape, device=device, dtype=torch.float32)
    if average_over_interval:
        delta = d1 - d0
        filt = (2.0 * ne / delta) * (torch.exp(-d0 / (2.0 * ne)) - torch.exp(-d1 / (2.0 * ne)))
    else:
        filt = torch.exp(-d1 / (2.0 * ne))
    filt = filt.to(torch.float32)
    filt[0, 0] = 1.0
    return filt


def apply_exposure_filter_2d_torch(
    img: torch.Tensor,
    pixel_size: float,
    exposure_start_e_per_a2: float,
    exposure_end_e_per_a2: float,
    modify_signal: int = 0,
    subtract_edge_mean: bool = True,
) -> torch.Tensor:
    x = img.to(torch.float32)
    if subtract_edge_mean:
        bg = edge_mean_2d_torch(x)
        work = x - bg
    else:
        bg = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        work = x
    filt = dose_filter_interval_torch(work.shape, pixel_size, exposure_start_e_per_a2, exposure_end_e_per_a2, x.device, True)
    if modify_signal == 1:
        filt = 1.0 - (1.0 - filt) / (1.0 + filt)
    elif modify_signal == 2:
        filt = torch.sqrt(torch.clamp(filt, min=0.0))
    out = torch.fft.ifft2(torch.fft.fft2(work) * filt).real.to(torch.float32)
    return out + bg


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
    out = []
    for i, slab in enumerate(phase_slabs):
        out.append(apply_exposure_filter_2d_torch(
            slab, cfg.pixel_size, exposure_start_e_per_a2, exposure_end_e_per_a2,
            modify_signal=int(getattr(cfg, "exposure_filter_modify_signal", 0)),
            subtract_edge_mean=True,
        ))
    #    if getattr(cfg, "verbose", False):
    #        print(f"  torch radiation damage slab {i+1}/{len(phase_slabs)}")
    return out


def ctf_2d_torch(
    shape: Tuple[int, int], pixel_size: float, kv: float, cs_mm: float,
    defocus_u: float, defocus_v: Optional[float] = None, defocus_angle_deg: float = 0.0,
    amplitude_contrast: float = 0.07, phase_shift_rad: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if defocus_v is None:
        defocus_v = defocus_u
    lam = electron_wavelength_angstrom(kv)
    cs_a = cs_mm * 1e7
    kx, ky, k2 = frequency_grid_torch(shape, pixel_size, device)
    theta = torch.atan2(ky, kx) - math.radians(defocus_angle_deg)
    defocus = 0.5 * (defocus_u + defocus_v) + 0.5 * (defocus_u - defocus_v) * torch.cos(2.0 * theta)
    chi = math.pi * lam * defocus * k2 - 0.5 * math.pi * cs_a * (lam ** 3) * (k2 ** 2) + phase_shift_rad
    amp = float(amplitude_contrast)
    return (-(max(0.0, 1.0 - amp * amp) ** 0.5) * torch.sin(chi) - amp * torch.cos(chi)).to(torch.float32)


def cistem_complex_lens_transfer_torch(
    shape: Tuple[int, int], pixel_size: float, kv: float, cs_mm: float,
    defocus_u: float, defocus_v: Optional[float] = None, defocus_angle_deg: float = 0.0,
    phase_shift_rad: float = 0.0, device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_real = ctf_2d_torch(shape, pixel_size, kv, cs_mm, defocus_u, defocus_v, defocus_angle_deg, 1.0, phase_shift_rad, device)
    h_imag = ctf_2d_torch(shape, pixel_size, kv, cs_mm, defocus_u, defocus_v, defocus_angle_deg, 0.0, phase_shift_rad, device)
    return torch.complex(h_real, h_imag)


def fresnel_propagator_torch(shape: Tuple[int, int], pixel_size: float, kv: float, dz_angstrom: float, device: torch.device) -> torch.Tensor:
    lam = electron_wavelength_angstrom(kv)
    _, _, k2 = frequency_grid_torch(shape, pixel_size, device)
    phase = math.pi * lam * float(dz_angstrom) * k2
    return torch.exp(1j * phase).to(torch.complex64)


def dqe_filter_torch(shape: Tuple[int, int], pixel_size: float, device: torch.device, root: bool = True) -> torch.Tensor:
    _, _, k2 = frequency_grid_torch(shape, pixel_size, device)
    freq = torch.sqrt(k2)
    out = torch.zeros_like(freq, dtype=torch.float32)
    for a, b, c in zip(DQE_A, DQE_B, DQE_C):
        out = out + float(a) * torch.exp(-((freq - float(b)) ** 2) / (2.0 * float(c) * float(c)))
    out = torch.clamp(out, min=0.0)
    mx = out.max()
    out = out / mx if mx > 0 else out
    if root:
        out = torch.sqrt(out)
    return out.to(torch.float32)


def apply_fourier_filter_torch(img: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(torch.fft.fft2(img) * filt).real.to(torch.float32)


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


def make_phase_amp_slabs_direct_from_atoms_numpy_cached(
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
        phase_slabs_np, amp_slabs_np, dz_list = make_phase_amp_slabs_direct_from_atoms_numpy_cached(atoms,cfg,)
    else:
        phase_slabs_np, amp_slabs_np, dz_list = make_phase_amp_slabs_direct_from_atoms_numpy(atoms,cfg,)
    phase_slabs, amp_slabs = numpy_slabs_to_torch(phase_slabs_np,amp_slabs_np,device=cfg.device if hasattr(cfg, "device") else "cuda",)
    if getattr(cfg, "radiation_damage", False) and getattr(cfg, "radiation_damage_where", "protein") == "protein":
        if cfg.verbose:
            print("Applying torch radiation-damage exposure filter to direct protein slabs...")
        phase_slabs = apply_radiation_damage_to_slabs_torch(phase_slabs, cfg)

    if cfg.explicit_water:
        wc = prepare_water_cache_torch(atoms, cfg)
        if wc is not None:
            phase_slabs, amp_slabs = fill_water_potential_torch(
                phase_slabs, amp_slabs, wc.water_coords, wc.distance_2d, dz_list, cfg, wc.templates
            )

    if getattr(cfg, "radiation_damage", False) and getattr(cfg, "radiation_damage_where", "protein") == "all":
        if cfg.verbose:
            print("Applying torch radiation-damage exposure filter to all direct phase slabs...")
        phase_slabs = apply_radiation_damage_to_slabs_torch(phase_slabs, cfg)

    return phase_slabs, amp_slabs, dz_list


def simulate_movie_from_direct_slabs_torch(atoms: List[Atom], cfg: SimConfig) -> torch.Tensor:
    """Per-frame movie simulation using direct atom-to-slab protein slabs."""
    n_frames = max(1, int(getattr(cfg, "number_of_frames", 1)))
    dose_per_frame = getattr(cfg, "dose_per_frame_e_per_a2", None)
    dose_per_frame = float(cfg.dose_e_per_a2) / float(n_frames) if dose_per_frame is None else float(dose_per_frame)
    pre = float(getattr(cfg, "pre_exposure_e_per_a2", 0.0))

    if cfg.verbose:
        print(
            f"Torch direct-slab per-frame simulation: frames={n_frames}, "
            f"dose_per_frame={dose_per_frame:.4f}, pre_exposure={pre:.4f}"
        )

    # Base protein slabs only. Radiation damage and explicit water are applied per frame.
#    base_phase_slabs, base_amp_slabs, dz_list = make_phase_amp_slabs_direct_from_atoms_torch(atoms, cfg)
    if getattr(cfg, "use_cache_atom", False):
        base_phase_slabs_np, base_amp_slabs_np, dz_list = make_phase_amp_slabs_direct_from_atoms_numpy_cached(atoms,cfg,)
    else:
        base_phase_slabs_np, base_amp_slabs_np, dz_list = make_phase_amp_slabs_direct_from_atoms_numpy(atoms,cfg,)
    base_phase_slabs, base_amp_slabs = numpy_slabs_to_torch(base_phase_slabs_np,base_amp_slabs_np,device=cfg.device if hasattr(cfg, "device") else "cuda",)
    water_cache = prepare_water_cache_torch(atoms, cfg)

    frames = []
    original_total_dose = float(cfg.dose_e_per_a2)
    for iframe in range(n_frames):
        d0 = pre + iframe * dose_per_frame
        d1 = d0 + dose_per_frame
        if cfg.verbose:
            print(f"Frame {iframe + 1}/{n_frames}: exposure {d0:.3f} -> {d1:.3f} e-/A^2")

        phase_slabs, amp_slabs = prepare_frame_slabs_from_base_torch(
            base_phase_slabs, base_amp_slabs, dz_list, cfg, water_cache,
            exposure_start_e_per_a2=d0, exposure_end_e_per_a2=d1,
            dose_per_frame_e_per_a2=dose_per_frame, iframe=iframe,
        )

        cfg.dose_e_per_a2 = dose_per_frame
        if cfg.mode == "projection":
            img = simulate_projection_from_slabs_torch(phase_slabs, cfg)
            
        elif cfg.mode == "multislice":
            img = propagate_slabs_cistem_like_torch(phase_slabs, amp_slabs, dz_list, cfg)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")
        frames.append(img.to(torch.float32))

    cfg.dose_e_per_a2 = original_total_dose
    stack = torch.stack(frames, dim=0).to(torch.float32)
    if getattr(cfg, "save_frames", False):
        return stack
    summed = stack.sum(dim=0).to(torch.float32)
    if getattr(cfg, "normalize_frame_sum", True):
        summed = summed / float(n_frames)
    return summed

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
    bf = 0.25 * float(cfg.water_bfactor)
    xs = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.float32)
    ys = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.float32)
    zs = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.float32)
    templates = []
    center = (subpix_n - 1) / 2.0
    for sz in range(subpix_n):
        for sy in range(subpix_n):
            for sx in range(subpix_n):
                dx = (sx - center) / subpix_n
                dy = (sy - center) / subpix_n
                dz = (sz - center) / subpix_n
                x1v = (xs - dx) * ps - 0.5 * ps
                y1v = (ys - dy) * ps - 0.5 * ps
                z1v = (zs - dz) * ps - 0.5 * ps
                pot3d = voxel_integrated_potential_torch(x1v, x1v + ps, y1v, y1v + ps, z1v, z1v + ps, ai, bf, lead_term, device)
                templates.append(pot3d.sum(dim=0).to(torch.float32))
    return torch.stack(templates, dim=0)


@dataclass
class TorchWaterCache:
    templates: torch.Tensor
    water_coords: np.ndarray
    distance_2d: Optional[torch.Tensor]


def prepare_water_cache_torch(atoms: List[Atom], cfg: SimConfig) -> Optional[TorchWaterCache]:
    if not cfg.explicit_water:
        return None
    if cfg.solvent:
        raise ValueError("Do not use --solvent and --explicit-water together.")
    if cfg.verbose:
        print("Generating explicit waters with CPU KDTree exclusion...")
    water_coords = generate_explicit_water_coords(cfg, atoms=atoms, exclude_from_atoms=True)
    if cfg.verbose:
        print("Precomputing projected water templates on torch device...")
    templates = precompute_projected_water_templates_torch(cfg)
    distance_2d = nearest_atom_distance_2d_torch(atoms, cfg, max_r_a=9.0) if cfg.water_soft_weight else None
    return TorchWaterCache(templates=templates, water_coords=water_coords, distance_2d=distance_2d)


def fill_water_potential_torch(
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
    subpix_n = int(round(templates.shape[0] ** (1.0 / 3.0)))
    if subpix_n ** 3 != templates.shape[0]:
        raise ValueError(f"Number of water templates must be subpix_n^3, got {templates.shape[0]}")
    radius_pix = templates.shape[-1] // 2
    dz_arr = torch.as_tensor(dz_list, device=device, dtype=torch.float32).contiguous()
    total_z_a = float(dz_arr.sum().detach().cpu())
    z_edges = torch.empty(n_slices + 1, device=device, dtype=torch.float32)
    z_edges[0] = -0.5 * total_z_a
    z_edges[1:] = z_edges[0] + torch.cumsum(dz_arr, dim=0)
    z_edges = z_edges.contiguous()

    # coords[:, i] is a strided view. Make these contiguous once so
    # torch.searchsorted does not create an internal temporary copy for millions of waters.
    x_a = coords[:, 0].contiguous()
    y_a = coords[:, 1].contiguous()
    z_a = coords[:, 2].contiguous()

    slab = torch.searchsorted(z_edges, z_a, right=True) - 1
    valid = (slab >= 0) & (slab < n_slices)
    if not bool(valid.any()):
        return phase_slabs, amp_slabs

    slab = slab[valid].long().contiguous()
    x_valid = x_a[valid].contiguous()
    y_valid = y_a[valid].contiguous()
    z_valid = z_a[valid].contiguous()

    x_pix = x_valid / ps + nx / 2.0
    y_pix = y_valid / ps + ny / 2.0
    ix_round = torch.round(x_pix).long()
    iy_round = torch.round(y_pix).long()
    ix = torch.remainder(ix_round, nx)
    iy = torch.remainder(iy_round, ny)
    fx = x_pix - ix_round.to(torch.float32)
    fy = y_pix - iy_round.to(torch.float32)
    local_z_pix = (z_valid - z_edges[slab]) / ps
    iz_int = torch.round(local_z_pix).long()
    fz = local_z_pix - iz_int.to(torch.float32)
    sx = torch.clamp(torch.floor((fx + 0.5) * subpix_n), 0, subpix_n - 1).long()
    sy = torch.clamp(torch.floor((fy + 0.5) * subpix_n), 0, subpix_n - 1).long()
    sz = torch.clamp(torch.floor((fz + 0.5) * subpix_n), 0, subpix_n - 1).long()
    template_index = sz * subpix_n * subpix_n + sy * subpix_n + sx
    weights = torch.ones_like(x_pix, dtype=torch.float32)
    if distance_2d is not None and getattr(cfg, "water_soft_weight", False):
        d = distance_2d[iy, ix]
        finite = torch.isfinite(d)
        weights[finite] = hydration_weight_torch(d[finite], ps)
    ratio = math.sqrt(float(inelastic_scalar_water) / 10.0)
    yoff = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.long)
    xoff = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.long)
    yy, xx = torch.meshgrid(yoff, xoff, indexing="ij")
    h, w = yy.shape
    n_added = 0
    for s in range(n_slices):
        mask_s = slab == s
        if not bool(mask_s.any()):
            continue
        tmpl_unique = torch.unique(template_index[mask_s])
        phase_flat = phase_slabs[s].reshape(-1)
        amp_flat = amp_slabs[s].reshape(-1)
        for t in tmpl_unique.tolist():
            mask = mask_s & (template_index == int(t))
            idx_all = torch.nonzero(mask, as_tuple=False).flatten()
            if idx_all.numel() == 0:
                continue
            template = templates[int(t)].to(torch.float32)
            for st in range(0, idx_all.numel(), chunk_size):
                ids = idx_all[st:st+chunk_size]
                cy = iy[ids]
                cx = ix[ids]
                flat_idx = (torch.remainder(cy[:, None, None] + yy[None, :, :], ny) * nx +
                            torch.remainder(cx[:, None, None] + xx[None, :, :], nx)).reshape(-1)
                vals = (weights[ids, None, None] * template[None, :, :]).reshape(-1)
                phase_flat.scatter_add_(0, flat_idx, vals)
                amp_flat.scatter_add_(0, flat_idx, vals * ratio)
                n_added += int(ids.numel())
    if getattr(cfg, "verbose", False):
        n_outside = int((~valid).sum().detach().cpu())
        print(f"fill_water_potential_torch: added={n_added}, outside_z={n_outside}")
    return phase_slabs, amp_slabs


def finalize_detector_image_torch(img: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    out = img.to(torch.float32)
    if cfg.dqe:
        mean = out.mean()
        contrast = out - mean
        contrast = apply_fourier_filter_torch(contrast, dqe_filter_torch(tuple(out.shape), cfg.pixel_size, out.device, root=True))
        out = mean + contrast
    if cfg.poisson:
        if cfg.seed is not None:
            torch.manual_seed(int(cfg.seed))
            if out.device.type == "cuda":
                torch.cuda.manual_seed_all(int(cfg.seed))
        electrons_per_pixel = float(cfg.dose_e_per_a2) * (float(cfg.pixel_size) ** 2)
        mean = out.mean()
        norm_intensity = out / (mean + 1e-12)
        out = torch.poisson(torch.clamp(norm_intensity * electrons_per_pixel, min=0.0)).to(torch.float32)
    return out


def simulate_projection_from_slabs_torch(phase_slabs: Sequence[torch.Tensor], cfg: SimConfig) -> torch.Tensor:
    proj = torch.stack(list(phase_slabs), dim=0).sum(dim=0).to(torch.float32)
    ctf = ctf_2d_torch(tuple(proj.shape), cfg.pixel_size, cfg.kv, cfg.cs_mm, cfg.defocus_u,
                       cfg.defocus_v, cfg.defocus_angle_deg, cfg.amplitude_contrast,
                       cfg.phase_shift_rad, proj.device)
    contrast = apply_fourier_filter_torch(proj, ctf)
    img = 1.0 + contrast
    return finalize_detector_image_torch(img, cfg)


def propagate_slabs_cistem_like_torch(phase_slabs, amp_slabs, dz_list, cfg: SimConfig) -> torch.Tensor:
    device = phase_slabs[0].device
    wave = torch.ones_like(phase_slabs[0], dtype=torch.complex64, device=device)
    for phase, amp, dz in zip(phase_slabs, amp_slabs, dz_list):
        phase0 = phase.to(torch.float32)
        phase0 = phase0 - edge_mean_2d_torch(phase0)
        amp0 = amp.to(torch.float32)
        transmission = torch.exp(torch.complex(-amp0, phase0)).to(torch.complex64)
        wave = wave * transmission
        prop = fresnel_propagator_torch(tuple(phase0.shape), cfg.pixel_size, cfg.kv, dz, device)
        wave = torch.fft.ifft2(torch.fft.fft2(wave) * prop).to(torch.complex64)
    lens = cistem_complex_lens_transfer_torch(tuple(wave.shape), cfg.pixel_size, cfg.kv, cfg.cs_mm,
                                              cfg.defocus_u, cfg.defocus_v, cfg.defocus_angle_deg,
                                              cfg.phase_shift_rad, device)
    image_wave = torch.fft.ifft2(torch.fft.fft2(wave) * lens).to(torch.complex64)
    img = image_wave.real ** 2 + image_wave.imag ** 2
    return finalize_detector_image_torch(img.to(torch.float32), cfg)


def prepare_phase_amp_slabs_torch(vol: torch.Tensor, atoms: List[Atom], cfg: SimConfig):
    phase_slabs, amp_slabs, dz_list = volume_to_slabs_torch(vol, cfg.n_slices, cfg.pixel_size)
    if getattr(cfg, "radiation_damage", False) and getattr(cfg, "radiation_damage_where", "protein") == "protein":
        if cfg.verbose:
            print("Applying torch radiation-damage exposure filter to protein slabs...")
        phase_slabs = apply_radiation_damage_to_slabs_torch(phase_slabs, cfg)
    if cfg.explicit_water:
        if cfg.solvent:
            raise ValueError("Do not use --solvent and --explicit-water together.")
        wc = prepare_water_cache_torch(atoms, cfg)
        if wc is not None:
            phase_slabs, amp_slabs = fill_water_potential_torch(phase_slabs, amp_slabs, wc.water_coords, wc.distance_2d, dz_list, cfg, wc.templates)
    if getattr(cfg, "radiation_damage", False) and getattr(cfg, "radiation_damage_where", "protein") == "all":
        if cfg.verbose:
            print("Applying torch radiation-damage exposure filter to all phase slabs...")
        phase_slabs = apply_radiation_damage_to_slabs_torch(phase_slabs, cfg)
    return phase_slabs, amp_slabs, dz_list


def prepare_frame_slabs_from_base_torch(
    base_phase_slabs: List[torch.Tensor], base_amp_slabs: List[torch.Tensor], dz_list: List[float], cfg: SimConfig,
    water_cache: Optional[TorchWaterCache], exposure_start_e_per_a2: float, exposure_end_e_per_a2: float,
    dose_per_frame_e_per_a2: float, iframe: int,
):
    phase_slabs = [s.clone() for s in base_phase_slabs]
    amp_slabs = [s.clone() for s in base_amp_slabs]
    if getattr(cfg, "radiation_damage", False):
        phase_slabs = apply_radiation_damage_to_slabs_torch(phase_slabs, cfg, exposure_start_e_per_a2, exposure_end_e_per_a2)
    if water_cache is not None:
        frame_water_coords = water_cache.water_coords
        if getattr(cfg, "shake_waters", False):
            seed = None if cfg.seed is None else int(cfg.seed) + int(iframe) + 1000003
            rng = np.random.default_rng(seed)
            frame_water_coords = shake_waters_3d(water_cache.water_coords, cfg, dose_per_frame_e_per_a2, rng)
        phase_slabs, amp_slabs = fill_water_potential_torch(
            phase_slabs, amp_slabs, frame_water_coords, water_cache.distance_2d, dz_list, cfg, water_cache.templates
        )
    return phase_slabs, amp_slabs


def simulate_movie_from_volume_torch(vol: torch.Tensor, atoms: List[Atom], cfg: SimConfig) -> torch.Tensor:
    n_frames = max(1, int(getattr(cfg, "number_of_frames", 1)))
    dose_per_frame = getattr(cfg, "dose_per_frame_e_per_a2", None)
    dose_per_frame = float(cfg.dose_e_per_a2) / float(n_frames) if dose_per_frame is None else float(dose_per_frame)
    pre = float(getattr(cfg, "pre_exposure_e_per_a2", 0.0))
    if cfg.verbose:
        print(f"Torch per-frame simulation: frames={n_frames}, dose_per_frame={dose_per_frame:.4f}, pre_exposure={pre:.4f}")
    base_phase_slabs, base_amp_slabs, dz_list = volume_to_slabs_torch(vol, cfg.n_slices, cfg.pixel_size)
    water_cache = prepare_water_cache_torch(atoms, cfg)
    frames = []
    original_total_dose = float(cfg.dose_e_per_a2)
    for iframe in range(n_frames):
        d0 = pre + iframe * dose_per_frame
        d1 = d0 + dose_per_frame
        if cfg.verbose:
            print(f"Frame {iframe + 1}/{n_frames}: exposure {d0:.3f} -> {d1:.3f} e-/A^2")
        phase_slabs, amp_slabs = prepare_frame_slabs_from_base_torch(
            base_phase_slabs, base_amp_slabs, dz_list, cfg, water_cache,
            exposure_start_e_per_a2=d0, exposure_end_e_per_a2=d1,
            dose_per_frame_e_per_a2=dose_per_frame, iframe=iframe,
        )
        cfg.dose_e_per_a2 = dose_per_frame
        if cfg.mode == "projection":
            img = simulate_projection_from_slabs_torch(phase_slabs, cfg)
        elif cfg.mode == "multislice":
            img = propagate_slabs_cistem_like_torch(phase_slabs, amp_slabs, dz_list, cfg)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")
        frames.append(img.to(torch.float32))
    cfg.dose_e_per_a2 = original_total_dose
    stack = torch.stack(frames, dim=0).to(torch.float32)
    if getattr(cfg, "save_frames", False):
        return stack
    summed = stack.sum(dim=0).to(torch.float32)
    if getattr(cfg, "normalize_frame_sum", True):
        summed = summed / float(n_frames)
    return summed



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

    if cfg.verbose:
        print(f"Loaded atoms: {len(atoms)}")
        if cfg.euler_rot_deg or cfg.euler_tilt_deg or cfg.euler_psi_deg:
            direction = "inverse/passive" if cfg.euler_inverse else "active"
            print(
                f"Applied ZYZ Euler rotation: rot={cfg.euler_rot_deg}, "
                f"tilt={cfg.euler_tilt_deg}, psi={cfg.euler_psi_deg} deg ({direction})"
            )
        print(f"Using direct atom-to-slab protein potential on {torch_device_from_cfg(cfg)}")

    if getattr(cfg, "per_frame", False):
        img_t = simulate_movie_from_direct_slabs_torch(atoms, cfg)
    else:
        phase_slabs, amp_slabs, dz_list = prepare_phase_amp_slabs_direct_torch(atoms, cfg)
        if cfg.mode == "projection":
            if cfg.verbose:
                print("Simulating torch projection from direct slabs...")
            img_t = simulate_projection_from_slabs_torch(phase_slabs, cfg)
            #img_t = simulate_raw_projection_from_slabs_torch(phase_slabs, cfg)
        elif cfg.mode == "multislice":
            if cfg.verbose:
                print("Running torch cisTEM-like multislice from direct slabs...")
            img_t = propagate_slabs_cistem_like_torch(phase_slabs, amp_slabs, dz_list, cfg)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

    img_np = torch_to_numpy(img_t)
    save_image_or_volume(output_path, img_np, cfg.pixel_size)
    if cfg.verbose:
        print(f"Saved {output_path}")
    return img_np

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyTorch/GPU version of cisTEM simulate Python scaffold with simplified explicit water.")
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
    p.add_argument("--radiation-damage", action="store_true", help="Apply Grant/Grigorieff-style radiation damage exposure filter to projected slabs")
    p.add_argument("--pre-exposure", type=float, default=0.0, help="Pre-exposure in e-/A^2 before this simulated image/frame")
    p.add_argument("--radiation-damage-where", choices=["protein", "all"], default="protein", help="Apply radiation damage to protein slabs before water, or to all phase slabs after water")
    p.add_argument("--exposure-filter-modify-signal", type=int, choices=[0, 1, 2], default=0, help="0: multiply filter, 1: 2F/(1+F), 2: sqrt(F)")
    p.add_argument("--per-frame", action="store_true", help="Simulate movie frames separately and sum/average them")
    p.add_argument("--number-of-frames", type=int, default=1)
    p.add_argument("--dose-per-frame", type=float, default=None, help="Dose per frame in e-/A^2. If omitted, use --dose / --number-of-frames")
    p.add_argument("--use-cache-atom", action="store_true", help="Use cached atomic potential. Speed up 3x but with less accuracy")
    p.add_argument("--save-frames", action="store_true", help="Save all frames as a stack instead of summed/averaged image")
    p.add_argument("--no-normalize-frame-sum", action="store_true", help="Do not divide summed frames by number of frames")
    p.add_argument("--shake-waters", action="store_true", help="Apply cisTEM-like random water displacement per frame")
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
        use_torch=True, device=a.device,
    )
    run(cfg, a.pdb, a.output)


if __name__ == "__main__":
    main()
