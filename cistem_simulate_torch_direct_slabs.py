"""Lean PyTorch implementation of the cisTEM simulate direct-slab path.

Required dependencies: NumPy, PyTorch, and mrcfile. The production path is
PyTorch-only for atom slabs, explicit-water generation, water shaking, water
splatting, imaging, and the true 3D hydration soft-weight atlas. Legacy SciPy,
NumPy simulation, scatter-splat, CPU-shake, full-volume, and backend fallback
paths have been removed.

Water projection uses only the Q^2 x/y subpixel templates and a fused
slab-as-batch/template-as-channel convolution. Cached protein slab generation
keeps protein-frame atom tensors resident across orientations and constructs
atom/slab metadata with vectorized device operations.

The public main(argv) entry point is retained for persistent-worker runners."""

# Changelog
# Add --ice-thickness (in Angstrom) to cube, the old version use the box size as the Z-length of cube.
# You can also change the position of protein to upper or lower surface of ice, default is center.
# Fix the old version enabling --water-soft-weight produces severe edging artifacts.
# Fix --shake-water using voxel instead of angstrom.
# Fix when running in single frame mode, dose does not apply.
# Fix wrong equation for hydration-weight 
# Remove the old unused functions.
# Speed up by 3 times. Now it takes 4 seconds on an RTX 6000 Ada to generate a movie, comparing to the old version: 11 seconds.

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import mrcfile
import numpy as np
import torch
import torch.nn.functional as F

PROGRAM_VERSION = "hydration-atlas-fused-water-tensor-atoms-1.1"
SOURCE_BASE_SHA256 = "bfdb2c7f945063315ca928558d2316bec5726c5000bb772f3046cde937930d7c"
OPTIMIZATION_BASE_SHA256 = "9959b60d99aaca2add219a5e91bc2648e6b7ddaa8491da307ed57a6dcf089722"
BOND_SCALING_DEFAULT = 1.043
WN = 0.8045 * 0.79
ATOM_INDEX: Dict[str, int] = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'NA': 5, 'MG': 6, 'P': 7, 'S': 8, 'CL': 9, 'K': 10, 'CA': 11, 'MN': 12, 'FE': 13, 'ZN': 14, 'H2O': 15, 'O-': 16}
ATOMIC_NUMBER = np.array([1, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20, 25, 26, 30, 10, 8], dtype=np.float32)
SCATTERING_A = np.array([[0.0349, 0.1201, 0.197, 0.0573, 0.1195], [0.0893, 0.2563, 0.757, 1.0487, 0.3575], [0.1022, 0.3219, 0.7982, 0.8197, 0.1715], [0.0974, 0.2921, 0.691, 0.699, 0.2039], [0.1083, 0.3175, 0.6487, 0.5846, 0.1421], [0.2142, 0.6853, 0.7692, 1.6589, 1.4482], [0.2314, 0.6866, 0.9677, 2.1882, 1.1339], [0.2548, 0.6106, 1.4541, 2.3204, 0.8477], [0.2497, 0.5628, 1.3899, 2.1865, 0.7715], [0.2443, 0.5397, 1.3919, 2.0197, 0.6621], [0.4115, -1.4031, 2.2784, 2.6742, 2.2162], [0.4054, 1.388, 2.1602, 3.7532, 2.2063], [0.3796, 1.2094, 1.7815, 2.542, 1.5937], [0.3946, 1.2725, 1.7031, 2.314, 1.4795], [0.4288, 1.2646, 1.4472, 1.8294, 1.0934], [WN * 0.07967, WN * 0.1053, WN * 0.2933, WN * 0.6831, WN * 1.304], [0.205, 0.628, 1.17, 1.03, 0.29]], dtype=np.float64)
SCATTERING_B = np.array([[0.5347, 3.5867, 12.347, 18.9525, 38.6269], [0.2465, 1.71, 6.4094, 18.6113, 50.2523], [0.2451, 1.7481, 6.1925, 17.3894, 48.1431], [0.2067, 1.3815, 4.6943, 12.7105, 32.4726], [0.2057, 1.3439, 4.2788, 11.3932, 28.7881], [0.3334, 2.3446, 10.083, 48.3037, 138.27], [0.3278, 2.272, 10.924, 39.2898, 101.9748], [0.2908, 1.874, 8.5176, 24.3434, 63.2996], [0.2681, 1.6711, 7.0267, 19.5377, 50.3888], [0.2468, 1.5242, 6.1537, 16.6687, 42.3086], [0.3703, 3.3874, 13.1029, 68.9592, 194.4329], [0.3499, 3.0991, 11.9608, 53.9353, 142.3892], [0.2699, 2.0455, 7.4726, 31.0604, 91.5622], [0.2717, 2.0443, 7.6007, 29.9714, 86.2265], [0.2593, 1.7998, 6.75, 25.586, 73.5284], [WN * 4.718, WN * 16.75, WN * 0.4524, WN * 13.43, WN * 4.448], [0.397, 2.64, 8.8, 27.1, 91.8]], dtype=np.float64)
HYDRATION_RADIUS_EXTRA_SHIFT = -0.5
HYDRATION_RADIUS_VALS = np.array([0.175, -0.135, 2.23, 3.43, 4.78, 1.0, 1.77, 0.955], dtype=np.float64)
PUSH_BACK_BY = -1.48
DQE_A = np.array([-0.01516, -0.5662, -0.09731, -0.01551, 21.47], dtype=np.float64)
DQE_B = np.array([0.02671, -0.02504, 0.162, 0.2831, -2.28], dtype=np.float64)
DQE_C = np.array([0.01774, 0.1441, 0.1082, 0.07916, 1.372], dtype=np.float64)
WATER_DENSITY_PER_A3 = 0.94 * 0.6022140857 / 18.01528
INELASTIC_SCALAR_WATER = 0.0725
VDW_RADIUS_A: Dict[str, float] = {'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'NA': 2.27, 'MG': 1.73, 'P': 1.8, 'S': 1.8, 'CL': 1.75, 'K': 2.75, 'CA': 2.31, 'MN': 1.79, 'FE': 1.56, 'ZN': 1.39, 'H2O': 1.52, 'O-': 1.52}

@dataclass
class Atom:
    element: str
    xyz: np.ndarray
    bfactor: float = 0.0
    occupancy: float = 1.0

class TimingRecorder:

    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self.stage_seconds: OrderedDict[str, float] = OrderedDict()
        self.stage_calls: OrderedDict[str, int] = OrderedDict()
        self.counters: OrderedDict[str, float] = OrderedDict()
        self.meta: OrderedDict[str, object] = OrderedDict()
        self.total_wall_s = 0.0
        self._started_at: Optional[float] = None
        if self.enabled and self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def synchronize(self) -> None:
        if self.enabled and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def start(self) -> None:
        if not self.enabled:
            return
        self.synchronize()
        self._started_at = time.perf_counter()

    def stop(self) -> None:
        if not self.enabled or self._started_at is None:
            return
        self.synchronize()
        self.total_wall_s = time.perf_counter() - self._started_at

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter() - start
            if name not in self.stage_seconds:
                self.stage_seconds[name] = 0.0
                self.stage_calls[name] = 0
            self.stage_seconds[name] += elapsed
            self.stage_calls[name] += 1

    def add_counter(self, name: str, value: float) -> None:
        if self.enabled:
            self.counters[name] = float(self.counters.get(name, 0.0)) + float(value)

    def set_counter(self, name: str, value: float) -> None:
        if self.enabled:
            self.counters[name] = float(value)

    def set_meta(self, name: str, value: object) -> None:
        if self.enabled:
            self.meta[name] = value

    def capture_cuda_memory(self) -> None:
        if not self.enabled or self.device.type != 'cuda':
            return
        self.synchronize()
        self.meta['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated(self.device) / 1048576.0
        self.meta['cuda_peak_reserved_mib'] = torch.cuda.max_memory_reserved(self.device) / 1048576.0

    def as_dict(self) -> dict:
        stages = []
        for name, seconds in self.stage_seconds.items():
            calls = int(self.stage_calls[name])
            stages.append({'name': name, 'seconds': float(seconds), 'calls': calls, 'mean_seconds': float(seconds) / max(calls, 1), 'percent_of_wall': 100.0 * float(seconds) / max(self.total_wall_s, 1e-20)})
        return {'schema': 'cistem_simulate_timing_v1', 'total_wall_seconds': float(self.total_wall_s), 'stages': stages, 'counters': dict(self.counters), 'meta': dict(self.meta)}

    def emit(self, output_path: str | Path, json_pattern: Optional[str]) -> None:
        if not self.enabled:
            return
        print(f'[timing] total_wall={self.total_wall_s:.6f} s', flush=True)
        for name, seconds in self.stage_seconds.items():
            calls = self.stage_calls[name]
            percent = 100.0 * seconds / max(self.total_wall_s, 1e-20)
            print(f'[timing] {name:38s} {seconds:12.6f} s  calls={calls:4d}  wall={percent:7.2f}%', flush=True)
        if self.counters:
            print('[timing] counters ' + ', '.join((f'{key}={value:g}' for key, value in self.counters.items())), flush=True)
        if json_pattern:
            output = Path(output_path)
            resolved = str(json_pattern).format(pid=os.getpid(), output_stem=output.stem, output_name=output.name)
            json_path = Path(resolved)
            if json_path.suffix.lower() != '.json':
                json_path = json_path / f'{output.stem}_timing.json'
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=False))
            print(f'[timing] wrote {json_path}', flush=True)

def get_timing_recorder(cfg: SimConfig) -> Optional[TimingRecorder]:
    recorder = getattr(cfg, '_timing_recorder', None)
    if recorder is None or not recorder.enabled:
        return None
    return recorder

def timing_section(cfg: SimConfig, name: str):
    recorder = get_timing_recorder(cfg)
    return nullcontext() if recorder is None else recorder.section(name)

@dataclass
class HydrationAtlas:
    weights: torch.Tensor
    origin_xyz: torch.Tensor
    shape_xyz: Tuple[int, int, int]
    spacing_a: float
    cutoff_a: float

@dataclass
class OrientedHydrationAtlas:
    atlas: HydrationAtlas
    protein_to_world: torch.Tensor
    world_to_protein_row: torch.Tensor
    translation_xyz: torch.Tensor
    world_aabb_min: torch.Tensor
    world_aabb_max: torch.Tensor
_HYDRATION_ATLAS_CACHE: OrderedDict[Tuple, HydrationAtlas] = OrderedDict()


@dataclass
class ProteinTensorData:
    """Protein-frame atom tensors cached once per persistent worker/device."""
    protein_xyz: torch.Tensor
    atom_index: torch.Tensor
    occupancy: torch.Tensor
    bfactor: torch.Tensor
    fingerprint: str


@dataclass
class OrientedProteinTensors:
    """Orientation-specific atom coordinates with static atom fields reused."""
    xyz: torch.Tensor
    atom_index: torch.Tensor
    occupancy: torch.Tensor
    bfactor: torch.Tensor


_PROTEIN_TENSOR_CACHE: OrderedDict[Tuple, ProteinTensorData] = OrderedDict()
_ATOM_XY_KERNEL_CACHE: OrderedDict[Tuple, torch.Tensor] = OrderedDict()
_ATOM_Z_PREFIX_CACHE: OrderedDict[Tuple, torch.Tensor] = OrderedDict()


def _protein_static_arrays(atoms: Sequence[Atom]) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray([atom.xyz for atom in atoms], dtype=np.float32)
    atom_index = np.asarray([ATOM_INDEX[atom.element.upper()] for atom in atoms], dtype=np.int16)
    occupancy = np.asarray([atom.occupancy for atom in atoms], dtype=np.float32)
    bfactor = np.asarray([atom.bfactor for atom in atoms], dtype=np.float32)
    digest_builder = hashlib.blake2b(digest_size=16)
    for array in (coords, atom_index, occupancy, bfactor):
        digest_builder.update(array.tobytes(order='C'))
    return digest_builder.hexdigest(), coords, atom_index, occupancy, bfactor


def get_or_build_protein_tensor_data(
    atoms: Sequence[Atom],
    cfg: SimConfig,
    device: torch.device,
) -> Tuple[ProteinTensorData, bool]:
    """Return persistent protein-frame tensors and whether the LRU cache hit."""
    digest, coords, atom_index, occupancy, bfactor = _protein_static_arrays(atoms)
    key = (str(torch.device(device)), digest, int(coords.shape[0]))
    cached = _PROTEIN_TENSOR_CACHE.get(key)
    recorder = get_timing_recorder(cfg)
    if cached is not None:
        _PROTEIN_TENSOR_CACHE.move_to_end(key)
        if recorder is not None:
            recorder.add_counter('protein_tensor_cache_hits', 1)
        return cached, True
    data = ProteinTensorData(
        protein_xyz=torch.as_tensor(coords, device=device, dtype=torch.float32).contiguous(),
        atom_index=torch.as_tensor(atom_index.astype(np.int64), device=device, dtype=torch.long).contiguous(),
        occupancy=torch.as_tensor(occupancy, device=device, dtype=torch.float32).contiguous(),
        bfactor=torch.as_tensor(bfactor, device=device, dtype=torch.float32).contiguous(),
        fingerprint=digest,
    )
    _PROTEIN_TENSOR_CACHE[key] = data
    _PROTEIN_TENSOR_CACHE.move_to_end(key)
    max_entries = max(1, int(getattr(cfg, 'protein_tensor_cache_entries', 2)))
    while len(_PROTEIN_TENSOR_CACHE) > max_entries:
        _PROTEIN_TENSOR_CACHE.popitem(last=False)
    if recorder is not None:
        recorder.add_counter('protein_tensor_cache_misses', 1)
        recorder.set_counter('protein_tensor_count', int(data.protein_xyz.shape[0]))
        tensor_bytes = (
            data.protein_xyz.numel() * data.protein_xyz.element_size()
            + data.atom_index.numel() * data.atom_index.element_size()
            + data.occupancy.numel() * data.occupancy.element_size()
            + data.bfactor.numel() * data.bfactor.element_size()
        )
        recorder.set_counter('protein_tensor_mib', tensor_bytes / 1048576.0)
    return data, False


def orient_protein_tensors(
    data: ProteinTensorData,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> OrientedProteinTensors:
    """Apply the current Euler rotation and ice-z translation on the device."""
    rotation_t = torch.as_tensor(rotation, device=data.protein_xyz.device, dtype=torch.float32)
    translation_t = torch.as_tensor(translation, device=data.protein_xyz.device, dtype=torch.float32)
    xyz = data.protein_xyz @ rotation_t.T
    xyz = (xyz + translation_t[None, :]).contiguous()
    return OrientedProteinTensors(
        xyz=xyz,
        atom_index=data.atom_index,
        occupancy=data.occupancy,
        bfactor=data.bfactor,
    )


def protein_tensors_from_oriented_atoms(atoms: Sequence[Atom], device: torch.device) -> OrientedProteinTensors:
    """Build orientation-specific tensors for direct function callers without a cache."""
    _, coords, atom_index, occupancy, bfactor = _protein_static_arrays(atoms)
    return OrientedProteinTensors(
        xyz=torch.as_tensor(coords, device=device, dtype=torch.float32).contiguous(),
        atom_index=torch.as_tensor(atom_index.astype(np.int64), device=device, dtype=torch.long).contiguous(),
        occupancy=torch.as_tensor(occupancy, device=device, dtype=torch.float32).contiguous(),
        bfactor=torch.as_tensor(bfactor, device=device, dtype=torch.float32).contiguous(),
    )

def clone_atoms(atoms: Sequence[Atom]) -> List[Atom]:
    return [Atom(atom.element, np.asarray(atom.xyz, dtype=np.float64).copy(), float(atom.bfactor), float(atom.occupancy)) for atom in atoms]

def effective_euler_rotation_matrix(cfg: SimConfig) -> np.ndarray:
    rotation = rotation_matrix_zyz_relion(cfg.euler_rot_deg, cfg.euler_tilt_deg, cfg.euler_psi_deg)
    if cfg.euler_inverse:
        rotation = rotation.T
    return np.asarray(rotation, dtype=np.float64)

def apply_rotation_matrix_to_atoms(atoms: Sequence[Atom], rotation: np.ndarray) -> None:
    rotation = np.asarray(rotation, dtype=np.float64)
    for atom in atoms:
        atom.xyz = rotation @ np.asarray(atom.xyz, dtype=np.float64)

def _protein_coordinate_fingerprint(atoms: Sequence[Atom]) -> Tuple[str, np.ndarray]:
    coords = np.asarray([atom.xyz for atom in atoms], dtype=np.float32)
    digest = hashlib.blake2b(coords.tobytes(order='C'), digest_size=16).hexdigest()
    return (digest, coords)

def _hydration_atlas_cache_key(atoms: Sequence[Atom], cfg: SimConfig, device: torch.device) -> Tuple:
    digest, coords = _protein_coordinate_fingerprint(atoms)
    return (str(device), digest, int(coords.shape[0]), float(cfg.pixel_size), float(cfg.water_soft_atlas_spacing_a), float(cfg.water_soft_atlas_cutoff_a))

def build_hydration_atlas(atoms: Sequence[Atom], cfg: SimConfig, device: torch.device) -> HydrationAtlas:
    if not hasattr(torch.Tensor, 'scatter_reduce_'):
        raise RuntimeError('This hydration-atlas implementation requires torch.Tensor.scatter_reduce_().')
    spacing = float(cfg.water_soft_atlas_spacing_a)
    cutoff = float(cfg.water_soft_atlas_cutoff_a)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError('water_soft_atlas_spacing_a must be finite and > 0')
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError('water_soft_atlas_cutoff_a must be finite and > 0')
    atom_np = np.asarray([atom.xyz for atom in atoms], dtype=np.float32)
    if atom_np.size == 0:
        raise ValueError('Cannot build a hydration atlas without atoms')
    margin = cutoff + spacing
    origin_np = np.floor((atom_np.min(axis=0) - margin) / spacing) * spacing
    maximum_np = np.ceil((atom_np.max(axis=0) + margin) / spacing) * spacing
    shape_xyz_np = np.rint((maximum_np - origin_np) / spacing).astype(np.int64) + 1
    if np.any(shape_xyz_np <= 0):
        raise ValueError(f'Invalid hydration atlas shape: {shape_xyz_np.tolist()}')
    nx, ny, nz = [int(value) for value in shape_xyz_np]
    voxel_count = int(nx) * int(ny) * int(nz)
    if voxel_count >= 2 ** 63:
        raise ValueError('Hydration atlas is too large for 64-bit indexing')
    atom_xyz = torch.as_tensor(atom_np, device=device, dtype=torch.float32)
    origin = torch.as_tensor(origin_np, device=device, dtype=torch.float32)
    dist2_flat = torch.full((voxel_count,), float('inf'), device=device, dtype=torch.float32)
    offset_radius = int(math.ceil(cutoff / spacing + math.sqrt(3.0) * 0.5))
    axis = torch.arange(-offset_radius, offset_radius + 1, device=device, dtype=torch.long)
    oz, oy, ox = torch.meshgrid(axis, axis, axis, indexing='ij')
    offsets = torch.stack((ox, oy, oz), dim=-1).reshape(-1, 3)
    offset_limit = cutoff + math.sqrt(3.0) * 0.5 * spacing + 1e-06
    offsets = offsets[offsets.to(torch.float32).square().sum(dim=1).sqrt() * spacing <= offset_limit]
    chunk_size = max(1, int(cfg.water_soft_atlas_atom_chunk_size))
    cutoff2 = cutoff * cutoff
    for start in range(0, int(atom_xyz.shape[0]), chunk_size):
        atom_chunk = atom_xyz[start:start + chunk_size]
        center = torch.floor((atom_chunk - origin) / spacing + 0.5).long()
        grid_index = center[:, None, :] + offsets[None, :, :]
        valid = (grid_index[..., 0] >= 0) & (grid_index[..., 0] < nx) & (grid_index[..., 1] >= 0) & (grid_index[..., 1] < ny) & (grid_index[..., 2] >= 0) & (grid_index[..., 2] < nz)
        grid_xyz = origin[None, None, :] + grid_index.to(torch.float32) * spacing
        d2 = (grid_xyz - atom_chunk[:, None, :]).square().sum(dim=-1)
        valid &= d2 <= cutoff2
        if valid.any():
            selected = grid_index[valid]
            flat = selected[:, 2] * (ny * nx) + selected[:, 1] * nx + selected[:, 0]
            dist2_flat.scatter_reduce_(0, flat.long(), d2[valid].to(torch.float32), reduce='amin', include_self=True)
    finite = torch.isfinite(dist2_flat)
    weights = torch.ones_like(dist2_flat, dtype=torch.float32)
    if finite.any():
        weights[finite] = hydration_weight_torch(torch.sqrt(dist2_flat[finite]), float(cfg.pixel_size))
    weights = weights.reshape(nz, ny, nx).to(torch.float16).contiguous()
    recorder = get_timing_recorder(cfg)
    if recorder is not None:
        recorder.set_counter('hydration_atlas_voxels', voxel_count)
        recorder.set_counter('hydration_atlas_mib', weights.numel() * weights.element_size() / 1048576.0)
        recorder.set_meta('hydration_atlas_shape_zyx', [nz, ny, nx])
        recorder.set_meta('hydration_atlas_spacing_a', spacing)
        recorder.set_meta('hydration_atlas_cutoff_a', cutoff)
    return HydrationAtlas(weights=weights, origin_xyz=origin.contiguous(), shape_xyz=(nx, ny, nz), spacing_a=spacing, cutoff_a=cutoff)

def get_or_build_hydration_atlas(atoms: Sequence[Atom], cfg: SimConfig) -> Tuple[HydrationAtlas, bool]:
    device = torch_device_from_cfg(cfg)
    key = _hydration_atlas_cache_key(atoms, cfg, device)
    cached = _HYDRATION_ATLAS_CACHE.get(key)
    if cached is not None:
        _HYDRATION_ATLAS_CACHE.move_to_end(key)
        recorder = get_timing_recorder(cfg)
        if recorder is not None:
            recorder.add_counter('hydration_atlas_cache_hits', 1)
            recorder.set_counter('hydration_atlas_voxels', cached.weights.numel())
            recorder.set_counter('hydration_atlas_mib', cached.weights.numel() * cached.weights.element_size() / 1048576.0)
        return (cached, True)
    recorder = get_timing_recorder(cfg)
    if recorder is not None:
        recorder.add_counter('hydration_atlas_cache_misses', 1)
    with timing_section(cfg, 'hydration_atlas_build'):
        atlas = build_hydration_atlas(atoms, cfg, device)
    _HYDRATION_ATLAS_CACHE[key] = atlas
    _HYDRATION_ATLAS_CACHE.move_to_end(key)
    max_entries = max(1, int(cfg.water_soft_atlas_cache_entries))
    while len(_HYDRATION_ATLAS_CACHE) > max_entries:
        _HYDRATION_ATLAS_CACHE.popitem(last=False)
    return (atlas, False)

def orient_hydration_atlas(atlas: HydrationAtlas, protein_to_world: np.ndarray, translation_xyz: np.ndarray) -> OrientedHydrationAtlas:
    device = atlas.weights.device
    rotation = torch.as_tensor(protein_to_world, device=device, dtype=torch.float32)
    translation = torch.as_tensor(translation_xyz, device=device, dtype=torch.float32)
    nx, ny, nz = atlas.shape_xyz
    maximum = atlas.origin_xyz + torch.tensor([(nx - 1) * atlas.spacing_a, (ny - 1) * atlas.spacing_a, (nz - 1) * atlas.spacing_a], device=device, dtype=torch.float32)
    minimum = atlas.origin_xyz
    corner_bits = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0],
         [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    corners = minimum[None, :] + corner_bits * (maximum - minimum)[None, :]
    world_corners = corners @ rotation.T + translation[None, :]
    # A nearest-neighbour voxel owns a half-spacing cube.  After arbitrary
    # rotation its projection on a world axis can reach sqrt(3)/2 spacings.
    padding = 0.5 * math.sqrt(3.0) * atlas.spacing_a + 1.0e-6
    return OrientedHydrationAtlas(atlas=atlas, protein_to_world=rotation.contiguous(), world_to_protein_row=rotation.contiguous(), translation_xyz=translation.contiguous(), world_aabb_min=(world_corners.min(dim=0).values - padding).contiguous(), world_aabb_max=(world_corners.max(dim=0).values + padding).contiguous())

def hydration_atlas_weights_nearest(world_coords: torch.Tensor, oriented: OrientedHydrationAtlas, cfg: SimConfig) -> torch.Tensor:
    coords = world_coords.to(device=oriented.atlas.weights.device, dtype=torch.float32)
    weights = torch.ones((coords.shape[0],), device=coords.device, dtype=torch.float32)
    if coords.numel() == 0:
        return weights
    candidate = torch.all((coords >= oriented.world_aabb_min[None, :]) & (coords <= oriented.world_aabb_max[None, :]), dim=1)
    candidate_ids = torch.nonzero(candidate, as_tuple=False).flatten()
    if candidate_ids.numel() == 0:
        recorder = get_timing_recorder(cfg)
        if recorder is not None:
            recorder.add_counter('hydration_query_points', int(coords.shape[0]))
        return weights
    selected_world = coords.index_select(0, candidate_ids)
    protein_coords = (selected_world - oriented.translation_xyz[None, :]) @ oriented.world_to_protein_row
    grid = torch.floor((protein_coords - oriented.atlas.origin_xyz[None, :]) / oriented.atlas.spacing_a + 0.5).long()
    nx, ny, nz = oriented.atlas.shape_xyz
    valid = (grid[:, 0] >= 0) & (grid[:, 0] < nx) & (grid[:, 1] >= 0) & (grid[:, 1] < ny) & (grid[:, 2] >= 0) & (grid[:, 2] < nz)
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_ids.numel() > 0:
        grid_valid = grid.index_select(0, valid_ids)
        flat = grid_valid[:, 2] * (ny * nx) + grid_valid[:, 1] * nx + grid_valid[:, 0]
        lookup = oriented.atlas.weights.reshape(-1).index_select(0, flat.long()).to(torch.float32)
        weights[candidate_ids.index_select(0, valid_ids)] = lookup
    recorder = get_timing_recorder(cfg)
    if recorder is not None:
        recorder.add_counter('hydration_query_points', int(coords.shape[0]))
        recorder.add_counter('hydration_aabb_candidates', int(candidate_ids.numel()))
        recorder.add_counter('hydration_atlas_valid_lookups', int(valid_ids.numel()))
    return weights

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
    ice_thickness_a: Optional[float] = None
    ice_protein_position: str = 'center'
    ice_surface_clearance_a: float = 0.0
    inelastic_potential: bool = False
    min_bfactor: float = 15.0
    bfactor_scaling: float = 0.0
    bond_scaling: float = BOND_SCALING_DEFAULT
    center_by_mass: bool = True
    euler_rot_deg: float = 0.0
    euler_tilt_deg: float = 0.0
    euler_psi_deg: float = 0.0
    euler_inverse: bool = False
    use_hydrogen: bool = False
    explicit_water: bool = False
    water_density_scale: float = 1.0
    water_max_count: Optional[int] = None
    water_template_radius_pix: int = 4
    water_subpix_n: int = 5
    water_exclude_below_a: float = 2.5
    water_soft_weight: bool = False
    water_soft_atlas_spacing_a: float = 1.5
    water_soft_atlas_cutoff_a: float = 9.0
    water_soft_atlas_atom_chunk_size: int = 128
    water_soft_atlas_cache_entries: int = 2
    water_bfactor: float = 34.0
    mode: str = 'projection'
    poisson: bool = False
    dqe: bool = False
    seed: Optional[int] = None
    verbose: bool = False
    radiation_damage: bool = False
    radiation_damage_where: str = 'protein'
    pre_exposure_e_per_a2: float = 0.0
    exposure_filter_modify_signal: int = 0
    per_frame: bool = False
    number_of_frames: int = 1
    dose_per_frame_e_per_a2: Optional[float] = None
    save_frames: bool = False
    normalize_frame_sum: bool = True
    shake_waters: bool = False
    use_cache_atom: bool = False
    device: str = 'cuda'
    solvent_padding_pix: int = 64
    edge_taper_width_pix: int = 24
    sampled_mask_erode_pix: int = 7
    sampled_mask_lowpass: float = 0.05
    disable_cistem_edge_pipeline: bool = False
    frame_batch_size: int = 1
    water_template_chunk_size: int = 16
    water_slab_chunk_size: int = 0
    water_seed_z_chunk: int = 16
    water_seed_max_octants_per_chunk: int = 67108864
    water_filter_chunk_size: int = 250000
    water_filter_cell_size_a: Optional[float] = None
    atom_cache_subpix_n: int = 9
    atom_cache_radius_pix: int = 9
    atom_template_chunk_size: int = 16
    protein_tensor_cache_entries: int = 2
    objective_aperture_diameter_micron: float = 100.0
    objective_aperture_falloff_pix: float = 14.0
    disable_cistem_inelastic_filter: bool = False
    disable_cistem_defocus_offset: bool = False
    timing: bool = False
    timing_json: Optional[str] = None

def electron_wavelength_angstrom(kv: float) -> float:
    return 1226.39 / math.sqrt(kv * 1000.0 + 9.7845e-07 * (kv * 1000.0) ** 2) * 0.01

def electron_dose_voltage_scaling(kv: float) -> float:
    """cisTEM ElectronDose voltage scaling for critical dose."""
    kv = float(kv)
    if 299.0 < kv < 301.0:
        return 1.0
    if 199.0 < kv < 201.0:
        return 0.8
    if 99.0 < kv < 101.0:
        return 0.532
    raise ValueError(f'Unsupported voltage for cisTEM ElectronDose: {kv}. cisTEM supports 100, 200, or 300 kV in this model.')

def infer_element(line: str) -> str:
    elem = line[76:78].strip().upper() if len(line) >= 78 else ''
    if not elem:
        name = line[12:16].strip().upper()
        name = ''.join((ch for ch in name if ch.isalpha()))
        if len(name) >= 2 and name[:2] in ATOM_INDEX:
            elem = name[:2]
        elif name:
            elem = name[0]
    return elem

def read_pdb_atoms(path: str | Path, use_hydrogen: bool=False, allow_hetatm: bool=True) -> List[Atom]:
    atoms: List[Atom] = []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in {'ATOM', 'HETATM'}:
                continue
            if rec == 'HETATM' and (not allow_hetatm):
                continue
            elem = infer_element(line)
            if not elem or elem not in ATOM_INDEX:
                continue
            if elem == 'H' and (not use_hydrogen):
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
        raise ValueError(f'No supported atoms found in {path}')
    return atoms

def center_atoms(atoms: List[Atom]) -> None:
    coords = np.array([a.xyz for a in atoms], dtype=np.float64)
    weights = np.array([ATOMIC_NUMBER[ATOM_INDEX[a.element]] for a in atoms], dtype=np.float64)
    com = (coords * weights[:, None]).sum(axis=0) / weights.sum()
    for atom in atoms:
        atom.xyz = atom.xyz - com

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
    A[0, 0] = cg * cc - sg * sa
    A[0, 1] = cg * cs + sg * ca
    A[0, 2] = -cg * sb
    A[1, 0] = -sg * cc - cg * sa
    A[1, 1] = -sg * cs + cg * ca
    A[1, 2] = sg * sb
    A[2, 0] = sc
    A[2, 1] = ss
    A[2, 2] = cb
    return A

def has_manual_ice_thickness(cfg: SimConfig) -> bool:
    return getattr(cfg, 'ice_thickness_a', None) is not None

def physical_ice_thickness_a(cfg: SimConfig) -> float:
    """Return the physical z thickness represented by the slab stack."""
    if has_manual_ice_thickness(cfg):
        thickness = float(getattr(cfg, 'ice_thickness_a'))
    else:
        thickness = float(cfg.box) * float(cfg.pixel_size)
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError(f'Ice thickness must be finite and > 0 A, got {thickness!r}')
    return thickness

def ice_box_periods_a(cfg: SimConfig) -> np.ndarray:
    """Periodic lengths [x, y, z] used by explicit-water wrapping."""
    xy = float(cfg.box) * float(cfg.pixel_size)
    z = physical_ice_thickness_a(cfg)
    if xy <= 0.0:
        raise ValueError('box and pixel_size must define a positive x/y extent')
    return np.asarray([xy, xy, z], dtype=np.float32)

def water_grid_geometry(cfg: SimConfig) -> Tuple[int, int, float, float, float]:
    """Return n_xy, n_z, xy spacing, z spacing, and exact z thickness.

    The historical path remains n_z=n_xy and dz=pixel_size.  With
    --ice-thickness, z is represented by ceil(thickness/pixel_size) seeding
    cells whose exact spacing is thickness/n_z.  Slab propagation itself is
    continuous and does not depend on this seeding-grid count.
    """
    n_xy = int(cfg.box)
    ps = float(cfg.pixel_size)
    if n_xy <= 0 or ps <= 0.0:
        raise ValueError('box and pixel_size must be positive')
    thickness = physical_ice_thickness_a(cfg)
    if has_manual_ice_thickness(cfg):
        n_z = max(1, int(math.ceil(thickness / ps)))
        dz = thickness / float(n_z)
    else:
        n_z = n_xy
        dz = ps
    return (n_xy, n_z, ps, float(dz), float(thickness))

def continuous_slab_z_geometry(cfg: SimConfig) -> Tuple[np.ndarray, List[float]]:
    """Return centered z edges and strictly positive slab thicknesses.

    In manual-ice mode the requested physical thickness is divided directly
    into n_slices continuous intervals.  Thus even when the ice is thinner than
    the x/y box or thinner than n_slices*pixel_size, no zero-thickness slabs are
    created.
    """
    n_slices = max(1, int(cfg.n_slices))
    thickness = physical_ice_thickness_a(cfg)
    edges = np.linspace(-0.5 * thickness, 0.5 * thickness, n_slices + 1, dtype=np.float64)
    dz = np.diff(edges)
    if not np.all(np.isfinite(dz)) or np.any(dz <= 0.0):
        raise ValueError(f'Invalid slab geometry: ice_thickness={thickness}, n_slices={n_slices}')
    return (edges, [float(x) for x in dz])

def protein_inelastic_to_elastic_ratio(atom_index: int, cfg: SimConfig) -> float:
    """cisTEM-style element-dependent non-water inelastic amplitude ratio."""
    if not bool(getattr(cfg, 'inelastic_potential', False)):
        return 0.0
    z_eff = float(ATOMIC_NUMBER[int(atom_index)])
    if z_eff <= 0.0:
        return 0.0
    return math.sqrt(float(INELASTIC_SCALAR_WATER) / z_eff)

def place_protein_in_ice_z(atoms: List[Atom], cfg: SimConfig) -> float:
    """Place the rotated protein along z inside a manually specified ice layer.

    center
        Put the scattering-weighted molecular center at z=0.
    top
        Put the positive-z van-der-Waals envelope on the +z interface.
    bottom
        Put the negative-z van-der-Waals envelope on the -z interface.

    Only z is translated.  x/y coordinates and the selected Euler orientation
    are preserved.  The function returns the applied z shift in Angstrom.
    """
    if not has_manual_ice_thickness(cfg) or len(atoms) == 0:
        return 0.0
    mode = str(getattr(cfg, 'ice_protein_position', 'center')).strip().lower()
    if mode not in {'center', 'top', 'bottom'}:
        raise ValueError(f'Unknown ice_protein_position: {mode!r}')
    thickness = physical_ice_thickness_a(cfg)
    half = 0.5 * thickness
    clearance = float(getattr(cfg, 'ice_surface_clearance_a', 0.0) or 0.0)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError('ice_surface_clearance_a must be finite and >= 0')
    if 2.0 * clearance >= thickness:
        raise ValueError('ice surface clearance leaves no usable ice thickness')
    z = np.asarray([float(a.xyz[2]) for a in atoms], dtype=np.float64)
    radii = np.asarray([float(VDW_RADIUS_A.get(a.element.upper(), 1.7)) for a in atoms], dtype=np.float64)
    if mode == 'center':
        weights = np.asarray([float(ATOMIC_NUMBER[ATOM_INDEX[a.element]]) * max(float(a.occupancy), 0.0) for a in atoms], dtype=np.float64)
        if not np.any(weights > 0.0):
            weights = np.ones_like(z)
        shift_z = -float(np.sum(z * weights) / np.sum(weights))
    elif mode == 'top':
        shift_z = half - clearance - float(np.max(z + radii))
    else:
        shift_z = -half + clearance - float(np.min(z - radii))
    z_after = z + shift_z
    envelope_min = float(np.min(z_after - radii))
    envelope_max = float(np.max(z_after + radii))
    allowed_min = -half + clearance
    allowed_max = half - clearance
    tol = max(1e-05, 1e-06 * thickness)
    if envelope_min < allowed_min - tol or envelope_max > allowed_max + tol:
        molecular_thickness = float(np.max(z + radii) - np.min(z - radii))
        raise ValueError(f'Protein van-der-Waals envelope does not fit in the requested ice: protein thickness={molecular_thickness:.3f} A, ice thickness={thickness:.3f} A, placement={mode}, clearance={clearance:.3f} A')
    for atom in atoms:
        atom.xyz = np.asarray(atom.xyz, dtype=np.float64).copy()
        atom.xyz[2] += shift_z
    return float(shift_z)

def complete_bfactor(atom_b: float, bfactor_scaling: float, min_bfactor: float) -> float:
    return 0.25 * (atom_b * bfactor_scaling + min_bfactor)

def cistem_subpixel_offsets(subpix_n: int) -> np.ndarray:
    """Offsets used by simulate.cpp for projected water templates.

    With SUB_PIXEL_NEIGHBORHOOD=2, cisTEM creates five offsets along each
    axis: -2/6, -1/6, 0, 1/6, 2/6. Generalizing to odd subpix_n=2N+1 gives
    (idx-N)/(2N+2), i.e. denominator subpix_n+1.
    """
    subpix_n = int(subpix_n)
    if subpix_n <= 0 or subpix_n % 2 != 1:
        raise ValueError('cisTEM water subpixel grid must be a positive odd integer, e.g. 5')
    half = (subpix_n - 1) // 2
    return (np.arange(subpix_n, dtype=np.float32) - half) / float(subpix_n + 1)

def effective_water_dose_per_frame(cfg: SimConfig) -> float:
    """Dose used in cisTEM's water-template B-factor.

    Movie simulations use the explicit/per-frame dose.  A non-movie image is a
    single integrated frame, so its water B-factor uses the requested total
    image dose rather than the old hard-coded 1 e-/A^2 fallback.
    """
    dose_per_frame = getattr(cfg, 'dose_per_frame_e_per_a2', None)
    if dose_per_frame is not None:
        return float(dose_per_frame)
    n_frames = max(1, int(getattr(cfg, 'number_of_frames', 1)))
    if getattr(cfg, 'per_frame', False) or n_frames > 1:
        return float(getattr(cfg, 'dose_e_per_a2', 1.0)) / float(n_frames)
    return float(getattr(cfg, 'dose_e_per_a2', 1.0))

def cistem_water_template_bfactor(cfg: SimConfig) -> float:
    return 0.25 * float(cfg.water_bfactor) * max(effective_water_dose_per_frame(cfg), 0.0)

def atom_neighborhood_radius(pixel_size: float, bfactor: float) -> int:
    sigma = math.sqrt(max(SCATTERING_B.max() + bfactor, 1e-06)) / (2.0 * math.pi)
    radius_a = max(3.0 * pixel_size, 4.0 * sigma + 2.0 * pixel_size)
    return max(2, int(math.ceil(radius_a / pixel_size)))

def make_cistem_work_config(cfg: SimConfig) -> Tuple[SimConfig, int, int]:
    """Return the internal padded cisTEM-like config, final box, and padding.

    In simulate.cpp, solvent is generated and propagated in a larger solvent/FFT
    image and only later PadToWantedSize() crops to the requested output.  This
    direct-slab Python code keeps cfg.box as the requested output size, then uses
    a larger working box for explicit water so water clipping/tapering occurs in
    the guard band rather than in the final image.
    """
    final_box = int(cfg.box)
    pad = int(max(0, getattr(cfg, 'solvent_padding_pix', 0) or 0))
    if not getattr(cfg, 'explicit_water', False):
        pad = 0
    if pad <= 0:
        return (cfg, final_box, 0)
    work_cfg = replace(cfg, box=final_box + 2 * pad)
    return (work_cfg, final_box, pad)

def save_image_or_volume(path: str | Path, data: np.ndarray, pixel_size: float) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    array = np.asarray(data, dtype=np.float32)
    if suffix == '.npy':
        np.save(path, array)
        return
    if suffix not in {'.mrc', '.mrcs'}:
        raise ValueError(f'Output must end in .mrc, .mrcs, or .npy, got {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(array)
        mrc.voxel_size = float(pixel_size)
        mrc.update_header_stats()

def torch_device_from_cfg(cfg: SimConfig) -> torch.device:
    device = torch.device(str(cfg.device))
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError(f'CUDA device {cfg.device!r} was requested, but torch.cuda.is_available() is false. Use --device cpu explicitly for a CPU test.')
    return device

class TorchSimCache:
    """Per-image cache for fixed-shape Fourier grids, filters, and edge masks."""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.full_grids: Dict[Tuple[Tuple[int, int], float], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.rfft_grids: Dict[Tuple[Tuple[int, int], float], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.dose_ne_rfft: Dict[Tuple[Tuple[int, int], float, float], torch.Tensor] = {}
        self.dose_filters_rfft: Dict[Tuple[Tuple[int, int], float, float, float, float, bool], torch.Tensor] = {}
        self.ctf_rfft_filters: Dict[Tuple, torch.Tensor] = {}
        self.fresnel_full_filters: Dict[Tuple[Tuple[int, int], float, float, float], torch.Tensor] = {}
        self.dqe_rfft_filters: Dict[Tuple[Tuple[int, int], float, bool], torch.Tensor] = {}
        self.gaussian_lowpass_rfft_filters: Dict[Tuple[Tuple[int, int], float], torch.Tensor] = {}
        self.edge_distance_maps: Dict[Tuple[int, int], torch.Tensor] = {}
        self.taper_masks: Dict[Tuple[Tuple[int, int], int], torch.Tensor] = {}
        self.edge_band_masks: Dict[Tuple[Tuple[int, int], int], Tuple[torch.Tensor, int]] = {}
        self.radial_bin_full_maps: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, int]] = {}
        self.inelastic_lorentzian_full_filters: Dict[Tuple[Tuple[int, int], float], torch.Tensor] = {}
        self.objective_aperture_masks: Dict[Tuple, torch.Tensor] = {}

    def full_grid(self, shape: Tuple[int, int], pixel_size: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(pixel_size))
        cached = self.full_grids.get(key)
        if cached is not None:
            return cached
        ny, nx = shape
        fx = torch.fft.fftfreq(nx, d=float(pixel_size), device=self.device)
        fy = torch.fft.fftfreq(ny, d=float(pixel_size), device=self.device)
        ky, kx = torch.meshgrid(fy, fx, indexing='ij')
        out = (kx.to(torch.float32), ky.to(torch.float32), (kx * kx + ky * ky).to(torch.float32))
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
        ky, kx = torch.meshgrid(fy, fx, indexing='ij')
        out = (kx.to(torch.float32), ky.to(torch.float32), (kx * kx + ky * ky).to(torch.float32))
        self.rfft_grids[key] = out
        return out

    def critical_dose_rfft(self, shape: Tuple[int, int], pixel_size: float, kv: float) -> torch.Tensor:
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(pixel_size), float(kv))
        cached = self.dose_ne_rfft.get(key)
        if cached is not None:
            return cached
        _, _, k2 = self.rfft_grid(shape, pixel_size)
        value = critical_exposure_grant_grigorieff_torch(torch.sqrt(k2), kv=float(kv)).to(torch.float32)
        self.dose_ne_rfft[key] = value
        return value

    def dose_filter_rfft(self, shape: Tuple[int, int], pixel_size: float, kv: float, exposure_start_e_per_a2: float, exposure_end_e_per_a2: float, average_over_interval: bool=False) -> torch.Tensor:
        shape = (int(shape[0]), int(shape[1]))
        d0 = float(exposure_start_e_per_a2)
        d1 = float(exposure_end_e_per_a2)
        if d1 < d0:
            raise ValueError('exposure_end_e_per_a2 must be >= exposure_start_e_per_a2')
        key = (shape, float(pixel_size), float(kv), d0, d1, bool(average_over_interval))
        cached = self.dose_filters_rfft.get(key)
        if cached is not None:
            return cached
        if d1 == d0:
            filt = torch.ones((shape[0], shape[1] // 2 + 1), device=self.device, dtype=torch.float32)
        else:
            ne = self.critical_dose_rfft(shape, pixel_size, kv)
            if average_over_interval:
                denom = max(d1, 1e-12)
                filt = 2.0 * ne / denom * (torch.exp(-0.5 * d0 / ne) - torch.exp(-0.5 * d1 / ne))
            else:
                filt = torch.exp(-0.5 * d1 / ne)
            filt = filt.to(torch.float32)
        filt[0, 0] = 1.0
        self.dose_filters_rfft[key] = filt
        return filt

    def ctf_rfft(self, shape: Tuple[int, int], pixel_size: float, kv: float, cs_mm: float, defocus_u: float, defocus_v: Optional[float], defocus_angle_deg: float, amplitude_contrast: float, phase_shift_rad: float) -> torch.Tensor:
        dv = float(defocus_u if defocus_v is None else defocus_v)
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(pixel_size), float(kv), float(cs_mm), float(defocus_u), dv, float(defocus_angle_deg), float(amplitude_contrast), float(phase_shift_rad))
        cached = self.ctf_rfft_filters.get(key)
        if cached is not None:
            return cached
        lam = electron_wavelength_angstrom(float(kv))
        cs_a = float(cs_mm) * 10000000.0
        kx, ky, k2 = self.rfft_grid(shape, pixel_size)
        theta = torch.atan2(ky, kx) - math.radians(float(defocus_angle_deg))
        defocus = 0.5 * (float(defocus_u) + dv) + 0.5 * (float(defocus_u) - dv) * torch.cos(2.0 * theta)
        chi = math.pi * lam * defocus * k2 - 0.5 * math.pi * cs_a * lam ** 3 * k2 ** 2 + float(phase_shift_rad)
        amp = float(amplitude_contrast)
        filt = (-max(0.0, 1.0 - amp * amp) ** 0.5 * torch.sin(chi) - amp * torch.cos(chi)).to(torch.float32)
        self.ctf_rfft_filters[key] = filt
        return filt

    def fresnel_full(self, shape: Tuple[int, int], pixel_size: float, kv: float, dz_angstrom: float) -> torch.Tensor:
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(pixel_size), float(kv), float(dz_angstrom))
        cached = self.fresnel_full_filters.get(key)
        if cached is not None:
            return cached
        _, _, k2 = self.full_grid(shape, pixel_size)
        phase = math.pi * electron_wavelength_angstrom(float(kv)) * float(dz_angstrom) * k2
        prop = torch.exp(1j * phase).to(torch.complex64)
        self.fresnel_full_filters[key] = prop
        return prop

    def dqe_rfft(self, shape: Tuple[int, int], pixel_size: float, root: bool=True) -> torch.Tensor:
        shape = (int(shape[0]), int(shape[1]))
        key = (shape, float(pixel_size), bool(root))
        cached = self.dqe_rfft_filters.get(key)
        if cached is not None:
            return cached
        _, _, k2 = self.rfft_grid(shape, pixel_size)
        freq = torch.sqrt(k2)
        out = torch.zeros_like(freq, dtype=torch.float32)
        for a, b, c in zip(DQE_A, DQE_B, DQE_C):
            out.add_(float(a) * torch.exp(-(freq - float(b)) ** 2 / (2.0 * float(c) * float(c))))
        out.clamp_(min=0.0)
        out.div_(torch.clamp(out.max(), min=1e-12))
        if root:
            out.sqrt_()
        self.dqe_rfft_filters[key] = out
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
            filt = torch.exp(-0.5 * (torch.sqrt(k2) / cutoff) ** 2).to(torch.float32)
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
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.minimum(torch.minimum(yy, xx), torch.minimum(ny - 1 - yy, nx - 1 - xx))
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
            t = torch.clamp(self.edge_distance(shape) / float(width), 0.0, 1.0)
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
        count = int(band.sum().detach().cpu().item())
        out = (band, count)
        self.edge_band_masks[key] = out
        return out

def get_torch_sim_cache(cfg: SimConfig, device: torch.device) -> TorchSimCache:
    device = torch.device(device)
    cache = getattr(cfg, '_torch_sim_cache', None)
    if cache is None or cache.device != device:
        cache = TorchSimCache(device)
        setattr(cfg, '_torch_sim_cache', cache)
    return cache

def as_slab_stack_torch(slabs: Sequence[torch.Tensor]) -> torch.Tensor:
    if isinstance(slabs, torch.Tensor):
        if slabs.ndim == 2:
            return slabs[None, ...].to(torch.float32)
        return slabs.to(torch.float32)
    return torch.stack([s.to(torch.float32) for s in slabs], dim=0)

def slab_stack_to_list_torch(stack: torch.Tensor) -> List[torch.Tensor]:
    return [s.contiguous() for s in torch.unbind(stack.to(torch.float32), dim=0)]

def edge_mean_2d_stack_torch(stack: torch.Tensor, width: int=4) -> torch.Tensor:
    x = as_slab_stack_torch(stack)
    w = min(int(width), x.shape[-2] // 4, x.shape[-1] // 4)
    if w <= 0:
        return x.mean(dim=(-2, -1))
    edges = torch.cat([x[..., :w, :].reshape(x.shape[0], -1), x[..., -w:, :].reshape(x.shape[0], -1), x[..., :, :w].reshape(x.shape[0], -1), x[..., :, -w:].reshape(x.shape[0], -1)], dim=1)
    return edges.mean(dim=1).to(torch.float32)

def apply_fourier_filter_real_rfft_torch(img: torch.Tensor, filt_rfft: torch.Tensor) -> torch.Tensor:
    x = img.to(torch.float32)
    return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1)) * filt_rfft, s=tuple(x.shape[-2:]), dim=(-2, -1)).to(torch.float32)

def center_crop_torch(arr: torch.Tensor, final_box: int) -> torch.Tensor:
    """Center-crop a 2D tensor or stack on the last two axes."""
    final_box = int(final_box)
    if final_box <= 0:
        return arr
    if arr.shape[-2] == final_box and arr.shape[-1] == final_box:
        return arr.to(torch.float32)
    ny, nx = (int(arr.shape[-2]), int(arr.shape[-1]))
    if final_box > ny or final_box > nx:
        raise ValueError(f'Cannot crop {ny}x{nx} tensor to {final_box}x{final_box}')
    y0 = (ny - final_box) // 2
    x0 = (nx - final_box) // 2
    return arr[..., y0:y0 + final_box, x0:x0 + final_box].contiguous().to(torch.float32)

def critical_exposure_grant_grigorieff_torch(freq_a_inv: torch.Tensor, kv: float=300.0) -> torch.Tensor:
    f = freq_a_inv.to(torch.float32)
    ne = torch.empty_like(f)
    positive = f > 1e-06
    ne[positive] = 0.24499 * torch.pow(f[positive], -1.6649) + 2.8141
    ne[~positive] = 1000000000.0
    ne = ne * float(electron_dose_voltage_scaling(kv))
    return ne

def apply_exposure_filter_2d_torch(img: torch.Tensor, pixel_size: float, exposure_start_e_per_a2: float, exposure_end_e_per_a2: float, modify_signal: int=0, subtract_edge_mean: bool=True, kv: float=300.0, cfg: Optional[SimConfig]=None) -> torch.Tensor:
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
    filt = cache.dose_filter_rfft(tuple(work.shape[-2:]), pixel_size, kv, exposure_start_e_per_a2, exposure_end_e_per_a2, average_over_interval=False)
    if modify_signal == 1:
        filt = 1.0 - (1.0 - filt) / (1.0 + filt)
    elif modify_signal == 2:
        filt = torch.sqrt(torch.clamp(filt, min=0.0))
    out = apply_fourier_filter_real_rfft_torch(work, filt) + bg[:, None, None]
    return out[0].to(torch.float32) if squeeze else out.to(torch.float32)

def apply_radiation_damage_to_slabs_torch(phase_slabs: List[torch.Tensor], cfg: SimConfig, exposure_start_e_per_a2: Optional[float]=None, exposure_end_e_per_a2: Optional[float]=None) -> List[torch.Tensor]:
    if exposure_start_e_per_a2 is None:
        exposure_start_e_per_a2 = float(getattr(cfg, 'pre_exposure_e_per_a2', 0.0))
    if exposure_end_e_per_a2 is None:
        exposure_end_e_per_a2 = exposure_start_e_per_a2 + float(cfg.dose_e_per_a2)
    if len(phase_slabs) == 0:
        return phase_slabs
    stack = as_slab_stack_torch(phase_slabs)
    filtered = apply_exposure_filter_2d_torch(stack, cfg.pixel_size, exposure_start_e_per_a2, exposure_end_e_per_a2, modify_signal=int(getattr(cfg, 'exposure_filter_modify_signal', 0)), subtract_edge_mean=True, kv=float(getattr(cfg, 'kv', 300.0)), cfg=cfg)
    return slab_stack_to_list_torch(filtered)

def ctf_2d_rfft_torch(shape: Tuple[int, int], pixel_size: float, kv: float, cs_mm: float, defocus_u: float, defocus_v: Optional[float]=None, defocus_angle_deg: float=0.0, amplitude_contrast: float=0.07, phase_shift_rad: float=0.0, device: Optional[torch.device]=None, cfg: Optional[SimConfig]=None) -> torch.Tensor:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = get_torch_sim_cache(cfg if cfg is not None else SimConfig(device=str(device), pixel_size=float(pixel_size), kv=float(kv)), device)
    return cache.ctf_rfft(shape, pixel_size, kv, cs_mm, defocus_u, defocus_v, defocus_angle_deg, amplitude_contrast, phase_shift_rad)

def dqe_filter_rfft_torch(shape: Tuple[int, int], pixel_size: float, device: torch.device, root: bool=True, cfg: Optional[SimConfig]=None) -> torch.Tensor:
    cache = get_torch_sim_cache(cfg if cfg is not None else SimConfig(device=str(device), pixel_size=float(pixel_size)), device)
    return cache.dqe_rfft(shape, pixel_size, root=root)

def voxel_integrated_potential_torch(x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor, atom_index: int, bfactor: float, lead_term: float, device: torch.device) -> torch.Tensor:
    out = torch.zeros((z1.numel(), y1.numel(), x1.numel()), device=device, dtype=torch.float32)
    for i in range(5):
        b_total = float(SCATTERING_B[int(atom_index), i] + float(bfactor))
        if b_total <= 0.0:
            continue
        bplus = math.sqrt(4.0 * math.pi * math.pi / b_total)
        dx = torch.erf(float(bplus) * x2) - torch.erf(float(bplus) * x1)
        dy = torch.erf(float(bplus) * y2) - torch.erf(float(bplus) * y1)
        dz = torch.erf(float(bplus) * z2) - torch.erf(float(bplus) * z1)
        out.add_(float(SCATTERING_A[int(atom_index), i] * float(lead_term)) * dz[:, None, None] * dy[None, :, None] * dx[None, None, :])
    return out

def voxel_integrated_projected_potential_torch(x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor, atom_index: int, bfactor: float, lead_term: float, device: torch.device) -> torch.Tensor:
    out = torch.zeros((y1.numel(), x1.numel()), device=device, dtype=torch.float32)
    for i in range(5):
        b_total = float(SCATTERING_B[int(atom_index), i] + float(bfactor))
        if b_total <= 0.0:
            continue
        bplus = math.sqrt(4.0 * math.pi * math.pi / b_total)
        dx = torch.erf(float(bplus) * x2) - torch.erf(float(bplus) * x1)
        dy = torch.erf(float(bplus) * y2) - torch.erf(float(bplus) * y1)
        dz = torch.erf(float(bplus) * z2) - torch.erf(float(bplus) * z1)
        out.add_(float(SCATTERING_A[int(atom_index), i] * float(lead_term)) * dz.sum() * dy[:, None] * dx[None, :])
    return out

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


def _precompute_cached_atom_xy_kernels_torch(
    cfg: SimConfig,
    element_ids: Sequence[int],
    subpix_n: int,
    template_radius_pix: int,
    device: torch.device,
) -> torch.Tensor:
    """Return persistent [E,sx,sy,Gaussian,K,K] xy kernels.

    The construction is fully tensorized.  Its result is independent of Euler
    orientation and is cached in the persistent worker on the selected device.
    """
    element_ids = tuple(int(value) for value in element_ids)
    r = int(template_radius_pix)
    q = int(subpix_n)
    key = (
        str(torch.device(device)), element_ids, q, r,
        float(cfg.pixel_size), float(cfg.min_bfactor),
    )
    cached = _ATOM_XY_KERNEL_CACHE.get(key)
    recorder = get_timing_recorder(cfg)
    if cached is not None:
        _ATOM_XY_KERNEL_CACHE.move_to_end(key)
        if recorder is not None:
            recorder.add_counter('protein_atom_kernel_cache_hits', 1)
        return cached

    ps = float(cfg.pixel_size)
    bf = complete_bfactor(0.0, 0.0, float(cfg.min_bfactor))
    grid = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    center = (q - 1) / 2.0
    offsets = (torch.arange(q, device=device, dtype=torch.float32) - center) / float(q)
    edge1 = (grid[None, :] - offsets[:, None]) * ps - 0.5 * ps
    edge2 = edge1 + ps

    element_t = torch.as_tensor(element_ids, device=device, dtype=torch.long)
    b_table = torch.as_tensor(SCATTERING_B, device=device, dtype=torch.float32)
    b_total = b_table.index_select(0, element_t) + float(bf)  # [E,5]
    bplus = torch.sqrt((4.0 * math.pi * math.pi) / b_total)  # [E,5]
    delta = (
        torch.erf(bplus[:, None, :, None] * edge2[None, :, None, :])
        - torch.erf(bplus[:, None, :, None] * edge1[None, :, None, :])
    )  # [E,Q,5,K]
    dx = delta
    dy = delta
    kernels = (
        dy[:, None, :, :, :, None]
        * dx[:, :, None, :, None, :]
    ).contiguous()  # [E,sx,sy,5,Ky,Kx]

    _ATOM_XY_KERNEL_CACHE[key] = kernels
    _ATOM_XY_KERNEL_CACHE.move_to_end(key)
    while len(_ATOM_XY_KERNEL_CACHE) > 8:
        _ATOM_XY_KERNEL_CACHE.popitem(last=False)
    if recorder is not None:
        recorder.add_counter('protein_atom_kernel_cache_misses', 1)
    return kernels


def _precompute_cached_atom_z_prefix_torch(
    cfg: SimConfig,
    element_ids: Sequence[int],
    subpix_n: int,
    template_radius_pix: int,
    device: torch.device,
) -> torch.Tensor:
    """Return persistent [E,sz,Gaussian,K+1] z-integral prefix sums."""
    element_ids = tuple(int(value) for value in element_ids)
    r = int(template_radius_pix)
    q = int(subpix_n)
    key = (
        str(torch.device(device)), element_ids, q, r,
        float(cfg.pixel_size), float(cfg.min_bfactor),
    )
    cached = _ATOM_Z_PREFIX_CACHE.get(key)
    recorder = get_timing_recorder(cfg)
    if cached is not None:
        _ATOM_Z_PREFIX_CACHE.move_to_end(key)
        if recorder is not None:
            recorder.add_counter('protein_atom_z_prefix_cache_hits', 1)
        return cached

    ps = float(cfg.pixel_size)
    bf = complete_bfactor(0.0, 0.0, float(cfg.min_bfactor))
    grid = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    center = (q - 1) / 2.0
    offsets = (torch.arange(q, device=device, dtype=torch.float32) - center) / float(q)
    z1 = (grid[None, :] - offsets[:, None]) * ps - 0.5 * ps
    z2 = z1 + ps
    element_t = torch.as_tensor(element_ids, device=device, dtype=torch.long)
    b_table = torch.as_tensor(SCATTERING_B, device=device, dtype=torch.float32)
    b_total = b_table.index_select(0, element_t) + float(bf)
    bplus = torch.sqrt((4.0 * math.pi * math.pi) / b_total)
    dz = (
        torch.erf(bplus[:, None, :, None] * z2[None, :, None, :])
        - torch.erf(bplus[:, None, :, None] * z1[None, :, None, :])
    )  # [E,Q,5,K]
    zero = torch.zeros((*dz.shape[:-1], 1), device=device, dtype=torch.float32)
    prefix = torch.cat((zero, torch.cumsum(dz, dim=-1)), dim=-1).contiguous()
    _ATOM_Z_PREFIX_CACHE[key] = prefix
    _ATOM_Z_PREFIX_CACHE.move_to_end(key)
    while len(_ATOM_Z_PREFIX_CACHE) > 8:
        _ATOM_Z_PREFIX_CACHE.popitem(last=False)
    if recorder is not None:
        recorder.add_counter('protein_atom_z_prefix_cache_misses', 1)
    return prefix

def _select_cached_atom_elements(atoms: List[Atom], elements_for_cache: Optional[Sequence[str]]) -> Tuple[Tuple[str, ...], Tuple[int, ...], Dict[int, int]]:
    present = sorted({a.element.upper() for a in atoms if a.element.upper() in ATOM_INDEX})
    if elements_for_cache is None:
        elems = set(present)
        elems.update({'C', 'N', 'O', 'S', 'P'})
    else:
        elems = {str(e).upper() for e in elements_for_cache}
        elems.update(present)
    elems = tuple(sorted((e for e in elems if e in ATOM_INDEX)))
    element_ids = tuple((int(ATOM_INDEX[e]) for e in elems))
    ai_to_epos = {ai: i for i, ai in enumerate(element_ids)}
    return (elems, element_ids, ai_to_epos)


def _vectorized_atom_slab_contributions(
    protein_tensors: OrientedProteinTensors,
    cfg: SimConfig,
    element_ids: Sequence[int],
    subpix_n: int,
    template_radius_pix: int,
    n: int,
    n_slices: int,
    dz_list: Sequence[float],
    z_edges: Optional[np.ndarray],
    slab_z_starts: Optional[np.ndarray],
    slab_z_ends: Optional[np.ndarray],
    z_prefix: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build all atom/slab/Gaussian splat metadata with device tensor operations."""
    device = protein_tensors.xyz.device
    q = int(subpix_n)
    r = int(template_radius_pix)
    ps = float(cfg.pixel_size)
    half = float(n) / 2.0
    n_elem = len(element_ids)

    xyz = protein_tensors.xyz.to(torch.float32)
    atom_ai = protein_tensors.atom_index.to(torch.long)
    occupancy = protein_tensors.occupancy.to(torch.float32)
    if xyz.numel() == 0:
        empty_l = torch.empty((0,), device=device, dtype=torch.long)
        empty_f = torch.empty((0,), device=device, dtype=torch.float32)
        return empty_l, empty_l, empty_l, empty_f

    epos_lookup = torch.full((len(ATOM_INDEX),), -1, device=device, dtype=torch.long)
    element_t = torch.as_tensor(tuple(int(value) for value in element_ids), device=device, dtype=torch.long)
    epos_lookup[element_t] = torch.arange(n_elem, device=device, dtype=torch.long)
    epos = epos_lookup.index_select(0, atom_ai)

    x_pix = xyz[:, 0] / ps + half
    y_pix = xyz[:, 1] / ps + half
    ix = torch.round(x_pix).long()
    iy = torch.round(y_pix).long()
    sx = torch.clamp(torch.floor((x_pix - ix.to(torch.float32) + 0.5) * q).long(), 0, q - 1)
    sy = torch.clamp(torch.floor((y_pix - iy.to(torch.float32) + 0.5) * q).long(), 0, q - 1)
    valid = (
        (epos >= 0)
        & (ix >= -r) & (ix < n + r)
        & (iy >= -r) & (iy < n + r)
    )
    active = torch.nonzero(valid, as_tuple=False).flatten()
    if active.numel() == 0:
        empty_l = torch.empty((0,), device=device, dtype=torch.long)
        empty_f = torch.empty((0,), device=device, dtype=torch.float32)
        return empty_l, empty_l, empty_l, empty_f

    xyz = xyz.index_select(0, active)
    atom_ai = atom_ai.index_select(0, active)
    occupancy = occupancy.index_select(0, active)
    epos = epos.index_select(0, active)
    ix = ix.index_select(0, active)
    iy = iy.index_select(0, active)
    sx = sx.index_select(0, active)
    sy = sy.index_select(0, active)
    cx_pad = ix + r
    cy_pad = iy + r

    manual_z = z_edges is not None
    if manual_z:
        z_edges_t = torch.as_tensor(z_edges, device=device, dtype=torch.float32)
        support_a = (float(r) + 0.5) * ps
        s0 = torch.searchsorted(z_edges_t, xyz[:, 2] - support_a, right=True) - 1
        s1 = torch.searchsorted(z_edges_t, xyz[:, 2] + support_a, right=False)
        s0 = torch.clamp(s0, 0, n_slices)
        s1 = torch.clamp(s1, 0, n_slices)
        keep = s0 < s1
        keep_ids = torch.nonzero(keep, as_tuple=False).flatten()
        if keep_ids.numel() == 0:
            empty_l = torch.empty((0,), device=device, dtype=torch.long)
            empty_f = torch.empty((0,), device=device, dtype=torch.float32)
            return empty_l, empty_l, empty_l, empty_f
        xyz = xyz.index_select(0, keep_ids)
        atom_ai = atom_ai.index_select(0, keep_ids)
        occupancy = occupancy.index_select(0, keep_ids)
        epos = epos.index_select(0, keep_ids)
        sx = sx.index_select(0, keep_ids)
        sy = sy.index_select(0, keep_ids)
        cx_pad = cx_pad.index_select(0, keep_ids)
        cy_pad = cy_pad.index_select(0, keep_ids)
        s0 = s0.index_select(0, keep_ids)
        s1 = s1.index_select(0, keep_ids)
        min_dz = max(min(float(value) for value in dz_list), 1.0e-12)
        max_overlap = min(n_slices, int(math.ceil(2.0 * support_a / min_dz)) + 3)
        offsets = torch.arange(max_overlap, device=device, dtype=torch.long)
        slab_grid = s0[:, None] + offsets[None, :]
        pair_valid = slab_grid < s1[:, None]
        pair_atom = torch.arange(xyz.shape[0], device=device, dtype=torch.long)[:, None].expand_as(slab_grid)[pair_valid]
        pair_slab = slab_grid[pair_valid].long()
        rel_z0 = z_edges_t.index_select(0, pair_slab) - xyz.index_select(0, pair_atom)[:, 2]
        rel_z1 = z_edges_t.index_select(0, pair_slab + 1) - xyz.index_select(0, pair_atom)[:, 2]
        ai_pair = atom_ai.index_select(0, pair_atom)
        epos_pair = epos.index_select(0, pair_atom)
        sx_pair = sx.index_select(0, pair_atom)
        sy_pair = sy.index_select(0, pair_atom)
        cx_pair = cx_pad.index_select(0, pair_atom)
        cy_pair = cy_pad.index_select(0, pair_atom)
        occ_pair = occupancy.index_select(0, pair_atom)

        b_table = torch.as_tensor(SCATTERING_B, device=device, dtype=torch.float32)
        b_total = b_table.index_select(0, ai_pair) + complete_bfactor(0.0, 0.0, float(cfg.min_bfactor))
        bplus = torch.sqrt((4.0 * math.pi * math.pi) / b_total)
        zsum = torch.erf(bplus * rel_z1[:, None]) - torch.erf(bplus * rel_z0[:, None])
    else:
        if slab_z_starts is None or slab_z_ends is None or z_prefix is None:
            raise RuntimeError('Cubic-z cached atom metadata is missing geometry or z-prefix data')
        starts_t = torch.as_tensor(slab_z_starts, device=device, dtype=torch.long)
        ends_t = torch.as_tensor(slab_z_ends, device=device, dtype=torch.long)
        z_pix = xyz[:, 2] / ps + half
        iz = torch.round(z_pix).long()
        sz = torch.clamp(torch.floor((z_pix - iz.to(torch.float32) + 0.5) * q).long(), 0, q - 1)
        z0_raw = iz - r
        z1_raw = iz + r + 1
        z0 = torch.clamp(z0_raw, 0, n)
        z1 = torch.clamp(z1_raw, 0, n)
        s0 = torch.searchsorted(ends_t, z0, right=True)
        s1 = torch.searchsorted(starts_t, z1 - 1, right=True)
        keep = (z0 < z1) & (s0 < s1)
        keep_ids = torch.nonzero(keep, as_tuple=False).flatten()
        if keep_ids.numel() == 0:
            empty_l = torch.empty((0,), device=device, dtype=torch.long)
            empty_f = torch.empty((0,), device=device, dtype=torch.float32)
            return empty_l, empty_l, empty_l, empty_f
        atom_ai = atom_ai.index_select(0, keep_ids)
        occupancy = occupancy.index_select(0, keep_ids)
        epos = epos.index_select(0, keep_ids)
        sx = sx.index_select(0, keep_ids)
        sy = sy.index_select(0, keep_ids)
        cx_pad = cx_pad.index_select(0, keep_ids)
        cy_pad = cy_pad.index_select(0, keep_ids)
        sz = sz.index_select(0, keep_ids)
        z0_raw = z0_raw.index_select(0, keep_ids)
        z0 = z0.index_select(0, keep_ids)
        z1 = z1.index_select(0, keep_ids)
        s0 = s0.index_select(0, keep_ids)
        s1 = s1.index_select(0, keep_ids)
        min_slab_pixels = max(1, min(int(e - s) for s, e in zip(slab_z_starts, slab_z_ends)))
        max_overlap = min(n_slices, int(math.ceil((2 * r + 1) / min_slab_pixels)) + 3)
        offsets = torch.arange(max_overlap, device=device, dtype=torch.long)
        slab_grid = s0[:, None] + offsets[None, :]
        pair_valid = slab_grid < s1[:, None]
        pair_atom = torch.arange(atom_ai.shape[0], device=device, dtype=torch.long)[:, None].expand_as(slab_grid)[pair_valid]
        pair_slab = slab_grid[pair_valid].long()
        zz0 = torch.maximum(z0.index_select(0, pair_atom), starts_t.index_select(0, pair_slab))
        zz1 = torch.minimum(z1.index_select(0, pair_atom), ends_t.index_select(0, pair_slab))
        valid_pair = zz0 < zz1
        pair_atom = pair_atom[valid_pair]
        pair_slab = pair_slab[valid_pair]
        zz0 = zz0[valid_pair]
        zz1 = zz1[valid_pair]
        tz0 = zz0 - z0_raw.index_select(0, pair_atom)
        tz1 = tz0 + (zz1 - zz0)
        k = 2 * r + 1
        if bool(torch.any((tz0 < 0) | (tz1 > k) | (tz0 >= tz1))):
            raise RuntimeError('Cached atom z clipping index out of range')
        ai_pair = atom_ai.index_select(0, pair_atom)
        epos_pair = epos.index_select(0, pair_atom)
        sx_pair = sx.index_select(0, pair_atom)
        sy_pair = sy.index_select(0, pair_atom)
        cx_pair = cx_pad.index_select(0, pair_atom)
        cy_pair = cy_pad.index_select(0, pair_atom)
        occ_pair = occupancy.index_select(0, pair_atom)
        sz_pair = sz.index_select(0, pair_atom)
        gaussian = torch.arange(5, device=device, dtype=torch.long)[None, :]
        zsum = (
            z_prefix[epos_pair[:, None], sz_pair[:, None], gaussian, tz1[:, None]]
            - z_prefix[epos_pair[:, None], sz_pair[:, None], gaussian, tz0[:, None]]
        )

    a_table = torch.as_tensor(SCATTERING_A, device=device, dtype=torch.float32)
    lead_term = (
        float(cfg.bond_scaling)
        * electron_wavelength_angstrom(float(cfg.kv))
        / 8.0 / (ps * ps)
    )
    weights = occ_pair[:, None] * a_table.index_select(0, ai_pair) * float(lead_term) * zsum
    gaussian = torch.arange(5, device=device, dtype=torch.long)[None, :]
    base_group = (
        (((pair_slab * n_elem + epos_pair) * q + sx_pair) * q + sy_pair) * 5
    )
    groups = base_group[:, None] + gaussian
    cx = cx_pair[:, None].expand(-1, 5)
    cy = cy_pair[:, None].expand(-1, 5)
    finite_nonzero = torch.isfinite(weights) & (weights != 0.0)
    groups = groups[finite_nonzero].long().contiguous()
    cx = cx[finite_nonzero].long().contiguous()
    cy = cy[finite_nonzero].long().contiguous()
    weights = weights[finite_nonzero].to(torch.float32).contiguous()

    recorder = get_timing_recorder(cfg)
    if recorder is not None:
        recorder.set_counter('protein_slab_atom_pairs', int(pair_slab.numel()))
        recorder.set_counter('protein_slab_contributions', int(weights.numel()))
    return groups, cx, cy, weights

def make_phase_amp_slabs_direct_from_atoms_torch_cached_grouped(
    atoms: List[Atom],
    cfg: SimConfig,
    subpix_n: int = 9,
    template_radius_pix: int = 9,
    elements_for_cache: Optional[Sequence[str]] = None,
    protein_tensors: Optional[OrientedProteinTensors] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
    """Cached atom slabs using separable 1D x/y grouped convolutions.

    The cached xy Gaussian kernel is exactly separable:
        K(y, x) = dy(y) * dx(x)

    Therefore the old KxK grouped convolution is replaced by a 1xK horizontal
    grouped convolution followed by a Kx1 vertical grouped convolution.  Atom
    metadata construction, exact slab-z integration, resident protein tensors,
    inelastic scaling, timing names, and the public function signature are
    unchanged.
    """
    if abs(float(cfg.bfactor_scaling)) > 1.0e-12:
        raise ValueError('The grouped atom cache requires --bfactor-scaling 0.')
    if int(subpix_n) <= 0 or int(subpix_n) % 2 != 1:
        raise ValueError('atom cached subpix_n must be a positive odd integer')
    if int(template_radius_pix) <= 0:
        raise ValueError('atom cached template_radius_pix must be positive')

    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    n_slices = max(1, int(cfg.n_slices))
    q = int(subpix_n)
    r = int(template_radius_pix)
    hpad = n + 2 * r
    wpad = n + 2 * r
    device = torch_device_from_cfg(cfg)
    manual_z = has_manual_ice_thickness(cfg)

    if manual_z:
        z_edges, dz_list = continuous_slab_z_geometry(cfg)
        slab_z_starts = slab_z_ends = None
    else:
        if n_slices > n:
            raise ValueError(
                'Without --ice-thickness, n_slices cannot exceed the cubic z box size'
            )
        z_chunks = np.array_split(np.arange(n, dtype=np.int32), n_slices)
        slab_z_starts = np.array(
            [int(chunk[0]) for chunk in z_chunks], dtype=np.int32
        )
        slab_z_ends = np.array(
            [int(chunk[-1]) + 1 for chunk in z_chunks], dtype=np.int32
        )
        dz_list = [float(len(chunk) * ps) for chunk in z_chunks]
        z_edges = None

    _, element_ids, _ = _select_cached_atom_elements(atoms, elements_for_cache)
    phase_stack = torch.zeros(
        (n_slices, n, n), device=device, dtype=torch.float32
    )
    amp_stack = torch.zeros_like(phase_stack)

    if not element_ids:
        return (
            slab_stack_to_list_torch(phase_stack),
            slab_stack_to_list_torch(amp_stack),
            dz_list,
        )

    if protein_tensors is None:
        protein_tensors = protein_tensors_from_oriented_atoms(atoms, device)

    if cfg.verbose:
        z_desc = (
            f'ice={physical_ice_thickness_a(cfg):.3f} A continuous-z'
            if manual_z
            else f'zbox={n} px'
        )
        print(
            'Using tensorized separable grouped-conv atom slabs: '
            f'atoms={protein_tensors.xyz.shape[0]}, box={n}, slices={n_slices}, '
            f'{z_desc}, subpix={q}, radius={r}, device={device}'
        )

    with timing_section(cfg, 'protein_atom_kernel_precompute'):
        # Keep the cache inside this function so this is a true one-function
        # replacement.  In a persistent worker, function attributes persist
        # across orientations exactly like the module-level caches.
        separable_cache = getattr(
            make_phase_amp_slabs_direct_from_atoms_torch_cached_grouped,
            '_separable_xy_kernel_cache',
            None,
        )
        if separable_cache is None:
            separable_cache = OrderedDict()
            setattr(
                make_phase_amp_slabs_direct_from_atoms_torch_cached_grouped,
                '_separable_xy_kernel_cache',
                separable_cache,
            )

        kernel_key = (
            str(torch.device(device)),
            tuple(int(value) for value in element_ids),
            q,
            r,
            float(cfg.pixel_size),
            float(cfg.min_bfactor),
        )
        xy_1d = separable_cache.get(kernel_key)
        recorder = get_timing_recorder(cfg)

        if xy_1d is not None:
            separable_cache.move_to_end(kernel_key)
            if recorder is not None:
                recorder.add_counter('protein_atom_kernel_cache_hits', 1)
        else:
            bf = complete_bfactor(0.0, 0.0, float(cfg.min_bfactor))
            grid = torch.arange(
                -r, r + 1, device=device, dtype=torch.float32
            )
            center = (q - 1) / 2.0
            offsets = (
                torch.arange(q, device=device, dtype=torch.float32) - center
            ) / float(q)
            edge1 = (
                (grid[None, :] - offsets[:, None]) * ps - 0.5 * ps
            )
            edge2 = edge1 + ps

            element_t = torch.as_tensor(
                tuple(int(value) for value in element_ids),
                device=device,
                dtype=torch.long,
            )
            b_table = torch.as_tensor(
                SCATTERING_B, device=device, dtype=torch.float32
            )
            b_total = b_table.index_select(0, element_t) + float(bf)
            bplus = torch.sqrt((4.0 * math.pi * math.pi) / b_total)

            # [element, subpixel, Gaussian, coordinate]
            xy_1d = (
                torch.erf(
                    bplus[:, None, :, None] * edge2[None, :, None, :]
                )
                - torch.erf(
                    bplus[:, None, :, None] * edge1[None, :, None, :]
                )
            ).contiguous()

            separable_cache[kernel_key] = xy_1d
            separable_cache.move_to_end(kernel_key)
            while len(separable_cache) > 8:
                separable_cache.popitem(last=False)
            if recorder is not None:
                recorder.add_counter('protein_atom_kernel_cache_misses', 1)

        z_prefix = (
            None
            if manual_z
            else _precompute_cached_atom_z_prefix_torch(
                cfg, element_ids, q, r, device
            )
        )

    with timing_section(cfg, 'protein_atom_metadata'):
        groups, cx, cy, vals = _vectorized_atom_slab_contributions(
            protein_tensors=protein_tensors,
            cfg=cfg,
            element_ids=element_ids,
            subpix_n=q,
            template_radius_pix=r,
            n=n,
            n_slices=n_slices,
            dz_list=dz_list,
            z_edges=z_edges,
            slab_z_starts=slab_z_starts,
            slab_z_ends=slab_z_ends,
            z_prefix=z_prefix,
        )

    if groups.numel() == 0:
        return (
            slab_stack_to_list_torch(phase_stack),
            slab_stack_to_list_torch(amp_stack),
            dz_list,
        )

    with timing_section(cfg, 'protein_group_sort'):
        order = torch.argsort(groups, stable=True)
        groups_sorted = groups.index_select(0, order)
        unique_groups, counts = torch.unique_consecutive(
            groups_sorted, return_counts=True
        )
        groups_cpu = [
            int(value) for value in unique_groups.detach().cpu().tolist()
        ]
        counts_cpu = [
            int(value) for value in counts.detach().cpu().tolist()
        ]
        starts_cpu = [0]
        for count in counts_cpu:
            starts_cpu.append(starts_cpu[-1] + count)

        cx_sorted = cx.index_select(0, order).contiguous()
        cy_sorted = cy.index_select(0, order).contiguous()
        vals_sorted = vals.index_select(0, order).contiguous()

    group_chunk = max(1, int(cfg.atom_template_chunk_size))
    hwpad = hpad * wpad
    n_elem = len(element_ids)
    inelastic_ratios = torch.as_tensor(
        [protein_inelastic_to_elastic_ratio(ai, cfg) for ai in element_ids],
        device=device,
        dtype=torch.float32,
    )

    if cfg.verbose:
        print(
            f'Tensorized separable atom slabs: contributions={groups.numel()}, '
            f'groups={len(groups_cpu)}, chunk={group_chunk}, '
            f'padded={hpad}x{wpad}'
        )

    recorder = get_timing_recorder(cfg)
    if recorder is not None:
        recorder.set_counter('protein_slab_groups', len(groups_cpu))

    with timing_section(cfg, 'protein_grouped_convolution'):
        for group_start in range(0, len(groups_cpu), group_chunk):
            group_end = min(
                len(groups_cpu), group_start + group_chunk
            )
            local_groups = groups_cpu[group_start:group_end]
            data_start = starts_cpu[group_start]
            data_end = starts_cpu[group_end]
            group_count = group_end - group_start

            if group_count <= 0 or data_end <= data_start:
                continue

            local_counts = torch.as_tensor(
                counts_cpu[group_start:group_end],
                device=device,
                dtype=torch.long,
            )
            local_channels = torch.repeat_interleave(
                torch.arange(
                    group_count, device=device, dtype=torch.long
                ),
                local_counts,
            )

            impulse = torch.zeros(
                (1, group_count, hpad, wpad),
                device=device,
                dtype=torch.float32,
            )
            flat_index = (
                local_channels * hwpad
                + cy_sorted[data_start:data_end] * wpad
                + cx_sorted[data_start:data_end]
            )
            impulse.view(-1).scatter_add_(
                0, flat_index, vals_sorted[data_start:data_end]
            )

            local_group_t = torch.as_tensor(
                local_groups, device=device, dtype=torch.long
            )
            tmp = local_group_t
            gaussian = torch.remainder(tmp, 5)
            tmp = torch.div(tmp, 5, rounding_mode='floor')
            sy = torch.remainder(tmp, q)
            tmp = torch.div(tmp, q, rounding_mode='floor')
            sx = torch.remainder(tmp, q)
            tmp = torch.div(tmp, q, rounding_mode='floor')
            epos = torch.remainder(tmp, n_elem)
            slab = torch.div(tmp, n_elem, rounding_mode='floor')

            # Original KxK kernel = dy[:,None] * dx[None,:].  conv2d is
            # cross-correlation, so each 1D vector is flipped to preserve the
            # old template-splat/convolution convention.
            kernel_x = xy_1d[epos, sx, gaussian]
            kernel_y = xy_1d[epos, sy, gaussian]
            kernel_x_conv = torch.flip(
                kernel_x, dims=(-1,)
            )[:, None, None, :].contiguous()
            kernel_y_conv = torch.flip(
                kernel_y, dims=(-1,)
            )[:, None, :, None].contiguous()

            intermediate = F.conv2d(
                impulse,
                kernel_x_conv,
                padding=(0, r),
                groups=group_count,
            )
            conv_maps = F.conv2d(
                intermediate,
                kernel_y_conv,
                padding=(r, 0),
                groups=group_count,
            )[0, :, r:r + n, r:r + n].to(torch.float32)

            phase_stack.index_add_(0, slab, conv_maps)
            if cfg.inelastic_potential:
                ratios = inelastic_ratios.index_select(0, epos)
                amp_stack.index_add_(
                    0, slab, conv_maps * ratios[:, None, None]
                )

    return (
        slab_stack_to_list_torch(phase_stack.contiguous()),
        slab_stack_to_list_torch(amp_stack.contiguous()),
        dz_list,
    )


def make_phase_amp_slabs_direct_from_atoms_torch_cached_grouped_old(
    atoms: List[Atom],
    cfg: SimConfig,
    subpix_n: int = 9,
    template_radius_pix: int = 9,
    elements_for_cache: Optional[Sequence[str]] = None,
    protein_tensors: Optional[OrientedProteinTensors] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
    """Cached atom slabs with resident tensors and vectorized slab metadata."""
    if abs(float(cfg.bfactor_scaling)) > 1.0e-12:
        raise ValueError('The grouped atom cache requires --bfactor-scaling 0.')
    if int(subpix_n) <= 0 or int(subpix_n) % 2 != 1:
        raise ValueError('atom cached subpix_n must be a positive odd integer')
    if int(template_radius_pix) <= 0:
        raise ValueError('atom cached template_radius_pix must be positive')

    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    n_slices = max(1, int(cfg.n_slices))
    r = int(template_radius_pix)
    hpad = n + 2 * r
    wpad = n + 2 * r
    device = torch_device_from_cfg(cfg)
    manual_z = has_manual_ice_thickness(cfg)
    if manual_z:
        z_edges, dz_list = continuous_slab_z_geometry(cfg)
        slab_z_starts = slab_z_ends = None
    else:
        if n_slices > n:
            raise ValueError('Without --ice-thickness, n_slices cannot exceed the cubic z box size')
        z_chunks = np.array_split(np.arange(n, dtype=np.int32), n_slices)
        slab_z_starts = np.array([int(chunk[0]) for chunk in z_chunks], dtype=np.int32)
        slab_z_ends = np.array([int(chunk[-1]) + 1 for chunk in z_chunks], dtype=np.int32)
        dz_list = [float(len(chunk) * ps) for chunk in z_chunks]
        z_edges = None

    _, element_ids, _ = _select_cached_atom_elements(atoms, elements_for_cache)
    phase_stack = torch.zeros((n_slices, n, n), device=device, dtype=torch.float32)
    amp_stack = torch.zeros_like(phase_stack)
    if not element_ids:
        return slab_stack_to_list_torch(phase_stack), slab_stack_to_list_torch(amp_stack), dz_list
    if protein_tensors is None:
        protein_tensors = protein_tensors_from_oriented_atoms(atoms, device)

    if cfg.verbose:
        z_desc = f'ice={physical_ice_thickness_a(cfg):.3f} A continuous-z' if manual_z else f'zbox={n} px'
        print(
            f'Using tensorized torch grouped-conv atom slabs: atoms={protein_tensors.xyz.shape[0]}, '
            f'box={n}, slices={n_slices}, {z_desc}, subpix={subpix_n}, radius={r}, device={device}'
        )

    with timing_section(cfg, 'protein_atom_kernel_precompute'):
        xy_kernels = _precompute_cached_atom_xy_kernels_torch(
            cfg, element_ids, int(subpix_n), r, device
        )
        z_prefix = None if manual_z else _precompute_cached_atom_z_prefix_torch(
            cfg, element_ids, int(subpix_n), r, device
        )

    with timing_section(cfg, 'protein_atom_metadata'):
        groups, cx, cy, vals = _vectorized_atom_slab_contributions(
            protein_tensors=protein_tensors,
            cfg=cfg,
            element_ids=element_ids,
            subpix_n=int(subpix_n),
            template_radius_pix=r,
            n=n,
            n_slices=n_slices,
            dz_list=dz_list,
            z_edges=z_edges,
            slab_z_starts=slab_z_starts,
            slab_z_ends=slab_z_ends,
            z_prefix=z_prefix,
        )
    if groups.numel() == 0:
        return slab_stack_to_list_torch(phase_stack), slab_stack_to_list_torch(amp_stack), dz_list

    with timing_section(cfg, 'protein_group_sort'):
        order = torch.argsort(groups, stable=True)
        groups_sorted = groups.index_select(0, order)
        unique_groups, counts = torch.unique_consecutive(groups_sorted, return_counts=True)
        groups_cpu = [int(value) for value in unique_groups.detach().cpu().tolist()]
        counts_cpu = [int(value) for value in counts.detach().cpu().tolist()]
        starts_cpu = [0]
        for count in counts_cpu:
            starts_cpu.append(starts_cpu[-1] + count)
        cx_sorted = cx.index_select(0, order).contiguous()
        cy_sorted = cy.index_select(0, order).contiguous()
        vals_sorted = vals.index_select(0, order).contiguous()

    group_chunk = max(1, int(cfg.atom_template_chunk_size))
    hwpad = hpad * wpad
    n_elem = len(element_ids)
    inelastic_ratios = torch.as_tensor(
        [protein_inelastic_to_elastic_ratio(ai, cfg) for ai in element_ids],
        device=device,
        dtype=torch.float32,
    )
    if cfg.verbose:
        print(
            f'Tensorized atom slabs: contributions={groups.numel()}, '
            f'groups={len(groups_cpu)}, chunk={group_chunk}, padded={hpad}x{wpad}'
        )
    recorder = get_timing_recorder(cfg)
    if recorder is not None:
        recorder.set_counter('protein_slab_groups', len(groups_cpu))

    with timing_section(cfg, 'protein_grouped_convolution'):
        for group_start in range(0, len(groups_cpu), group_chunk):
            group_end = min(len(groups_cpu), group_start + group_chunk)
            local_groups = groups_cpu[group_start:group_end]
            data_start = starts_cpu[group_start]
            data_end = starts_cpu[group_end]
            group_count = group_end - group_start
            if group_count <= 0 or data_end <= data_start:
                continue
            local_counts = torch.as_tensor(
                counts_cpu[group_start:group_end], device=device, dtype=torch.long
            )
            local_channels = torch.repeat_interleave(
                torch.arange(group_count, device=device, dtype=torch.long), local_counts
            )
            impulse = torch.zeros(
                (1, group_count, hpad, wpad), device=device, dtype=torch.float32
            )
            flat_index = (
                local_channels * hwpad
                + cy_sorted[data_start:data_end] * wpad
                + cx_sorted[data_start:data_end]
            )
            impulse.view(-1).scatter_add_(0, flat_index, vals_sorted[data_start:data_end])

            local_group_t = torch.as_tensor(local_groups, device=device, dtype=torch.long)
            tmp = local_group_t
            gaussian = torch.remainder(tmp, 5)
            tmp = torch.div(tmp, 5, rounding_mode='floor')
            sy = torch.remainder(tmp, int(subpix_n))
            tmp = torch.div(tmp, int(subpix_n), rounding_mode='floor')
            sx = torch.remainder(tmp, int(subpix_n))
            tmp = torch.div(tmp, int(subpix_n), rounding_mode='floor')
            epos = torch.remainder(tmp, n_elem)
            slab = torch.div(tmp, n_elem, rounding_mode='floor')

            kernels = xy_kernels[epos, sx, sy, gaussian]
            kernels_conv = torch.flip(kernels, dims=(-2, -1))[:, None].contiguous()
            conv_maps = F.conv2d(
                impulse, kernels_conv, padding=r, groups=group_count
            )[0, :, r:r + n, r:r + n].to(torch.float32)
            phase_stack.index_add_(0, slab, conv_maps)
            ratios = inelastic_ratios.index_select(0, epos)
            if cfg.inelastic_potential:
                amp_stack.index_add_(0, slab, conv_maps * ratios[:, None, None])

    return (
        slab_stack_to_list_torch(phase_stack.contiguous()),
        slab_stack_to_list_torch(amp_stack.contiguous()),
        dz_list,
    )

def direct_slab_z_bounds(n: int, n_slices: int) -> Tuple[List[int], List[int], List[float]]:
    """Integer z-pixel split used only by the historical cubic-z path."""
    n = int(n)
    n_slices = max(1, int(n_slices))
    if n <= 0:
        raise ValueError('z box size must be positive')
    if n_slices > n:
        raise ValueError('n_slices cannot exceed the cubic z pixel count unless --ice-thickness is used; manual ice thickness uses continuous positive-thickness slabs.')
    q, r = divmod(n, n_slices)
    starts: List[int] = []
    ends: List[int] = []
    z0 = 0
    for s in range(n_slices):
        size = q + (1 if s < r else 0)
        starts.append(z0)
        z0 += size
        ends.append(z0)
    return (starts, ends, [])

def make_phase_amp_slabs_direct_from_atoms_torch(atoms: List[Atom], cfg: SimConfig):
    """Generate protein slabs directly from atoms without a 3D protein volume."""
    device = torch_device_from_cfg(cfg)
    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    n_slices = max(1, int(cfg.n_slices))
    half = n / 2.0
    manual_z = has_manual_ice_thickness(cfg)
    if manual_z:
        z_edges, dz_list = continuous_slab_z_geometry(cfg)
        slab_starts = slab_ends = None
    else:
        slab_starts, slab_ends, _ = direct_slab_z_bounds(n, n_slices)
        dz_list = [float((e - s) * ps) for s, e in zip(slab_starts, slab_ends)]
        z_edges = None
    phase_slabs = [torch.zeros((n, n), device=device, dtype=torch.float32) for _ in range(n_slices)]
    amp_slabs = [torch.zeros_like(phase_slabs[0]) for _ in range(n_slices)]
    lam = electron_wavelength_angstrom(cfg.kv)
    lead_term = cfg.bond_scaling * lam / 8.0 / (ps * ps)
    for i_atom, atom in enumerate(atoms):
        ai = ATOM_INDEX[atom.element]
        x = float(atom.xyz[0]) / ps + half
        y = float(atom.xyz[1]) / ps + half
        ix, iy = (int(round(x)), int(round(y)))
        bf = complete_bfactor(atom.bfactor, cfg.bfactor_scaling, cfg.min_bfactor)
        r = atom_neighborhood_radius(ps, bf)
        x0, x1i = (max(0, ix - r), min(n, ix + r + 1))
        y0, y1i = (max(0, iy - r), min(n, iy + r + 1))
        if x0 >= x1i or y0 >= y1i:
            continue
        xs = torch.arange(x0, x1i, device=device, dtype=torch.float32)
        ys = torch.arange(y0, y1i, device=device, dtype=torch.float32)
        x1v = (xs - half) * ps - float(atom.xyz[0]) - 0.5 * ps
        y1v = (ys - half) * ps - float(atom.xyz[1]) - 0.5 * ps
        ratio = protein_inelastic_to_elastic_ratio(ai, cfg)
        if manual_z:
            assert z_edges is not None
            support_a = (float(r) + 0.5) * ps
            s0 = max(0, int(np.searchsorted(z_edges, float(atom.xyz[2]) - support_a, side='right') - 1))
            s1 = min(n_slices, int(np.searchsorted(z_edges, float(atom.xyz[2]) + support_a, side='left')))
            slab_iter = range(s0, s1)
            for s in slab_iter:
                z1v = torch.tensor([float(z_edges[s] - atom.xyz[2])], device=device, dtype=torch.float32)
                z2v = torch.tensor([float(z_edges[s + 1] - atom.xyz[2])], device=device, dtype=torch.float32)
                pot2d = voxel_integrated_projected_potential_torch(x1v, x1v + ps, y1v, y1v + ps, z1v, z2v, ai, bf, lead_term, device)
                contribution = pot2d * float(atom.occupancy)
                phase_slabs[s][y0:y1i, x0:x1i] += contribution
                if ratio != 0.0:
                    amp_slabs[s][y0:y1i, x0:x1i] += contribution * float(ratio)
        else:
            assert slab_starts is not None and slab_ends is not None
            z = float(atom.xyz[2]) / ps + half
            iz = int(round(z))
            z0, z1i = (max(0, iz - r), min(n, iz + r + 1))
            if z0 >= z1i:
                continue
            for s, (sz0, sz1) in enumerate(zip(slab_starts, slab_ends)):
                zz0, zz1 = (max(z0, sz0), min(z1i, sz1))
                if zz0 >= zz1:
                    continue
                zs = torch.arange(zz0, zz1, device=device, dtype=torch.float32)
                z1v = (zs - half) * ps - float(atom.xyz[2]) - 0.5 * ps
                pot2d = voxel_integrated_projected_potential_torch(x1v, x1v + ps, y1v, y1v + ps, z1v, z1v + ps, ai, bf, lead_term, device)
                contribution = pot2d * float(atom.occupancy)
                phase_slabs[s][y0:y1i, x0:x1i] += contribution
                if ratio != 0.0:
                    amp_slabs[s][y0:y1i, x0:x1i] += contribution * float(ratio)
        if cfg.verbose and (i_atom + 1) % 10000 == 0:
            print(f'  direct slab protein atoms: {i_atom + 1}/{len(atoms)}')
    return (phase_slabs, amp_slabs, dz_list)


def prepare_phase_amp_slabs_direct_torch(
    atoms: List[Atom],
    cfg: SimConfig,
    hydration_atlas: Optional[OrientedHydrationAtlas] = None,
    protein_tensors: Optional[OrientedProteinTensors] = None,
):
    with timing_section(cfg, 'protein_slab_generation'):
        if cfg.use_cache_atom:
            phase_slabs, amp_slabs, dz_list = make_phase_amp_slabs_direct_from_atoms_torch_cached_grouped(
                atoms,
                cfg,
                subpix_n=int(cfg.atom_cache_subpix_n),
                template_radius_pix=int(cfg.atom_cache_radius_pix),
                protein_tensors=protein_tensors,
            )
        else:
            phase_slabs, amp_slabs, dz_list = make_phase_amp_slabs_direct_from_atoms_torch(atoms, cfg)
    if cfg.radiation_damage and cfg.radiation_damage_where in {'protein', 'all'}:
        with timing_section(cfg, 'radiation_damage'):
            phase_slabs = apply_radiation_damage_to_slabs_torch(phase_slabs, cfg)
    if cfg.explicit_water:
        water_cache = prepare_water_cache_torch(atoms, cfg, hydration_atlas)
        if water_cache is not None:
            with timing_section(cfg, 'water_splat_total'):
                phase_slabs, amp_slabs = fill_water_potential_torch(
                    phase_slabs, amp_slabs, water_cache.water_coords,
                    water_cache.hydration_atlas, dz_list, cfg, water_cache.templates,
                )
            with timing_section(cfg, 'edge_pipeline'):
                phase_slabs, amp_slabs = apply_cistem_edge_pipeline_torch(
                    phase_slabs, amp_slabs, cfg
                )
    return phase_slabs, amp_slabs, dz_list

def hydration_weight_torch(radius_a: torch.Tensor, pixel_size: float) -> torch.Tensor:
    v = torch.as_tensor(HYDRATION_RADIUS_VALS, device=radius_a.device, dtype=torch.float32)
    shifted = radius_a + float(PUSH_BACK_BY)
    erf_arg = shifted - (v[2] + float(HYDRATION_RADIUS_EXTRA_SHIFT) * pixel_size) / (math.sqrt(2.0) * v[5])
    return (0.5 + 0.5 * torch.erf(erf_arg) + v[0] * torch.exp(-(shifted - (v[3] + float(HYDRATION_RADIUS_EXTRA_SHIFT) * pixel_size)) ** 2 / (2.0 * v[6] ** 2)) + v[1] * torch.exp(-(shifted - (v[4] + float(HYDRATION_RADIUS_EXTRA_SHIFT) * pixel_size)) ** 2 / (2.0 * v[7] ** 2))).to(torch.float32)


def precompute_projected_water_templates_torch(cfg: SimConfig) -> torch.Tensor:
    """Precompute only the xy subpixel water templates: Q^2 instead of Q^3.

    Water is projected through z before insertion into a slab.  A z translation
    therefore does not change the ideal projected template.  To preserve the
    previous finite-radius approximation as closely as possible, the central
    z-subpixel template is used for every water while x/y retain Q bins each.
    """
    device = torch_device_from_cfg(cfg)
    subpix_n = int(cfg.water_subpix_n)
    if subpix_n <= 0 or subpix_n % 2 != 1:
        raise ValueError('water_subpix_n must be a positive odd integer')
    radius_pix = int(cfg.water_template_radius_pix)
    ai = ATOM_INDEX['O']
    ps = float(cfg.pixel_size)
    lead_term = cfg.bond_scaling * electron_wavelength_angstrom(cfg.kv) / 8.0 / (ps * ps)
    bf = cistem_water_template_bfactor(cfg)
    xs = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.float32)
    ys = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.float32)
    zs = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.float32)
    z1 = zs * ps - 0.5 * ps  # central z-subpixel bin
    offsets = cistem_subpixel_offsets(subpix_n)
    templates: List[torch.Tensor] = []
    for sy, dy in enumerate(offsets):
        for sx, dx in enumerate(offsets):
            x1 = (xs - float(dx)) * ps - 0.5 * ps
            y1 = (ys - float(dy)) * ps - 0.5 * ps
            template = voxel_integrated_projected_potential_torch(
                x1, x1 + ps, y1, y1 + ps, z1, z1 + ps,
                ai, bf, lead_term, device,
            )
            templates.append(template.to(torch.float32))
    result = torch.stack(templates, dim=0).contiguous()
    recorder = get_timing_recorder(cfg)
    if recorder is not None:
        recorder.set_counter('water_template_count', int(result.shape[0]))
    return result

@dataclass
class TorchWaterCache:
    templates: torch.Tensor
    water_coords: torch.Tensor
    hydration_atlas: Optional[OrientedHydrationAtlas]
    generator: Optional[torch.Generator] = None
    static_phase_contrib: Optional[torch.Tensor] = None
    static_amp_contrib: Optional[torch.Tensor] = None

def _make_torch_generator_for_device(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator

@dataclass
class TorchAtomCellIndex:
    atom_xyz_sorted: torch.Tensor
    starts: torch.Tensor
    counts: torch.Tensor
    dims: Tuple[int, int, int]
    half_box_a: Tuple[float, float, float]
    cell_size_a: float

def _atoms_xyz_tensor_torch(atoms: List[Atom], device: torch.device) -> torch.Tensor:
    if len(atoms) == 0:
        return torch.empty((0, 3), device=device, dtype=torch.float32)
    arr = np.asarray([a.xyz for a in atoms], dtype=np.float32)
    return torch.as_tensor(arr, device=device, dtype=torch.float32).contiguous()

def _water_octant_offsets_torch(device: torch.device) -> torch.Tensor:
    """Return cisTEM-style half-voxel octant offsets in x/y/z order."""
    return torch.tensor([[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], device=device, dtype=torch.float32)

def _build_atom_cell_index_torch(atoms: List[Atom], cfg: SimConfig, device: torch.device, exclude_radius_a: float) -> Optional[TorchAtomCellIndex]:
    """Build an anisotropic uniform-grid atom index for exact radius queries."""
    if len(atoms) == 0:
        return None
    n = int(cfg.box)
    ps = float(cfg.pixel_size)
    xy = float(n) * ps
    z_extent = physical_ice_thickness_a(cfg)
    extents = np.asarray([xy, xy, z_extent], dtype=np.float64)
    half_np = 0.5 * extents
    r = float(exclude_radius_a)
    atom_xyz = _atoms_xyz_tensor_torch(atoms, device)
    if atom_xyz.numel() == 0:
        return None
    half_t = torch.as_tensor(half_np, device=device, dtype=torch.float32)
    in_range = torch.all((atom_xyz >= -half_t - r) & (atom_xyz < half_t + r), dim=1)
    atom_xyz = atom_xyz[in_range].contiguous()
    if atom_xyz.numel() == 0:
        return None
    requested = getattr(cfg, 'water_filter_cell_size_a', None)
    if requested is None or float(requested) <= 0.0:
        cell_size = max(r, 4.0 * ps, float(np.max(extents)) / 256.0)
    else:
        cell_size = max(r, float(requested))
    max_cells = 256 ** 3
    while True:
        dims = np.maximum(1, np.ceil(extents / cell_size).astype(np.int64))
        n_cells = int(np.prod(dims, dtype=np.int64))
        if n_cells <= max_cells:
            break
        cell_size *= max(1.01, (n_cells / max_cells) ** (1.0 / 3.0))
    nx, ny, nz = [int(x) for x in dims]
    shifted = (atom_xyz + half_t) / float(cell_size)
    cx = torch.clamp(torch.floor(shifted[:, 0]).long(), 0, nx - 1)
    cy = torch.clamp(torch.floor(shifted[:, 1]).long(), 0, ny - 1)
    cz = torch.clamp(torch.floor(shifted[:, 2]).long(), 0, nz - 1)
    cell_id = (cz * (nx * ny) + cy * nx + cx).long().contiguous()
    order = torch.argsort(cell_id, stable=True)
    cell_sorted = cell_id[order]
    atom_sorted = atom_xyz[order].contiguous()
    counts = torch.bincount(cell_sorted, minlength=n_cells).to(torch.long)
    starts = torch.cumsum(counts, dim=0) - counts
    return TorchAtomCellIndex(atom_xyz_sorted=atom_sorted, starts=starts.contiguous(), counts=counts.contiguous(), dims=(nx, ny, nz), half_box_a=(float(half_np[0]), float(half_np[1]), float(half_np[2])), cell_size_a=float(cell_size))

def _filter_waters_by_atom_distance_torch_grid_with_index(water_coords: torch.Tensor, atom_index: Optional[TorchAtomCellIndex], exclude_radius_a: float, chunk_size: int=250000) -> torch.Tensor:
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
    half = torch.as_tensor(atom_index.half_box_a, device=device, dtype=torch.float32)
    cell_size = float(atom_index.cell_size_a)
    xy_cells = nx * ny
    neighbor_offsets = tuple(((dx, dy, dz) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)))
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
            block_offsets = torch.repeat_interleave(torch.cumsum(cell_counts, dim=0) - cell_counts, cell_counts, output_size=total_pairs)
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

def _coords_from_flat_octant_indices_torch(flat: torch.Tensor, n_xy: int, n_z: int, ps: float, dz: float, z_offset: int=0) -> torch.Tensor:
    device = flat.device
    flat = flat.to(torch.long)
    oct_id = torch.remainder(flat, 8).long()
    voxel = torch.div(flat, 8, rounding_mode='floor')
    ix = torch.remainder(voxel, n_xy).to(torch.float32)
    tmp = torch.div(voxel, n_xy, rounding_mode='floor')
    iy = torch.remainder(tmp, n_xy).to(torch.float32)
    iz = torch.div(tmp, n_xy, rounding_mode='floor').to(torch.float32) + float(z_offset)
    offs = _water_octant_offsets_torch(device).index_select(0, oct_id)
    out = torch.empty((flat.numel(), 3), device=device, dtype=torch.float32)
    out[:, 0] = (ix + offs[:, 0] - float(n_xy) / 2.0) * float(ps)
    out[:, 1] = (iy + offs[:, 1] - float(n_xy) / 2.0) * float(ps)
    out[:, 2] = (iz + offs[:, 2] - float(n_z) / 2.0) * float(dz)
    return out.contiguous()

def generate_explicit_water_coords_torch(cfg: SimConfig, atoms: Optional[List[Atom]]=None, exclude_from_atoms: bool=True, device: Optional[torch.device]=None) -> torch.Tensor:
    """Generate explicit waters on the torch device in an anisotropic ice box."""
    device = torch_device_from_cfg(cfg) if device is None else torch.device(device)
    n, nz, ps, dz, thickness = water_grid_geometry(cfg)
    gen = _make_torch_generator_for_device(device, cfg.seed)
    xy = float(n) * ps
    expected_n_water = WATER_DENSITY_PER_A3 * xy * xy * thickness * float(cfg.water_density_scale)
    total_octants = int(n) * int(n) * int(nz) * 8
    p_water_octant = float(min(1.0, max(0.0, expected_n_water / float(total_octants))))
    if cfg.verbose:
        print(f'Explicit-water grid: {n} x {n} x {nz}; spacing={ps:.4f}, {ps:.4f}, {dz:.4f} A')
        print(f'Physical ice thickness: {thickness:.4f} A')
        print(f'Expected waters: {expected_n_water:.3e}')
        print(f'Water probability per voxel octant: {p_water_octant:.6g}')
        print(f'Generating explicit waters on {device} with torch RNG + grid exclusion...')
    atom_index = None
    if exclude_from_atoms and atoms and (float(cfg.water_exclude_below_a) > 0.0):
        atom_index = _build_atom_cell_index_torch(atoms, cfg, device, float(cfg.water_exclude_below_a))
    filter_chunk = int(max(1, getattr(cfg, 'water_filter_chunk_size', 250000)))
    manual_periods = torch.as_tensor(ice_box_periods_a(cfg), device=device, dtype=torch.float32) if has_manual_ice_thickness(cfg) else None

    def maybe_filter(c: torch.Tensor) -> torch.Tensor:
        c = c.reshape(-1, 3).to(torch.float32).contiguous()
        if c.numel() == 0:
            return c
        if manual_periods is not None:
            half_periods = 0.5 * manual_periods
            c = (torch.remainder(c + half_periods, manual_periods) - half_periods).contiguous()
        if atom_index is None:
            return c
        return _filter_waters_by_atom_distance_torch_grid_with_index(c, atom_index, float(cfg.water_exclude_below_a), filter_chunk)
    if p_water_octant <= 0.0:
        water_coords = torch.empty((0, 3), device=device, dtype=torch.float32)
    elif cfg.water_max_count is not None and int(cfg.water_max_count) > 0:
        target = int(cfg.water_max_count)
        oversample = 3 if atom_index is not None else 1
        n_candidates = min(total_octants, max(target * oversample, target))
        flat = torch.randint(0, total_octants, (n_candidates,), device=device, generator=gen)
        water_coords = maybe_filter(_coords_from_flat_octant_indices_torch(flat, n, nz, ps, dz))
    else:
        z_chunk = int(max(1, getattr(cfg, 'water_seed_z_chunk', 16)))
        max_octants = int(max(8, getattr(cfg, 'water_seed_max_octants_per_chunk', 0) or 0))
        if max_octants > 0:
            z_chunk = min(z_chunk, max(1, max_octants // max(1, n * n * 8)))
        z_chunk = min(nz, max(1, z_chunk))
        offsets = _water_octant_offsets_torch(device)
        chunks: List[torch.Tensor] = []
        n_before = n_after = 0
        for z0 in range(0, nz, z_chunk):
            z1 = min(nz, z0 + z_chunk)
            occ = torch.rand((z1 - z0, n, n, 8), device=device, generator=gen) < p_water_octant
            idx = torch.nonzero(occ, as_tuple=False)
            del occ
            if idx.numel() == 0:
                continue
            n_before += int(idx.shape[0])
            iz = idx[:, 0].to(torch.float32) + float(z0)
            iy = idx[:, 1].to(torch.float32)
            ix = idx[:, 2].to(torch.float32)
            io = idx[:, 3].long()
            offs = offsets.index_select(0, io)
            coords = torch.empty((idx.shape[0], 3), device=device, dtype=torch.float32)
            coords[:, 0] = (ix + offs[:, 0] - float(n) / 2.0) * ps
            coords[:, 1] = (iy + offs[:, 1] - float(n) / 2.0) * ps
            coords[:, 2] = (iz + offs[:, 2] - float(nz) / 2.0) * dz
            del idx, iz, iy, ix, io, offs
            coords = maybe_filter(coords)
            if coords.numel() > 0:
                n_after += int(coords.shape[0])
                chunks.append(coords)
        water_coords = torch.cat(chunks, dim=0).contiguous() if chunks else torch.empty((0, 3), device=device, dtype=torch.float32)
        if cfg.verbose:
            print(f'Generated waters before protein exclusion: {n_before}')
            print(f'Remaining waters after protein exclusion: {n_after}')
    if has_manual_ice_thickness(cfg) and water_coords.numel() > 0:
        periods = torch.as_tensor(ice_box_periods_a(cfg), device=device, dtype=torch.float32)
        half_periods = 0.5 * periods
        water_coords = torch.remainder(water_coords + half_periods, periods) - half_periods
        water_coords = water_coords.contiguous()
    if cfg.water_max_count is not None and int(cfg.water_max_count) > 0 and (int(water_coords.shape[0]) > int(cfg.water_max_count)):
        target = int(cfg.water_max_count)
        perm = torch.randperm(int(water_coords.shape[0]), device=device, generator=gen)[:target]
        water_coords = water_coords.index_select(0, perm).contiguous()
        if cfg.verbose:
            print(f'Downsampled waters to water_max_count: {int(water_coords.shape[0])}')
    if cfg.verbose:
        print(f'Final waters: {int(water_coords.shape[0])}')
    return water_coords.to(device=device, dtype=torch.float32).contiguous()

def prepare_water_cache_torch(atoms: List[Atom], cfg: SimConfig, hydration_atlas: Optional[OrientedHydrationAtlas]=None) -> Optional[TorchWaterCache]:
    if not cfg.explicit_water:
        return None
    device = torch_device_from_cfg(cfg)
    with timing_section(cfg, 'water_coordinate_generation'):
        water_coords = generate_explicit_water_coords_torch(cfg, atoms=atoms, exclude_from_atoms=True, device=device).to(device=device, dtype=torch.float32).contiguous()
    with timing_section(cfg, 'water_template_precompute'):
        templates = precompute_projected_water_templates_torch(cfg).to(device=device, dtype=torch.float32).contiguous()
    base_seed = None if cfg.seed is None else int(cfg.seed) + 1000003
    generator = _make_torch_generator_for_device(device, base_seed)
    recorder = get_timing_recorder(cfg)
    if recorder is not None:
        recorder.set_counter('water_count', int(water_coords.shape[0]))
    if cfg.verbose:
        print(f'Keeping {int(water_coords.shape[0])} water coordinates resident on {device}.')
    return TorchWaterCache(templates=templates, water_coords=water_coords, hydration_atlas=hydration_atlas, generator=generator)

def shake_waters_3d_torch_inplace(water_cache: TorchWaterCache, cfg: SimConfig, dose_per_frame_e_per_a2: float) -> torch.Tensor:
    """cisTEM-like cumulative water shaking in Angstrom with anisotropic wrap."""
    coords = water_cache.water_coords
    if coords.numel() == 0:
        return coords
    sigma_a = 1.5 * float(dose_per_frame_e_per_a2) * float(cfg.pixel_size)
    if sigma_a != 0.0:
        kwargs = dict(device=coords.device, dtype=coords.dtype)
        noise = torch.randn(coords.shape, generator=water_cache.generator, **kwargs) if water_cache.generator is not None else torch.randn(coords.shape, **kwargs)
        coords.add_(noise, alpha=sigma_a)
    periods = torch.as_tensor(ice_box_periods_a(cfg), device=coords.device, dtype=coords.dtype)
    half = 0.5 * periods
    coords.add_(half)
    coords.remainder_(periods)
    coords.sub_(half)
    water_cache.static_phase_contrib = None
    water_cache.static_amp_contrib = None
    return coords


def _water_coordinate_bins_torch(
    water_coords: torch.Tensor,
    hydration_atlas: Optional[OrientedHydrationAtlas],
    dz_list: List[float],
    cfg: SimConfig,
    templates: torch.Tensor,
    shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return slab, xy-template, centers, weights, validity masks, and z edges."""
    device = templates.device
    coords = torch.as_tensor(water_coords, device=device, dtype=torch.float32).contiguous()
    n_templates = int(templates.shape[0])
    subpix_n = int(round(math.sqrt(n_templates)))
    if subpix_n * subpix_n != n_templates or subpix_n % 2 != 1:
        raise ValueError('Water templates must form an odd subpix_n^2 xy grid')
    ny, nx = int(shape[0]), int(shape[1])
    n_slices = len(dz_list)
    ps = float(cfg.pixel_size)
    total_z_a = float(sum(float(value) for value in dz_list))
    dz_arr = torch.as_tensor(dz_list, device=device, dtype=torch.float32)
    z_edges = torch.empty(n_slices + 1, device=device, dtype=torch.float32)
    z_edges[0] = -0.5 * total_z_a
    z_edges[1:] = z_edges[0] + torch.cumsum(dz_arr, dim=0)
    if coords.numel() == 0:
        empty_l = torch.empty((0,), device=device, dtype=torch.long)
        empty_f = torch.empty((0,), device=device, dtype=torch.float32)
        empty_b = torch.empty((0,), device=device, dtype=torch.bool)
        return empty_l, empty_l, empty_l, empty_l, empty_f, empty_b, empty_b, z_edges

    slab_all = torch.searchsorted(z_edges, coords[:, 2].contiguous(), right=True) - 1
    valid_z = (slab_all >= 0) & (slab_all < n_slices)
    coords_z = coords[valid_z].contiguous()
    slab = slab_all[valid_z].long().contiguous()
    x_pix = coords_z[:, 0] / ps + nx / 2.0
    y_pix = coords_z[:, 1] / ps + ny / 2.0
    ix_center = torch.floor(x_pix + 0.5).long()
    iy_center = torch.floor(y_pix + 0.5).long()
    valid_xy = (
        (ix_center >= 0) & (ix_center < nx)
        & (iy_center >= 0) & (iy_center < ny)
    )
    coords_valid = coords_z[valid_xy].contiguous()
    slab = slab[valid_xy].contiguous()
    x_pix = x_pix[valid_xy].contiguous()
    y_pix = y_pix[valid_xy].contiguous()
    ix = ix_center[valid_xy].contiguous()
    iy = iy_center[valid_xy].contiguous()
    if ix.numel() == 0:
        return slab, ix, iy, ix, torch.empty_like(x_pix), valid_z, valid_xy, z_edges

    fx = x_pix - ix.to(torch.float32)
    fy = y_pix - iy.to(torch.float32)
    half = (subpix_n - 1) // 2
    sx = torch.clamp(torch.trunc(fx * subpix_n).long() + half, 0, subpix_n - 1)
    sy = torch.clamp(torch.trunc(fy * subpix_n).long() + half, 0, subpix_n - 1)
    template_index = (sy * subpix_n + sx).long().contiguous()
    if hydration_atlas is None:
        weights = torch.ones_like(fx, dtype=torch.float32)
    else:
        with timing_section(cfg, 'water_soft_atlas_lookup'):
            weights = hydration_atlas_weights_nearest(coords_valid, hydration_atlas, cfg)
    return slab, template_index, ix, iy, weights.contiguous(), valid_z, valid_xy, z_edges


def fill_water_potential_torch(
    phase_slabs,
    amp_slabs,
    water_coords,
    hydration_atlas: Optional[OrientedHydrationAtlas],
    dz_list: List[float],
    cfg: SimConfig,
    templates: torch.Tensor,
    inelastic_scalar_water: float = INELASTIC_SCALAR_WATER,
):
    """Fused water splat: slab is batch and xy template is input channel.

    For each slab chunk an impulse tensor [B,Q^2,H,W] is filled with one
    scatter_add.  A single multi-channel conv2d then sums all Q^2 projected
    water templates into [B,1,H,W].  No water-group sort, CPU group table, or
    per-group map-add loop is required.
    """
    input_was_tensor = isinstance(phase_slabs, torch.Tensor)
    phase_stack = as_slab_stack_torch(phase_slabs)
    amp_stack = as_slab_stack_torch(amp_slabs)
    device = phase_stack.device
    coords = torch.as_tensor(water_coords, device=device, dtype=torch.float32)
    if coords.numel() == 0 or phase_stack.numel() == 0:
        if input_was_tensor:
            return phase_stack, amp_stack
        return slab_stack_to_list_torch(phase_stack), slab_stack_to_list_torch(amp_stack)

    templates = templates.to(device=device, dtype=torch.float32).contiguous()
    n_slices, ny, nx = (
        int(phase_stack.shape[0]), int(phase_stack.shape[-2]), int(phase_stack.shape[-1])
    )
    n_templates = int(templates.shape[0])
    subpix_n = int(round(math.sqrt(n_templates)))
    if subpix_n * subpix_n != n_templates:
        raise ValueError('The fused water convolution requires Q^2 xy templates')
    radius_pix = int(templates.shape[-1] // 2)

    with timing_section(cfg, 'water_binning_and_soft_weight'):
        slab, template_index, ix, iy, weights, valid_z, valid_xy, _ = _water_coordinate_bins_torch(
            coords, hydration_atlas, dz_list, cfg, templates, (ny, nx)
        )
    if slab.numel() == 0:
        if input_was_tensor:
            return phase_stack, amp_stack
        return slab_stack_to_list_torch(phase_stack), slab_stack_to_list_torch(amp_stack)

    templates_conv = torch.flip(templates, dims=(-2, -1))[None].contiguous()
    configured_slab_chunk = int(getattr(cfg, 'water_slab_chunk_size', 0))
    slab_chunk_size = n_slices if configured_slab_chunk <= 0 else min(n_slices, configured_slab_chunk)
    ratio = math.sqrt(float(inelastic_scalar_water) / 7.35)
    hw = ny * nx
    conv_batches = 0
    for slab_start in range(0, n_slices, slab_chunk_size):
        slab_end = min(n_slices, slab_start + slab_chunk_size)
        chunk_slices = slab_end - slab_start
        if slab_start == 0 and slab_end == n_slices:
            selected = None
            local_slab = slab
            local_template = template_index
            local_ix = ix
            local_iy = iy
            local_weights = weights
        else:
            selected = torch.nonzero(
                (slab >= slab_start) & (slab < slab_end), as_tuple=False
            ).flatten()
            if selected.numel() == 0:
                continue
            local_slab = slab.index_select(0, selected) - slab_start
            local_template = template_index.index_select(0, selected)
            local_ix = ix.index_select(0, selected)
            local_iy = iy.index_select(0, selected)
            local_weights = weights.index_select(0, selected)
        with timing_section(cfg, 'water_impulse_scatter'):
            impulse = torch.zeros(
                (chunk_slices, n_templates, ny, nx),
                device=device,
                dtype=torch.float32,
            )
            flat_index = (
                ((local_slab * n_templates + local_template) * ny + local_iy) * nx
                + local_ix
            )
            impulse.view(-1).scatter_add_(0, flat_index, local_weights)
        with timing_section(cfg, 'water_grouped_convolution'):
            conv_maps = F.conv2d(
                impulse,
                templates_conv,
                bias=None,
                stride=1,
                padding=radius_pix,
            )[:, 0].to(torch.float32)
        phase_stack[slab_start:slab_end].add_(conv_maps)
        amp_stack[slab_start:slab_end].add_(conv_maps, alpha=float(ratio))
        conv_batches += 1

    recorder = get_timing_recorder(cfg)
    if recorder is not None:
        recorder.add_counter('water_splat_count', int(slab.numel()))
        recorder.add_counter('water_splat_groups', n_slices * n_templates)
        recorder.add_counter('water_fused_conv_batches', conv_batches)
    if cfg.verbose:
        print(
            f'fill_water_potential_torch(fused): added={slab.numel()}, '
            f'xy_templates={n_templates}, slab_batches={conv_batches}, '
            f'outside_z={int((~valid_z).sum().detach().cpu().item()) if valid_z.numel() else 0}, '
            f'skipped_edge={int((~valid_xy).sum().detach().cpu().item()) if valid_xy.numel() else 0}'
        )
    if input_was_tensor:
        return phase_stack, amp_stack
    return slab_stack_to_list_torch(phase_stack), slab_stack_to_list_torch(amp_stack)

def apply_cistem_edge_pipeline_torch(phase_slabs, amp_slabs, cfg: SimConfig):
    """Batch-aware post-water taper_edges + sampled-potential mask."""
    input_was_list = not isinstance(phase_slabs, torch.Tensor)
    phase = as_slab_stack_torch(phase_slabs) if input_was_list else phase_slabs.to(torch.float32)
    amp = as_slab_stack_torch(amp_slabs) if input_was_list else amp_slabs.to(torch.float32)
    if not getattr(cfg, 'explicit_water', False) or getattr(cfg, 'disable_cistem_edge_pipeline', False):
        return (slab_stack_to_list_torch(phase), slab_stack_to_list_torch(amp)) if input_was_list else (phase, amp)
    if phase.numel() == 0:
        return (slab_stack_to_list_torch(phase), slab_stack_to_list_torch(amp)) if input_was_list else (phase, amp)
    squeeze_b = False
    if phase.ndim == 3:
        phase = phase.unsqueeze(0)
        amp = amp.unsqueeze(0)
        squeeze_b = True
    elif phase.ndim != 4:
        raise ValueError('phase_slabs must be [S,H,W] or [B,S,H,W]')
    device = phase.device
    cache = get_torch_sim_cache(cfg, device)
    shape = tuple(phase.shape[-2:])
    sampled = phase.sum(dim=1).to(torch.float32)
    width = int(max(0, getattr(cfg, 'edge_taper_width_pix', 24)))
    if width > 0:
        band, count = cache.edge_band_mask(shape, width)
        if count > 0:
            phase_bg = phase[:, :, band].mean(dim=2).to(torch.float32)
            amp_bg = amp[:, :, band].mean(dim=2).to(torch.float32)
        else:
            phase_bg = phase.mean(dim=(-2, -1)).to(torch.float32)
            amp_bg = amp.mean(dim=(-2, -1)).to(torch.float32)
        taper = cache.taper_mask(shape, width)
        phase = phase_bg[:, :, None, None] + (phase - phase_bg[:, :, None, None]) * taper[None, None, :, :]
        amp = amp_bg[:, :, None, None] + (amp - amp_bg[:, :, None, None]) * taper[None, None, :, :]
    phase_means = phase.mean(dim=(-2, -1)).to(torch.float32)
    amp_means = amp.mean(dim=(-2, -1)).to(torch.float32)
    erode_pix = int(max(0, getattr(cfg, 'sampled_mask_erode_pix', 7)))
    lowpass = float(getattr(cfg, 'sampled_mask_lowpass', 0.05))
    if erode_pix > 0 or lowpass > 0.0:
        mask = (sampled > 0.001).to(torch.float32)
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
        return (slab_stack_to_list_torch(phase), slab_stack_to_list_torch(amp))
    return (phase, amp)

def dose_filter_rfft_batch_torch(shape: Tuple[int, int], cfg: SimConfig, exposure_start: torch.Tensor, exposure_end: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return [B,H,W//2+1] cisTEM-style end-dose filters."""
    cache = get_torch_sim_cache(cfg, device)
    ne = cache.critical_dose_rfft(shape, float(cfg.pixel_size), float(getattr(cfg, 'kv', 300.0)))
    d0 = exposure_start.to(device=device, dtype=torch.float32).reshape(-1, 1, 1)
    d1 = exposure_end.to(device=device, dtype=torch.float32).reshape(-1, 1, 1)
    filt = torch.exp(-0.5 * d1 / ne[None, :, :]).to(torch.float32)
    _ = d0
    filt[:, 0, 0] = 1.0
    modify_signal = int(getattr(cfg, 'exposure_filter_modify_signal', 0))
    if modify_signal == 1:
        filt = 1.0 - (1.0 - filt) / (1.0 + filt)
    elif modify_signal == 2:
        filt = torch.sqrt(torch.clamp(filt, min=0.0))
    return filt.to(torch.float32)

def apply_radiation_damage_to_frame_batch_torch(base_phase_stack: torch.Tensor, cfg: SimConfig, exposure_starts: Sequence[float], exposure_ends: Sequence[float]) -> torch.Tensor:
    """Apply per-frame, per-slab radiation damage as one batched rFFT operation."""
    base = as_slab_stack_torch(base_phase_stack)
    device = base.device
    b = len(exposure_starts)
    if b <= 0:
        raise ValueError('empty exposure batch')
    if not getattr(cfg, 'radiation_damage', False):
        return base.unsqueeze(0).expand(b, -1, -1, -1).clone().to(torch.float32)
    stack = base.unsqueeze(0).expand(b, -1, -1, -1).clone().to(torch.float32)
    bg = edge_mean_2d_stack_torch(base).to(torch.float32)
    work = stack - bg[None, :, None, None]
    exp_start_t = torch.as_tensor(exposure_starts, device=device, dtype=torch.float32)
    exp_end_t = torch.as_tensor(exposure_ends, device=device, dtype=torch.float32)
    filt = dose_filter_rfft_batch_torch(tuple(work.shape[-2:]), cfg, exp_start_t, exp_end_t, device)
    out = torch.fft.irfft2(torch.fft.rfft2(work, dim=(-2, -1)) * filt[:, None, :, :], s=tuple(work.shape[-2:]), dim=(-2, -1)).to(torch.float32)
    out = out + bg[None, :, None, None]
    return out.contiguous()

def _prepare_frame_batch_slabs_from_base_torch(base_phase_stack: torch.Tensor, base_amp_stack: torch.Tensor, dz_list: List[float], cfg: SimConfig, water_cache: Optional[TorchWaterCache], frame_indices: Sequence[int], dose_per_frame_e_per_a2: float, pre_exposure_e_per_a2: float) -> Tuple[torch.Tensor, torch.Tensor]:
    frame_indices = [int(x) for x in frame_indices]
    starts = [float(pre_exposure_e_per_a2) + i * float(dose_per_frame_e_per_a2) for i in frame_indices]
    ends = [start + float(dose_per_frame_e_per_a2) for start in starts]
    batch = len(frame_indices)
    with timing_section(cfg, 'radiation_damage'):
        phase_batch = apply_radiation_damage_to_frame_batch_torch(base_phase_stack, cfg, starts, ends)
    amp_batch = as_slab_stack_torch(base_amp_stack).unsqueeze(0).expand(batch, -1, -1, -1).clone().to(torch.float32)
    if water_cache is not None:
        if cfg.shake_waters:
            for local_index, _frame_index in enumerate(frame_indices):
                with timing_section(cfg, 'water_shake'):
                    coords = shake_waters_3d_torch_inplace(water_cache, cfg, dose_per_frame_e_per_a2)
                with timing_section(cfg, 'water_splat_total'):
                    fill_water_potential_torch(phase_batch[local_index], amp_batch[local_index], coords, water_cache.hydration_atlas, dz_list, cfg, water_cache.templates)
        else:
            if water_cache.static_phase_contrib is None or water_cache.static_amp_contrib is None:
                water_phase = torch.zeros_like(base_phase_stack, dtype=torch.float32)
                water_amp = torch.zeros_like(base_amp_stack, dtype=torch.float32)
                with timing_section(cfg, 'water_splat_total'):
                    fill_water_potential_torch(water_phase, water_amp, water_cache.water_coords, water_cache.hydration_atlas, dz_list, cfg, water_cache.templates)
                water_cache.static_phase_contrib = water_phase.contiguous()
                water_cache.static_amp_contrib = water_amp.contiguous()
            phase_batch = phase_batch + water_cache.static_phase_contrib[None]
            amp_batch = amp_batch + water_cache.static_amp_contrib[None]
        with timing_section(cfg, 'edge_pipeline'):
            phase_batch, amp_batch = apply_cistem_edge_pipeline_torch(phase_batch, amp_batch, cfg)
    return (phase_batch.contiguous(), amp_batch.contiguous())

def finalize_detector_image_torch(img: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    with timing_section(cfg, 'detector_finalize'):
        out = img.to(torch.float32)
        spatial_dims = (-2, -1)
        if cfg.dqe:
            mean = out.mean(dim=spatial_dims, keepdim=True)
            contrast = out - mean
            filt = dqe_filter_rfft_torch(tuple(out.shape[-2:]), cfg.pixel_size, out.device, root=True, cfg=cfg)
            out = mean + apply_fourier_filter_real_rfft_torch(contrast, filt)
        if cfg.poisson:
            if cfg.seed is not None:
                torch.manual_seed(int(cfg.seed))
                if out.device.type == 'cuda':
                    torch.cuda.manual_seed_all(int(cfg.seed))
            electrons_per_pixel = float(cfg.dose_e_per_a2) * float(cfg.pixel_size) ** 2
            mean = out.mean(dim=spatial_dims, keepdim=True)
            norm_intensity = out / (mean + 1e-12)
            out = torch.poisson(torch.clamp(norm_intensity * electrons_per_pixel, min=0.0)).to(torch.float32)
        return out

def simulate_projection_from_slab_stack_batch_torch(phase_stack: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    squeeze = False
    phase = phase_stack.to(torch.float32)
    if phase.ndim == 3:
        phase = phase.unsqueeze(0)
        squeeze = True
    elif phase.ndim != 4:
        raise ValueError('phase_stack must be [S,H,W] or [B,S,H,W]')
    with timing_section(cfg, 'projection_and_ctf'):
        proj = phase.sum(dim=1)
        ctf = ctf_2d_rfft_torch(tuple(proj.shape[-2:]), cfg.pixel_size, cfg.kv, cfg.cs_mm, cfg.defocus_u, cfg.defocus_v, cfg.defocus_angle_deg, cfg.amplitude_contrast, cfg.phase_shift_rad, proj.device, cfg=cfg)
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
    return 1.0

def _cistem_radial_bin_index_full_torch(shape: Tuple[int, int], device: torch.device, cfg: SimConfig) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return full-FFT radial bin indices in reciprocal-pixel units.

    cisTEM builds a Curve from 0 to 0.5*sqrt(2) with roughly
    (nx/2+1)*sqrt(2)+1 samples before applying the whitening curve to the
    inelastic amplitude grating.  This helper caches the nearest-bin map and
    bin counts for a given image shape.
    """
    cache = get_torch_sim_cache(cfg, device)
    store = getattr(cache, 'radial_bin_full_maps', None)
    if store is None:
        store = {}
        setattr(cache, 'radial_bin_full_maps', store)
    shape = (int(shape[0]), int(shape[1]))
    key = shape
    cached = store.get(key)
    if cached is not None:
        return cached
    ny, nx = shape
    fx = torch.fft.fftfreq(nx, d=1.0, device=device)
    fy = torch.fft.fftfreq(ny, d=1.0, device=device)
    ky, kx = torch.meshgrid(fy, fx, indexing='ij')
    freq_pix = torch.sqrt(kx * kx + ky * ky).to(torch.float32)
    n_bins = int((max(nx, ny) / 2.0 + 1.0) * math.sqrt(2.0) + 1.0)
    n_bins = max(8, n_bins)
    step = 0.5 * math.sqrt(2.0) / float(max(1, n_bins - 1))
    bin_idx = torch.clamp(torch.round(freq_pix / step).long(), 0, n_bins - 1).contiguous()
    counts = torch.bincount(bin_idx.reshape(-1), minlength=n_bins).to(torch.float32)
    counts = torch.clamp(counts, min=1.0)
    out = (bin_idx, counts, n_bins)
    store[key] = out
    return out

def _cistem_inelastic_lorentzian_filter_full_torch(shape: Tuple[int, int], pixel_size: float, device: torch.device, cfg: SimConfig) -> torch.Tensor:
    """cisTEM's empirical Lorentzian-like plasmon conversion filter.

    This intentionally preserves the expression as written in
    wave_function_propagator.cpp, including the `q1 * frequency_squared * +q2 *
    frequency` term, which C++ parses as q1*q2*f^3.
    """
    cache = get_torch_sim_cache(cfg, device)
    store = getattr(cache, 'inelastic_lorentzian_full_filters', None)
    if store is None:
        store = {}
        setattr(cache, 'inelastic_lorentzian_full_filters', store)
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
    filt = (numerator / torch.clamp(denominator, min=1e-20)).to(torch.float32)
    store[key] = filt
    return filt

def _cistem_whiten_inelastic_fft_full_torch(fft_img: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    """Apply cisTEM-like radial whitening to full FFTs of amplitude gratings.

    Input shape is [M,H,W] complex.  The output is the same shape.  The exact
    cisTEM Curve interpolation is approximated with nearest radial bins; this
    keeps the important behavior: suppress the very strong smooth/DC inelastic
    background before the Lorentzian conversion filter.
    """
    if fft_img.ndim != 3:
        raise ValueError('fft_img must be [M,H,W]')
    m, ny, nx = fft_img.shape
    device = fft_img.device
    bin_idx, counts, n_bins = _cistem_radial_bin_index_full_torch((ny, nx), device, cfg)
    flat_bins = bin_idx.reshape(-1)
    power = (fft_img.real * fft_img.real + fft_img.imag * fft_img.imag).reshape(m, -1).to(torch.float32)
    sums = torch.zeros((m, n_bins), device=device, dtype=torch.float32)
    sums.scatter_add_(1, flat_bins[None, :].expand(m, -1), power)
    avg = sums / counts[None, :]
    radial_weight = torch.rsqrt(torch.clamp(avg, min=1e-30))
    radial_weight = radial_weight / torch.clamp(radial_weight.max(dim=1, keepdim=True).values, min=1e-30)
    weight = radial_weight.gather(1, flat_bins[None, :].expand(m, -1)).reshape(m, ny, nx).to(torch.float32)
    return fft_img * weight

def cistem_filter_inelastic_amplitude_batch_torch(amp: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    """Match WaveFunctionPropagator's amplitude_grating preprocessing.

    cisTEM copies the inelastic potential into amplitude_grating, and for the
    real image path it scales it by voltage, whitens its Fourier amplitude, and
    applies an empirical Lorentzian plasmon filter before forming
    exp(-amplitude) * cos/sin(phase).  This function applies the same operation
    to a batch of per-slab amplitude images.
    """
    if bool(getattr(cfg, 'disable_cistem_inelastic_filter', False)):
        return amp.to(torch.float32)
    x = amp.to(torch.float32)
    squeeze = False
    original_shape = x.shape
    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeeze = True
    elif x.ndim != 3:
        raise ValueError('amp must be [H,W] or [M,H,W]')
    m, ny, nx = x.shape
    mean = x.mean(dim=(-2, -1), keepdim=True)
    active = (mean > 0.001).to(torch.float32)
    scaled = x * float(_cistem_inelastic_voltage_scale(float(getattr(cfg, 'kv', 300.0))))
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

def cistem_defocus_offset_batch_torch(phase_stack: torch.Tensor, propagator_distances: Sequence[float], cfg: SimConfig) -> torch.Tensor:
    """Compute simulate.cpp's scattering-center defocus offset for [B,S,H,W]."""
    phase = phase_stack.to(torch.float32)
    if phase.ndim == 3:
        phase = phase.unsqueeze(0)
    if bool(getattr(cfg, 'disable_cistem_defocus_offset', False)):
        return torch.zeros((phase.shape[0],), device=phase.device, dtype=torch.float32)
    b, s, _, _ = phase.shape
    prop = torch.as_tensor(list(propagator_distances), device=phase.device, dtype=torch.float32)
    if prop.numel() != s:
        raise ValueError('propagator_distances length must match number of slabs')
    cumulative_from_slab = torch.flip(torch.cumsum(torch.flip(prop, dims=[0]), dim=0), dims=[0])
    mass = phase.sum(dim=(-2, -1)).to(torch.float32)
    total_mass = mass.sum(dim=1)
    weighted = (mass * cumulative_from_slab[None, :]).sum(dim=1)
    center = weighted / torch.clamp(total_mass, min=1e-20)
    center = torch.where(torch.abs(total_mass) > 1e-10, center, torch.zeros_like(center))
    offset = center - prop[0] / 2.0
    return offset.to(torch.float32)

def cistem_complex_transfer_full_batch_torch(shape: Tuple[int, int], cfg: SimConfig, device: torch.device, defocus_offset: torch.Tensor) -> torch.Tensor:
    """Complex final CTF transfer used by WaveFunctionPropagator.

    ctf[0] is initialized with amplitude contrast 1 and ctf[1] with amplitude
    contrast 0; the real/imag recombination is equivalent to multiplying by
    ctf_amp1 + i*ctf_amp0.
    """
    cache = get_torch_sim_cache(cfg, device)
    ny, nx = (int(shape[0]), int(shape[1]))
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
    cs_a = float(cfg.cs_mm) * 10000000.0
    chi = math.pi * lam * defocus * k2[None, :, :] - 0.5 * math.pi * cs_a * lam ** 3 * k2[None, :, :] ** 2 + float(cfg.phase_shift_rad)
    h_real = -torch.cos(chi)
    h_imag = -torch.sin(chi)
    return torch.complex(h_real.to(torch.float32), h_imag.to(torch.float32)).to(torch.complex64)

def cistem_objective_aperture_mask_full_torch(shape: Tuple[int, int], cfg: SimConfig, device: torch.device) -> torch.Tensor:
    """Approximate WaveFunctionPropagator's objective aperture cosine mask."""
    diameter = float(getattr(cfg, 'objective_aperture_diameter_micron', 100.0))
    if diameter <= 0.0:
        return torch.ones(shape, device=device, dtype=torch.float32)
    cache = get_torch_sim_cache(cfg, device)
    store = getattr(cache, 'objective_aperture_masks', None)
    if store is None:
        store = {}
        setattr(cache, 'objective_aperture_masks', store)
    key = (int(shape[0]), int(shape[1]), float(cfg.pixel_size), float(cfg.kv), diameter, float(getattr(cfg, 'objective_aperture_falloff_pix', 14.0)))
    cached = store.get(key)
    if cached is not None:
        return cached
    wavelength = electron_wavelength_angstrom(float(cfg.kv))
    objective_lens_focal_length_mm = 3.5
    resolution_a = wavelength * objective_lens_focal_length_mm * 10000000.0 / (diameter / 2.0 * 10000.0)
    cutoff_recip_pix = float(cfg.pixel_size) / max(resolution_a, 1e-12)
    _, _, k2_pix = cache.full_grid(shape, pixel_size=1.0)
    freq_pix = torch.sqrt(k2_pix).to(torch.float32)
    ny, nx = (int(shape[0]), int(shape[1]))
    max_freq = 0.5 * math.sqrt(2.0)
    if cutoff_recip_pix >= max_freq:
        mask = torch.ones(shape, device=device, dtype=torch.float32)
    else:
        falloff_pix = max(0.0, float(getattr(cfg, 'objective_aperture_falloff_pix', 14.0)))
        falloff = falloff_pix / float(max(nx, ny))
        if falloff <= 0.0:
            mask = (freq_pix <= cutoff_recip_pix).to(torch.float32)
        else:
            inner = max(0.0, cutoff_recip_pix - falloff)
            t = torch.clamp((freq_pix - inner) / max(cutoff_recip_pix - inner, 1e-12), 0.0, 1.0)
            mask = torch.where(freq_pix <= inner, torch.ones_like(freq_pix), torch.where(freq_pix >= cutoff_recip_pix, torch.zeros_like(freq_pix), 0.5 + 0.5 * torch.cos(math.pi * t))).to(torch.float32)
    store[key] = mask
    return mask

def propagate_slab_stack_batch_cistem_like_torch(phase_stack: torch.Tensor, amp_stack: torch.Tensor, dz_list: List[float], cfg: SimConfig) -> torch.Tensor:
    squeeze = False
    phase = phase_stack.to(torch.float32)
    amp = amp_stack.to(torch.float32)
    if phase.ndim == 3:
        phase = phase.unsqueeze(0)
        amp = amp.unsqueeze(0)
        squeeze = True
    elif phase.ndim != 4:
        raise ValueError('phase_stack must be [S,H,W] or [B,S,H,W]')
    device = phase.device
    batch, n_slices, ny, nx = phase.shape
    shape = (int(ny), int(nx))
    cache = get_torch_sim_cache(cfg, device)
    with timing_section(cfg, 'multislice_setup'):
        prop_distances = cistem_propagator_distances_from_dz(dz_list)
        defocus_offsets = cistem_defocus_offset_batch_torch(phase, prop_distances, cfg)
    with timing_section(cfg, 'inelastic_filter'):
        amp_for_grating = cistem_filter_inelastic_amplitude_batch_torch(amp.reshape(batch * n_slices, ny, nx), cfg).reshape(batch, n_slices, ny, nx)
    wave = torch.ones((batch, ny, nx), dtype=torch.complex64, device=device)
    with timing_section(cfg, 'multislice_propagation'):
        for slab_index, prop_distance in enumerate(prop_distances):
            transmission = torch.exp(torch.complex(-amp_for_grating[:, slab_index], phase[:, slab_index])).to(torch.complex64)
            wave.mul_(transmission)
            prop = -cache.fresnel_full(shape, cfg.pixel_size, cfg.kv, float(prop_distance))
            wave = torch.fft.ifft2(torch.fft.fft2(wave, dim=(-2, -1)) * prop, dim=(-2, -1)).to(torch.complex64)
    with timing_section(cfg, 'objective_lens_and_aperture'):
        lens = cistem_complex_transfer_full_batch_torch(shape, cfg, device, defocus_offsets)
        image_wave = torch.fft.ifft2(torch.fft.fft2(wave, dim=(-2, -1)) * lens, dim=(-2, -1)).to(torch.complex64)
        aperture = cistem_objective_aperture_mask_full_torch(shape, cfg, device)
        if float(aperture.min().detach().cpu().item()) < 1.0:
            image_wave = torch.fft.ifft2(torch.fft.fft2(image_wave, dim=(-2, -1)) * aperture, dim=(-2, -1)).to(torch.complex64)
        img = image_wave.real.square() + image_wave.imag.square()
    img = finalize_detector_image_torch(img, cfg)
    return img[0].contiguous() if squeeze else img.contiguous()

def propagate_slabs_cistem_like_torch(phase_slabs, amp_slabs, dz_list, cfg: SimConfig) -> torch.Tensor:
    return propagate_slab_stack_batch_cistem_like_torch(as_slab_stack_torch(phase_slabs), as_slab_stack_torch(amp_slabs), dz_list, cfg)


def simulate_movie_from_direct_slabs_torch(
    atoms: List[Atom],
    cfg: SimConfig,
    hydration_atlas: Optional[OrientedHydrationAtlas] = None,
    protein_tensors: Optional[OrientedProteinTensors] = None,
) -> torch.Tensor:
    n_frames = max(1, int(cfg.number_of_frames))
    dose_per_frame = cfg.dose_per_frame_e_per_a2
    if dose_per_frame is None:
        dose_per_frame = float(cfg.dose_e_per_a2) / float(n_frames)
    else:
        dose_per_frame = float(dose_per_frame)
    pre = float(cfg.pre_exposure_e_per_a2)
    frame_batch_size = max(1, int(cfg.frame_batch_size))
    if cfg.verbose:
        print(
            f'Torch direct-slab movie: frames={n_frames}, '
            f'dose_per_frame={dose_per_frame:.4f}, pre_exposure={pre:.4f}, '
            f'frame_batch_size={frame_batch_size}'
        )
    with timing_section(cfg, 'protein_slab_generation'):
        if cfg.use_cache_atom:
            base_phase_slabs, base_amp_slabs, dz_list = make_phase_amp_slabs_direct_from_atoms_torch_cached_grouped(
                atoms,
                cfg,
                subpix_n=int(cfg.atom_cache_subpix_n),
                template_radius_pix=int(cfg.atom_cache_radius_pix),
                protein_tensors=protein_tensors,
            )
        else:
            base_phase_slabs, base_amp_slabs, dz_list = make_phase_amp_slabs_direct_from_atoms_torch(atoms, cfg)
    base_phase_stack = as_slab_stack_torch(base_phase_slabs).contiguous()
    base_amp_stack = as_slab_stack_torch(base_amp_slabs).contiguous()
    with timing_section(cfg, 'water_cache_prepare'):
        water_cache = prepare_water_cache_torch(atoms, cfg, hydration_atlas)
    frames: List[torch.Tensor] = []
    summed: Optional[torch.Tensor] = None
    original_total_dose = float(cfg.dose_e_per_a2)
    try:
        cfg.dose_e_per_a2 = float(dose_per_frame)
        for batch_start in range(0, n_frames, frame_batch_size):
            batch_end = min(n_frames, batch_start + frame_batch_size)
            frame_indices = list(range(batch_start, batch_end))
            phase_batch, amp_batch = _prepare_frame_batch_slabs_from_base_torch(
                base_phase_stack, base_amp_stack, dz_list, cfg, water_cache,
                frame_indices, float(dose_per_frame), pre,
            )
            with timing_section(cfg, 'image_formation_total'):
                if cfg.mode == 'projection':
                    images = simulate_projection_from_slab_stack_batch_torch(phase_batch, cfg)
                elif cfg.mode == 'multislice':
                    images = propagate_slab_stack_batch_cistem_like_torch(
                        phase_batch, amp_batch, dz_list, cfg
                    )
                else:
                    raise ValueError(f'Unknown mode: {cfg.mode}')
            if images.ndim == 2:
                images = images.unsqueeze(0)
            with timing_section(cfg, 'frame_accumulation'):
                if cfg.save_frames:
                    frames.append(images.contiguous())
                else:
                    batch_sum = images.sum(dim=0).to(torch.float32)
                    summed = batch_sum if summed is None else summed + batch_sum
    finally:
        cfg.dose_e_per_a2 = original_total_dose
    if cfg.save_frames:
        if not frames:
            raise RuntimeError('No frames were generated')
        return torch.cat(frames, dim=0).to(torch.float32)
    if summed is None:
        raise RuntimeError('No frames were generated')
    if cfg.normalize_frame_sum:
        summed = summed / float(n_frames)
    return summed.to(torch.float32)


def run(cfg: SimConfig, pdb_path: str | Path, output_path: str | Path) -> np.ndarray:
    device = torch_device_from_cfg(cfg)
    recorder = TimingRecorder(bool(cfg.timing or cfg.timing_json), device)
    setattr(cfg, '_timing_recorder', recorder)
    recorder.set_meta('pdb', str(Path(pdb_path).resolve()))
    recorder.set_meta('output', str(Path(output_path).resolve()))
    recorder.set_meta('device', str(device))
    recorder.set_meta('program_version', PROGRAM_VERSION)
    recorder.set_meta('optimization_base_sha256', OPTIMIZATION_BASE_SHA256)
    recorder.start()
    try:
        with timing_section(cfg, 'pdb_read_and_center'):
            protein_frame_atoms = read_pdb_atoms(pdb_path, use_hydrogen=cfg.use_hydrogen)
            if cfg.center_by_mass:
                center_atoms(protein_frame_atoms)
        recorder.set_counter('protein_atom_count', len(protein_frame_atoms))
        with timing_section(cfg, 'orientation_and_ice_placement'):
            rotation = effective_euler_rotation_matrix(cfg)
            atoms = clone_atoms(protein_frame_atoms)
            apply_rotation_matrix_to_atoms(atoms, rotation)
            work_cfg, final_box, solvent_pad = make_cistem_work_config(cfg)
            setattr(work_cfg, '_timing_recorder', recorder)
            z_shift = place_protein_in_ice_z(atoms, work_cfg)
            translation = np.asarray([0.0, 0.0, z_shift], dtype=np.float64)

        recorder.set_meta('mode', work_cfg.mode)
        recorder.set_meta('final_box', int(final_box))
        recorder.set_meta('working_box', int(work_cfg.box))
        recorder.set_meta('n_slices', int(work_cfg.n_slices))
        recorder.set_meta('number_of_frames', int(work_cfg.number_of_frames if work_cfg.per_frame else 1))
        recorder.set_meta('ice_thickness_a', float(physical_ice_thickness_a(work_cfg)))

        hydration_view: Optional[OrientedHydrationAtlas] = None
        if work_cfg.explicit_water and work_cfg.water_soft_weight:
            with timing_section(work_cfg, 'hydration_atlas_cache_lookup'):
                atlas, cache_hit = get_or_build_hydration_atlas(protein_frame_atoms, work_cfg)
            with timing_section(work_cfg, 'hydration_atlas_orientation_setup'):
                hydration_view = orient_hydration_atlas(atlas, rotation, translation)
            recorder.set_meta('hydration_atlas_cache', 'hit' if cache_hit else 'miss')

        oriented_protein_tensors: Optional[OrientedProteinTensors] = None
        if work_cfg.use_cache_atom:
            with timing_section(work_cfg, 'protein_tensor_cache_lookup'):
                protein_data, protein_cache_hit = get_or_build_protein_tensor_data(
                    protein_frame_atoms, work_cfg, device
                )
            with timing_section(work_cfg, 'protein_tensor_orientation'):
                oriented_protein_tensors = orient_protein_tensors(
                    protein_data, rotation, translation
                )
            recorder.set_meta('protein_tensor_cache', 'hit' if protein_cache_hit else 'miss')

        if cfg.verbose:
            print(f'Loaded atoms: {len(atoms)}')
            if solvent_pad > 0:
                print(
                    f'Using solvent guard band: final_box={final_box}, '
                    f'working_box={work_cfg.box}, pad={solvent_pad} px'
                )
            if has_manual_ice_thickness(work_cfg):
                _, dz = continuous_slab_z_geometry(work_cfg)
                print(
                    f'Ice thickness={physical_ice_thickness_a(work_cfg):.3f} A, '
                    f'placement={work_cfg.ice_protein_position}, z_shift={z_shift:.3f} A, '
                    f'slabs={len(dz)}, slab_dz={dz[0]:.4f} A'
                )
            if hydration_view is not None:
                print(
                    f'3D hydration atlas: shape_zyx={tuple(hydration_view.atlas.weights.shape)}, '
                    f'spacing={hydration_view.atlas.spacing_a:.3f} A, '
                    f'cutoff={hydration_view.atlas.cutoff_a:.3f} A'
                )

        if work_cfg.per_frame:
            image_tensor = simulate_movie_from_direct_slabs_torch(
                atoms, work_cfg, hydration_view, oriented_protein_tensors
            )
        else:
            phase_slabs, amp_slabs, dz_list = prepare_phase_amp_slabs_direct_torch(
                atoms, work_cfg, hydration_view, oriented_protein_tensors
            )
            with timing_section(work_cfg, 'image_formation_total'):
                if work_cfg.mode == 'projection':
                    image_tensor = simulate_projection_from_slabs_torch(phase_slabs, work_cfg)
                elif work_cfg.mode == 'multislice':
                    image_tensor = propagate_slabs_cistem_like_torch(
                        phase_slabs, amp_slabs, dz_list, work_cfg
                    )
                else:
                    raise ValueError(f'Unknown mode: {work_cfg.mode}')
        with timing_section(work_cfg, 'crop_and_device_to_host'):
            image_tensor = center_crop_torch(image_tensor, final_box)
            image_numpy = image_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        with timing_section(work_cfg, 'write_output'):
            save_image_or_volume(output_path, image_numpy, cfg.pixel_size)
        return image_numpy
    finally:
        recorder.stop()
        recorder.capture_cuda_memory()
        recorder.emit(output_path, cfg.timing_json)

def parse_args(argv: Optional[Iterable[str]]=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Lean PyTorch cisTEM-like direct-slab simulator with explicit water and a cached 3D hydration atlas.')
    parser.add_argument('--version', action='version', version=f'%(prog)s {PROGRAM_VERSION} (release-base {OPTIMIZATION_BASE_SHA256[:12]})')
    parser.add_argument('pdb', help='Input PDB file')
    parser.add_argument('output', help='Output .mrc, .mrcs, or .npy image')
    parser.add_argument('--box', type=int, default=256)
    parser.add_argument('--pixel-size', type=float, default=1.0)
    parser.add_argument('--kv', type=float, default=300.0)
    parser.add_argument('--cs', type=float, default=2.7)
    parser.add_argument('--defocus', type=float, default=15000.0)
    parser.add_argument('--defocus-v', type=float, default=None)
    parser.add_argument('--defocus-angle', type=float, default=0.0)
    parser.add_argument('--amplitude-contrast', type=float, default=0.07)
    parser.add_argument('--phase-shift', type=float, default=0.0)
    parser.add_argument('--dose', type=float, default=30.0)
    parser.add_argument('--mode', choices=['projection', 'multislice'], default='multislice')
    parser.add_argument('--n-slices', type=int, default=50)
    parser.add_argument('--ice-thickness', type=float, default=None)
    parser.add_argument('--ice-protein-position', choices=['center', 'top', 'bottom'], default='center')
    parser.add_argument('--ice-surface-clearance', type=float, default=0.0)
    parser.add_argument('--inelastic-potential', action='store_true')
    parser.add_argument('--min-bfactor', type=float, default=15.0)
    parser.add_argument('--bfactor-scaling', type=float, default=0.0)
    parser.add_argument('--bond-scaling', type=float, default=BOND_SCALING_DEFAULT)
    parser.add_argument('--use-hydrogen', action='store_true')
    parser.add_argument('--no-center', action='store_true')
    parser.add_argument('--rot', type=float, default=0.0)
    parser.add_argument('--tilt', type=float, default=0.0)
    parser.add_argument('--psi', type=float, default=0.0)
    parser.add_argument('--euler-inverse', action='store_true')
    parser.add_argument('--explicit-water', action='store_true')
    parser.add_argument('--water-density-scale', type=float, default=1.0)
    parser.add_argument('--water-max-count', type=int, default=None)
    parser.add_argument('--water-template-radius-pix', type=int, default=4)
    parser.add_argument('--water-subpix-n', type=int, default=5)
    parser.add_argument('--water-exclude-below', type=float, default=2.5)
    parser.add_argument('--water-soft-weight', action='store_true', help='Use a true 3D nearest-neighbor hydration-weight atlas')
    parser.add_argument('--water-soft-atlas-spacing', type=float, default=1.5, help='Hydration atlas spacing in Angstrom; 1.5-2.0 is recommended')
    parser.add_argument('--water-soft-atlas-cutoff', type=float, default=9.0)
    parser.add_argument('--water-soft-atlas-atom-chunk-size', type=int, default=128)
    parser.add_argument('--water-soft-atlas-cache-entries', type=int, default=2)
    parser.add_argument('--water-bfactor', type=float, default=34.0)
    parser.add_argument('--solvent-padding-pix', type=int, default=64)
    parser.add_argument('--edge-taper-width-pix', type=int, default=24)
    parser.add_argument('--sampled-mask-erode-pix', type=int, default=7)
    parser.add_argument('--sampled-mask-lowpass', type=float, default=0.05)
    parser.add_argument('--disable-cistem-edge-pipeline', action='store_true')
    parser.add_argument('--objective-aperture', type=float, default=100.0)
    parser.add_argument('--objective-aperture-falloff-pix', type=float, default=14.0)
    parser.add_argument('--disable-cistem-inelastic-filter', action='store_true')
    parser.add_argument('--disable-cistem-defocus-offset', action='store_true')
    parser.add_argument('--radiation-damage', action='store_true')
    parser.add_argument('--pre-exposure', type=float, default=0.0)
    parser.add_argument('--radiation-damage-where', choices=['protein', 'all'], default='protein')
    parser.add_argument('--exposure-filter-modify-signal', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--per-frame', action='store_true')
    parser.add_argument('--number-of-frames', type=int, default=1)
    parser.add_argument('--dose-per-frame', type=float, default=None)
    parser.add_argument('--use-cache-atom', action='store_true')
    parser.add_argument('--atom-cache-subpix-n', type=int, default=9)
    parser.add_argument('--atom-cache-radius-pix', type=int, default=9)
    parser.add_argument('--atom-template-chunk-size', type=int, default=16)
    parser.add_argument('--protein-tensor-cache-entries', type=int, default=2, help='Persistent worker LRU entries for protein-frame atom tensors')
    parser.add_argument('--save-frames', action='store_true')
    parser.add_argument('--no-normalize-frame-sum', action='store_true')
    parser.add_argument('--shake-waters', action='store_true')
    parser.add_argument('--frame-batch-size', type=int, default=1)
    parser.add_argument('--water-template-chunk-size', type=int, default=16, help='Retained for command compatibility; ignored by the fused Q^2 water backend')
    parser.add_argument('--water-slab-chunk-size', type=int, default=0, help='Slabs per fused Q^2-channel water convolution; 0 processes all slabs in one call (fastest, highest memory)')
    parser.add_argument('--water-seed-z-chunk', type=int, default=16)
    parser.add_argument('--water-seed-max-octants-per-chunk', type=int, default=67108864)
    parser.add_argument('--water-filter-chunk-size', type=int, default=250000)
    parser.add_argument('--water-filter-cell-size', type=float, default=None)
    parser.add_argument('--dqe', action='store_true')
    parser.add_argument('--poisson', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--timing', action='store_true', help='Synchronize CUDA at stage boundaries and print a timing report')
    parser.add_argument('--timing-json', default=None, help='Optional JSON path; supports {pid}, {output_stem}, and {output_name}')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args(argv)

def main(argv: Optional[Iterable[str]]=None) -> None:
    args = parse_args(argv)
    cfg = SimConfig(box=args.box, pixel_size=args.pixel_size, kv=args.kv, cs_mm=args.cs, defocus_u=args.defocus, defocus_v=args.defocus_v, defocus_angle_deg=args.defocus_angle, amplitude_contrast=args.amplitude_contrast, phase_shift_rad=args.phase_shift, dose_e_per_a2=args.dose, n_slices=args.n_slices, ice_thickness_a=args.ice_thickness, ice_protein_position=args.ice_protein_position, ice_surface_clearance_a=args.ice_surface_clearance, inelastic_potential=bool(args.inelastic_potential), min_bfactor=args.min_bfactor, bfactor_scaling=args.bfactor_scaling, bond_scaling=args.bond_scaling, center_by_mass=not args.no_center, euler_rot_deg=args.rot, euler_tilt_deg=args.tilt, euler_psi_deg=args.psi, euler_inverse=args.euler_inverse, use_hydrogen=args.use_hydrogen, explicit_water=args.explicit_water, water_density_scale=args.water_density_scale, water_max_count=args.water_max_count, water_template_radius_pix=args.water_template_radius_pix, water_subpix_n=args.water_subpix_n, water_exclude_below_a=args.water_exclude_below, water_soft_weight=args.water_soft_weight, water_soft_atlas_spacing_a=args.water_soft_atlas_spacing, water_soft_atlas_cutoff_a=args.water_soft_atlas_cutoff, water_soft_atlas_atom_chunk_size=max(1, int(args.water_soft_atlas_atom_chunk_size)), water_soft_atlas_cache_entries=max(1, int(args.water_soft_atlas_cache_entries)), water_bfactor=args.water_bfactor, mode=args.mode, poisson=args.poisson, dqe=args.dqe, seed=args.seed, verbose=args.verbose, radiation_damage=args.radiation_damage, radiation_damage_where=args.radiation_damage_where, pre_exposure_e_per_a2=args.pre_exposure, exposure_filter_modify_signal=args.exposure_filter_modify_signal, per_frame=args.per_frame, number_of_frames=args.number_of_frames, dose_per_frame_e_per_a2=args.dose_per_frame, save_frames=args.save_frames, normalize_frame_sum=not args.no_normalize_frame_sum, shake_waters=args.shake_waters, use_cache_atom=args.use_cache_atom, device=args.device, solvent_padding_pix=args.solvent_padding_pix, edge_taper_width_pix=args.edge_taper_width_pix, sampled_mask_erode_pix=args.sampled_mask_erode_pix, sampled_mask_lowpass=args.sampled_mask_lowpass, disable_cistem_edge_pipeline=args.disable_cistem_edge_pipeline, frame_batch_size=max(1, int(args.frame_batch_size)), water_template_chunk_size=max(1, int(args.water_template_chunk_size)), water_slab_chunk_size=int(args.water_slab_chunk_size), water_seed_z_chunk=max(1, int(args.water_seed_z_chunk)), water_seed_max_octants_per_chunk=max(8, int(args.water_seed_max_octants_per_chunk)), water_filter_chunk_size=max(1, int(args.water_filter_chunk_size)), water_filter_cell_size_a=args.water_filter_cell_size, atom_cache_subpix_n=max(1, int(args.atom_cache_subpix_n)), atom_cache_radius_pix=max(1, int(args.atom_cache_radius_pix)), atom_template_chunk_size=max(1, int(args.atom_template_chunk_size)), protein_tensor_cache_entries=max(1, int(args.protein_tensor_cache_entries)), objective_aperture_diameter_micron=float(args.objective_aperture), objective_aperture_falloff_pix=float(args.objective_aperture_falloff_pix), disable_cistem_inelastic_filter=bool(args.disable_cistem_inelastic_filter), disable_cistem_defocus_offset=bool(args.disable_cistem_defocus_offset), timing=bool(args.timing or args.timing_json), timing_json=args.timing_json)
    run(cfg, args.pdb, args.output)
if __name__ == '__main__':
    main()
