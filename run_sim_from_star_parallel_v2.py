#!/usr/bin/env python3
"""
Persistent-worker RELION STAR runner for cisTEM_simulate_pytorch.

The original wrapper launches one new Python subprocess per simulated image.
On systems where importing PyTorch is slow, that makes startup dominate the
runtime.  This replacement starts one long-lived worker per GPU (or per CPU
job), imports cistem_simulate_torch_direct_slabs.py once inside each worker,
and then processes many STAR rows in that same process.

Important properties
--------------------
* One image at a time per GPU worker; no GPU is intentionally oversubscribed.
* A shared task queue provides dynamic load balancing across GPUs.
* PyTorch is not imported by the parent process.
* Each GPU worker sets/binds its CUDA device before importing the simulator.
* The simulator's public ``main(argv)`` entry point is called directly; no
  per-image subprocess is created.
* The historical wrapper defaults are retained:
    --mode multislice
    --use-cache-atom
    --no-center
    --explicit-water
    --radiation-damage
    --radiation-damage-where all
    --per-frame
    --shake-waters
* Temporary single-image MRC files remain available for restart with
  --skip-existing.  Final MRCS assembly is streamed through a memory map rather
  than loading the complete stack into RAM.

Preferred command line
----------------------
python run_sim_from_star_parallel.py MODEL.pdb INPUT.star OUTPUT.mrcs \\
    --output-star OUTPUT.star \\
    --gpu-ids 0,1,2,3 \\
    --defocus-min-um 0.8 --defocus-max-um 2.0 \\
    --keep-tmp --inverse-contrast \\
    -- --pixel-size 1.5 --box 320 --n-slices 50

``--inverse-contrast`` is a wrapper option and therefore belongs before ``--``.
Simulator options may also be written before ``--`` for compatibility with the
old script; unknown options are forwarded to the simulator.

The old four-positional-argument form is also accepted, so existing commands
continue to work:
python run_sim_from_star_parallel.py \\
    cistem_simulate_torch_direct_slabs.py MODEL.pdb INPUT.star OUTPUT.mrcs ...

Notes
-----
A multi-GPU run necessarily has one Python/PyTorch process per GPU.  Therefore
"load once" means once per persistent GPU worker, rather than once for the
entire multi-GPU job.  This changes the old cost from O(number of images)
PyTorch imports to O(number of GPUs) imports.
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import queue
import shlex
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import mrcfile  # type: ignore
except Exception:  # pragma: no cover - handled with a clear runtime error
    mrcfile = None


# These are the options that were appended unconditionally by the original
# run_sim_from_star_parallel.py.  Here they behave as defaults: a user-provided
# value such as ``--mode projection`` is respected.  Use
# ``--no-wrapper-defaults`` to turn off all of these defaults at once.
WRAPPER_SIMULATOR_DEFAULTS: Tuple[Tuple[str, Optional[str]], ...] = (
    ("--mode", "multislice"),
    ("--use-cache-atom", None),
    ("--no-center", None),
    ("--explicit-water", None),
    ("--radiation-damage", None),
    ("--radiation-damage-where", "all"),
    ("--per-frame", None),
    ("--shake-waters", None),
)

DEFAULT_SIMULATOR_FILENAME = "cistem_simulate_torch_direct_slabs.py"
DEFAULT_BOX = 256
DEFAULT_PIXEL_SIZE = 1.0


@dataclass(frozen=True)
class SimulationTask:
    """One selected STAR row to be simulated."""

    local_index: int
    global_index: int
    output_image: str
    rot: float
    tilt: float
    psi: float
    defocus_angstrom: Optional[float]


@dataclass(frozen=True)
class WorkerSpec:
    """CUDA binding and simulator device for one persistent worker."""

    worker_id: int
    label: str
    cuda_visible_devices: Optional[str]
    simulator_device: str


# ---------------------------------------------------------------------------
# STAR parsing and writing
# ---------------------------------------------------------------------------


def _split_star_row(line: str) -> List[str]:
    """Split a simple STAR data row while retaining quoted fields."""

    try:
        return shlex.split(line, comments=False, posix=True)
    except ValueError:
        # RELION image names and generated sampling files normally need only
        # whitespace splitting.  This fallback keeps the parser permissive if a
        # malformed quote appears in an unrelated field.
        return line.split()


def parse_relion_star_angles(
    star_path: Union[str, os.PathLike[str]],
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read the ``data_particles`` loop from a RELION STAR file.

    This intentionally remains a small dependency-free parser.  It supports the
    RELION sampling STAR produced by generate_healpix_order_and_relion_star.py
    and ordinary one-line particle loops.  Semicolon-delimited multiline STAR
    values are not supported because they are not expected in particle rows.
    """

    path = Path(star_path)
    with path.open("r", errors="replace") as handle:
        lines = handle.readlines()

    particles_line: Optional[int] = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "data_particles":
            particles_line = i
            break
    if particles_line is None:
        raise ValueError(f"No data_particles block found in {path}")

    loop_line: Optional[int] = None
    for i in range(particles_line + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.lower() == "loop_":
            loop_line = i
            break
        if stripped.lower().startswith("data_"):
            break
    if loop_line is None:
        raise ValueError(f"No loop_ found in data_particles block of {path}")

    labels: List[str] = []
    row_start: Optional[int] = None
    for i in range(loop_line + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("_"):
            labels.append(stripped.split()[0])
            continue
        row_start = i
        break

    if not labels:
        raise ValueError(f"No labels found in data_particles loop of {path}")
    if row_start is None:
        raise ValueError(f"No particle rows found in {path}")

    rows: List[Dict[str, str]] = []
    for line_number in range(row_start, len(lines)):
        stripped = lines[line_number].strip()
        if not stripped or stripped.startswith("#"):
            continue
        lower = stripped.lower()
        if lower == "loop_" or lower.startswith("data_") or stripped.startswith("_"):
            break

        parts = _split_star_row(stripped)
        if len(parts) < len(labels):
            raise ValueError(
                f"STAR row {line_number + 1} in {path} has {len(parts)} values "
                f"but the particle loop has {len(labels)} labels"
            )
        rows.append({label: value for label, value in zip(labels, parts)})

    required = ("_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi")
    missing = [label for label in required if label not in labels]
    if missing:
        raise ValueError(f"Missing required STAR labels in {path}: {missing}")
    if not rows:
        raise ValueError(f"No particle rows found in {path}")

    return rows, labels


def write_output_star(
    output_star: Union[str, os.PathLike[str]],
    rows: Sequence[Dict[str, str]],
    labels: Sequence[str],
    output_mrcs_name: Union[str, os.PathLike[str]],
    defocus_values_angstrom: Optional[np.ndarray] = None,
) -> None:
    """Write a RELION-style particle STAR matching the historical wrapper."""

    labels_out = list(labels)
    if "_rlnImageName" not in labels_out:
        labels_out.append("_rlnImageName")
    for label in ("_rlnDefocusU", "_rlnDefocusV", "_rlnDefocusAngle"):
        if label not in labels_out:
            labels_out.append(label)

    if defocus_values_angstrom is not None and len(defocus_values_angstrom) != len(rows):
        raise ValueError(
            "Length of defocus_values_angstrom does not match the number of output rows"
        )

    output_path = Path(output_star)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stack_basename = Path(output_mrcs_name).name

    with output_path.open("w") as handle:
        handle.write("# relion simulated\n\n")
        handle.write("data_particles\n\n")
        handle.write("loop_\n")
        for i, label in enumerate(labels_out, start=1):
            handle.write(f"{label} #{i}\n")

        for i, row in enumerate(rows, start=1):
            if defocus_values_angstrom is not None:
                defocus_u = float(defocus_values_angstrom[i - 1])
                defocus_v = defocus_u
                defocus_angle = 0.0
            else:
                defocus_u = float(row.get("_rlnDefocusU", 0.0))
                defocus_v = float(row.get("_rlnDefocusV", defocus_u))
                defocus_angle = float(row.get("_rlnDefocusAngle", 0.0))

            values: List[str] = []
            for label in labels_out:
                if label == "_rlnImageName":
                    values.append(f"{i}@{stack_basename}")
                elif label == "_rlnDefocusU":
                    values.append(f"{defocus_u:.6f}")
                elif label == "_rlnDefocusV":
                    values.append(f"{defocus_v:.6f}")
                elif label == "_rlnDefocusAngle":
                    values.append(f"{defocus_angle:.6f}")
                else:
                    values.append(str(row.get(label, "0")))
            handle.write("\t".join(values) + "\n")


# ---------------------------------------------------------------------------
# Simulator argument helpers
# ---------------------------------------------------------------------------


def _option_is_present(arguments: Sequence[str], name: str) -> bool:
    prefix = name + "="
    return any(arg == name or arg.startswith(prefix) for arg in arguments)


def _last_option_value(
    arguments: Sequence[str],
    name: str,
    default: Optional[str] = None,
) -> Optional[str]:
    """Return the last effective value of ``--name value`` or ``--name=value``."""

    result = default
    prefix = name + "="
    i = 0
    while i < len(arguments):
        arg = arguments[i]
        if arg == name:
            if i + 1 >= len(arguments):
                raise ValueError(f"Missing value after simulator option {name}")
            result = arguments[i + 1]
            i += 2
            continue
        if arg.startswith(prefix):
            result = arg[len(prefix) :]
        i += 1
    return result


def apply_wrapper_simulator_defaults(
    simulator_args: Sequence[str],
    disabled: bool,
) -> List[str]:
    """Apply the old wrapper's hard-coded settings as overridable defaults."""

    result = list(simulator_args)
    if disabled:
        return result

    for option, value in WRAPPER_SIMULATOR_DEFAULTS:
        if _option_is_present(result, option):
            continue
        result.append(option)
        if value is not None:
            result.append(value)
    return result


def infer_box_and_pixel_size(simulator_args: Sequence[str]) -> Tuple[int, float]:
    """Infer simulator geometry, falling back to the simulator's own defaults."""

    box_text = _last_option_value(simulator_args, "--box", str(DEFAULT_BOX))
    pixel_text = _last_option_value(
        simulator_args, "--pixel-size", str(DEFAULT_PIXEL_SIZE)
    )
    assert box_text is not None
    assert pixel_text is not None
    box = int(box_text)
    pixel_size = float(pixel_text)
    if box <= 0:
        raise ValueError("--box must be positive")
    if pixel_size <= 0.0:
        raise ValueError("--pixel-size must be positive")
    return box, pixel_size


def generate_random_defocus_values_angstrom(
    n_particles: int,
    defocus_min_um: float,
    defocus_max_um: float,
    box: int,
    pixel_size: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate the same random-defocus convention used by the old wrapper."""

    if defocus_max_um < defocus_min_um:
        raise ValueError("--defocus-max-um must be >= --defocus-min-um")

    rng = np.random.default_rng(seed)
    defocus_um = rng.uniform(
        float(defocus_min_um),
        float(defocus_max_um),
        size=int(n_particles),
    ).astype(np.float64)
    offset_angstrom = 0.5 * float(box) * float(pixel_size)
    return (defocus_um * 10000.0 + offset_angstrom).astype(np.float64)


def parse_gpu_ids(gpu_ids_text: Optional[str]) -> Optional[List[str]]:
    if gpu_ids_text is None or not gpu_ids_text.strip():
        return None
    values = [value.strip() for value in gpu_ids_text.split(",") if value.strip()]
    if not values:
        return None
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate GPU IDs are not allowed: {values}")
    return values


def _inherited_visible_gpu_tokens() -> Optional[List[str]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    raw = raw.strip()
    if not raw or raw in {"-1", "NoDevFiles", "none", "None"}:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return tokens or None


def build_worker_specs(
    device: str,
    gpu_ids: Optional[Sequence[str]],
    jobs: Optional[int],
) -> List[WorkerSpec]:
    """Build one non-oversubscribed worker specification per selected GPU."""

    device_lower = device.lower()
    is_cuda = device_lower.startswith("cuda")

    if jobs is not None and jobs < 1:
        raise ValueError("--jobs must be >= 1")

    if not is_cuda:
        if gpu_ids:
            raise ValueError("--gpu-ids cannot be used when --device is not CUDA")
        count = 2 if jobs is None else jobs
        return [
            WorkerSpec(i, f"CPU worker {i}", None, device)
            for i in range(count)
        ]

    if gpu_ids:
        selected = list(gpu_ids)
        if jobs is not None:
            selected = selected[: min(jobs, len(selected))]
        return [
            WorkerSpec(
                worker_id=i,
                label=f"GPU {gpu_id}",
                cuda_visible_devices=str(gpu_id),
                simulator_device="cuda",
            )
            for i, gpu_id in enumerate(selected)
        ]

    # When a scheduler already supplies CUDA_VISIBLE_DEVICES, use one worker per
    # inherited logical CUDA device without rewriting the environment.  This is
    # safer than interpreting scheduler-local IDs as host-physical IDs.
    inherited = _inherited_visible_gpu_tokens()
    if device_lower == "cuda" and inherited and len(inherited) > 1:
        count = len(inherited) if jobs is None else min(jobs, len(inherited))
        return [
            WorkerSpec(
                worker_id=i,
                label=f"visible cuda:{i} ({inherited[i]})",
                cuda_visible_devices=None,
                simulator_device=f"cuda:{i}",
            )
            for i in range(count)
        ]

    # No explicit GPU list: deliberately use only one worker.  Creating several
    # workers with the same plain ``cuda`` device would oversubscribe one GPU and
    # violate the one-image-at-a-time-per-GPU requirement.  ``--jobs`` is capped
    # rather than rejected so the historical default of 2 remains usable on a
    # single-GPU machine.
    return [WorkerSpec(0, device, None, device)]


# ---------------------------------------------------------------------------
# MRC I/O
# ---------------------------------------------------------------------------


def _require_mrcfile() -> None:
    if mrcfile is None:
        raise ImportError("mrcfile is required. Install it with: pip install mrcfile")


def read_mrc_2d(path: Union[str, os.PathLike[str]]) -> np.ndarray:
    _require_mrcfile()
    assert mrcfile is not None
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
        if data.ndim == 3:
            if data.shape[0] != 1:
                raise ValueError(
                    f"{path} has shape {data.shape}; expected one 2D image. "
                    "Do not use simulator option --save-frames with this runner."
                )
            data = data[0]
        if data.ndim != 2:
            raise ValueError(f"{path} has shape {data.shape}; expected a 2D image")
        # The MRC object is about to close, so detach from its memory map.
        return np.array(data, dtype=np.float32, copy=True)


def write_mrcs_stack_streaming(
    output_mrcs: Union[str, os.PathLike[str]],
    image_paths_in_order: Sequence[str],
    pixel_size: Optional[float],
    inverse_contrast: bool,
) -> None:
    """Merge images into MRCS without materializing the full stack in RAM."""

    _require_mrcfile()
    assert mrcfile is not None
    if not image_paths_in_order:
        raise ValueError("No images were provided for MRCS assembly")

    first = read_mrc_2d(image_paths_in_order[0])
    ny, nx = first.shape
    output_path = Path(output_mrcs)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with mrcfile.new_mmap(
        output_path,
        shape=(len(image_paths_in_order), ny, nx),
        mrc_mode=2,  # float32
        overwrite=True,
    ) as mrc:
        if pixel_size is not None:
            mrc.voxel_size = float(pixel_size)

        for i, image_path in enumerate(image_paths_in_order):
            image = first if i == 0 else read_mrc_2d(image_path)
            if image.shape != (ny, nx):
                raise ValueError(
                    f"Image shape mismatch: {image_path} has {image.shape}, "
                    f"expected {(ny, nx)}"
                )
            if inverse_contrast:
                mrc.data[i, :, :] = -image
            else:
                mrc.data[i, :, :] = image

        mrc.update_header_stats()
        mrc.flush()


# ---------------------------------------------------------------------------
# Persistent worker implementation
# ---------------------------------------------------------------------------


def _load_simulator_module(simulator_script: str):
    """Import the simulator from a path exactly once in the current worker."""

    path = Path(simulator_script).resolve()
    simulator_dir = str(path.parent)
    if simulator_dir not in sys.path:
        # Keep future simulator-local imports working even when the runner is
        # launched from a different working directory.
        sys.path.insert(0, simulator_dir)
    module_name = f"_cistem_simulator_worker_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and some runtime annotations expect the module to be present
    # in sys.modules while its source is executed.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    main_function = getattr(module, "main", None)
    if not callable(main_function):
        raise AttributeError(
            f"{path} does not expose callable main(argv); cannot run it in-process"
        )
    return module


def _build_simulator_argv(
    task: SimulationTask,
    pdb_path: str,
    simulator_args: Sequence[str],
    simulator_device: str,
) -> List[str]:
    """Build argv for one direct call to simulator.main(argv)."""

    argv: List[str] = [pdb_path, task.output_image]
    argv.extend(simulator_args)

    # STAR/task-controlled values come last so a duplicate user option cannot
    # accidentally make every row use the same orientation or wrong device.
    argv.extend(
        [
            "--rot",
            f"{task.rot:.12g}",
            "--tilt",
            f"{task.tilt:.12g}",
            "--psi",
            f"{task.psi:.12g}",
        ]
    )
    if task.defocus_angstrom is not None:
        argv.extend(["--defocus", f"{task.defocus_angstrom:.12g}"])
    argv.extend(["--device", simulator_device])
    return argv


def _persistent_worker(
    spec: WorkerSpec,
    simulator_script: str,
    pdb_path: str,
    simulator_args: Sequence[str],
    task_queue,
    result_queue,
) -> None:
    """Worker process: bind GPU, import once, then execute many images."""

    try:
        if spec.cuda_visible_devices is not None:
            # Must happen before importing the simulator, because it imports
            # torch at module scope.
            os.environ["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_devices
            os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

        import_start = time.perf_counter()
        simulator = _load_simulator_module(simulator_script)

        # With an inherited multi-GPU CUDA_VISIBLE_DEVICES list, each worker sees
        # all allocated devices and receives cuda:N.  Set the worker's current
        # CUDA device as well as passing --device, so any simulator operation
        # that relies on PyTorch's current-device state stays on its assigned GPU.
        if spec.simulator_device.lower().startswith("cuda"):
            torch_module = getattr(simulator, "torch", None)
            if torch_module is not None and torch_module.cuda.is_available():
                if ":" in spec.simulator_device:
                    local_device_index = int(spec.simulator_device.rsplit(":", 1)[1])
                else:
                    local_device_index = 0
                torch_module.cuda.set_device(local_device_index)

        import_elapsed = time.perf_counter() - import_start
        result_queue.put(
            (
                "ready",
                spec.worker_id,
                spec.label,
                os.getpid(),
                import_elapsed,
            )
        )
    except BaseException:
        result_queue.put(
            (
                "init_error",
                spec.worker_id,
                spec.label,
                os.getpid(),
                traceback.format_exc(),
            )
        )
        return

    while True:
        task = task_queue.get()
        if task is None:
            break
        assert isinstance(task, SimulationTask)

        started = time.perf_counter()
        try:
            argv = _build_simulator_argv(
                task=task,
                pdb_path=pdb_path,
                simulator_args=simulator_args,
                simulator_device=spec.simulator_device,
            )
            simulator.main(argv)
            elapsed = time.perf_counter() - started
            result_queue.put(
                (
                    "done",
                    task.local_index,
                    task.output_image,
                    elapsed,
                    spec.worker_id,
                    spec.label,
                )
            )
        except BaseException:
            # Never let --skip-existing accept a partial file from a failed run.
            try:
                Path(task.output_image).unlink(missing_ok=True)
            except Exception:
                pass
            result_queue.put(
                (
                    "task_error",
                    task.local_index,
                    task.global_index,
                    task.output_image,
                    spec.worker_id,
                    spec.label,
                    traceback.format_exc(),
                )
            )
            return


def _terminate_workers(processes: Sequence[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)


def run_persistent_workers(
    worker_specs: Sequence[WorkerSpec],
    tasks: Sequence[SimulationTask],
    simulator_script: str,
    pdb_path: str,
    simulator_args: Sequence[str],
    progress_every: int,
) -> Dict[int, str]:
    """Run tasks using one spawned, persistent process per WorkerSpec."""

    if not tasks:
        return {}
    if not worker_specs:
        raise ValueError("No worker specifications were created")

    # Avoid paying an import cost on GPUs that have no remaining image to run.
    worker_specs = list(worker_specs[: min(len(worker_specs), len(tasks))])

    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    processes: List[mp.Process] = []

    for spec in worker_specs:
        process = context.Process(
            target=_persistent_worker,
            args=(
                spec,
                simulator_script,
                pdb_path,
                list(simulator_args),
                task_queue,
                result_queue,
            ),
            name=f"cistem-sim-worker-{spec.worker_id}",
        )
        process.start()
        processes.append(process)

    try:
        ready_workers = set()
        while len(ready_workers) < len(processes):
            try:
                message = result_queue.get(timeout=1.0)
            except queue.Empty:
                dead = [p for p in processes if not p.is_alive() and p.exitcode is not None]
                if dead:
                    details = ", ".join(
                        f"{p.name}: exit code {p.exitcode}" for p in dead
                    )
                    raise RuntimeError(
                        "A worker exited while loading the simulator/PyTorch: " + details
                    )
                continue

            kind = message[0]
            if kind == "ready":
                _, worker_id, label, pid, import_elapsed = message
                ready_workers.add(worker_id)
                print(
                    f"Worker {worker_id} ready on {label} (PID {pid}); "
                    f"simulator/PyTorch import {import_elapsed:.2f} s",
                    flush=True,
                )
            elif kind == "init_error":
                _, worker_id, label, pid, error_text = message
                raise RuntimeError(
                    f"Worker {worker_id} on {label} (PID {pid}) failed during import:\n"
                    f"{error_text}"
                )
            else:
                raise RuntimeError(f"Unexpected worker message during startup: {message!r}")

        # All workers are now initialized.  The shared queue lets the first free
        # GPU claim the next image, which is better balanced than static chunks.
        for task in tasks:
            task_queue.put(task)
        for _ in processes:
            task_queue.put(None)

        completed: Dict[int, str] = {}
        total = len(tasks)
        while len(completed) < total:
            try:
                message = result_queue.get(timeout=1.0)
            except queue.Empty:
                alive = [p for p in processes if p.is_alive()]
                if not alive:
                    exits = ", ".join(
                        f"{p.name}: exit code {p.exitcode}" for p in processes
                    )
                    raise RuntimeError(
                        "All workers exited before every image completed. " + exits
                    )
                continue

            kind = message[0]
            if kind == "done":
                _, local_index, output_image, elapsed, worker_id, label = message
                completed[int(local_index)] = str(output_image)
                n_done = len(completed)
                if (
                    progress_every <= 1
                    or n_done == total
                    or n_done % progress_every == 0
                ):
                    print(
                        f"Finished {n_done}/{total}: selected row {local_index} "
                        f"on {label} (worker {worker_id}, {elapsed:.2f} s)",
                        flush=True,
                    )
            elif kind == "task_error":
                (
                    _,
                    local_index,
                    global_index,
                    output_image,
                    worker_id,
                    label,
                    error_text,
                ) = message
                raise RuntimeError(
                    f"Simulation failed for selected row {local_index} "
                    f"(global STAR row {global_index}, output {output_image}) "
                    f"on {label}, worker {worker_id}:\n{error_text}"
                )
            elif kind == "init_error":
                _, worker_id, label, pid, error_text = message
                raise RuntimeError(
                    f"Worker {worker_id} on {label} (PID {pid}) failed during import:\n"
                    f"{error_text}"
                )
            else:
                raise RuntimeError(f"Unexpected worker message: {message!r}")

        for process in processes:
            process.join()
        bad = [process for process in processes if process.exitcode != 0]
        if bad:
            details = ", ".join(
                f"{process.name}: exit code {process.exitcode}" for process in bad
            )
            raise RuntimeError("One or more workers exited abnormally: " + details)

        return completed
    except BaseException:
        _terminate_workers(processes)
        raise
    finally:
        try:
            task_queue.close()
            result_queue.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_REEXEC_MARKER = "CISTEM_SIM_PERSISTENT_REEXEC"


def _canonical_executable(command: str) -> Path:
    """Resolve an executable path while accepting names such as ``python3``."""

    located = shutil.which(command)
    candidate = Path(located if located is not None else command).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"Python executable not found: {command}")
    if not os.access(candidate, os.X_OK):
        raise PermissionError(f"Python executable is not executable: {candidate}")
    return candidate.resolve()


def _maybe_reexec_with_requested_python(
    requested_python: Optional[str],
    original_argv: Sequence[str],
) -> None:
    """Honor the old ``--python`` option by re-executing this integrated runner."""

    if requested_python is None:
        return
    requested = _canonical_executable(requested_python)
    current = Path(sys.executable).resolve()
    if requested == current:
        return
    if os.environ.get(_REEXEC_MARKER) == "1":
        raise RuntimeError(
            f"Re-executed with {requested}, but sys.executable is still {current}"
        )

    print(
        f"Re-executing integrated runner with requested Python: {requested}",
        flush=True,
    )
    environment = os.environ.copy()
    environment[_REEXEC_MARKER] = "1"
    os.execve(
        str(requested),
        [str(requested), str(Path(__file__).resolve()), *original_argv],
        environment,
    )


def _split_wrapper_and_simulator_argv(
    argv: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Split at ``--`` while keeping pre-separator unknown options compatible."""

    values = list(argv)
    if "--" not in values:
        return values, []
    index = values.index("--")
    return values[:index], values[index + 1 :]


def _extract_legacy_simulator_positional(
    wrapper_argv: List[str],
) -> Tuple[List[str], Optional[str]]:
    """Accept the old leading simulator_script positional argument.

    Existing usage begins with four positional values and the first one is a
    Python file.  The integrated form needs only PDB, STAR and MRCS.
    """

    if len(wrapper_argv) >= 4:
        first = wrapper_argv[0]
        if not first.startswith("-") and first.lower().endswith(".py"):
            return wrapper_argv[1:], first
    return wrapper_argv, None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one simulated image per RELION STAR row using persistent "
            "one-process-per-GPU workers."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdb", help="Input PDB file")
    parser.add_argument("input_star", help="Input RELION STAR file")
    parser.add_argument("output_mrcs", help="Output simulated MRCS stack")
    parser.add_argument(
        "--simulator-script",
        default=None,
        help=(
            "Path to cistem_simulate_torch_direct_slabs.py. Default: the file "
            "with that name next to this runner"
        ),
    )
    parser.add_argument(
        "--output-star",
        default=None,
        help="Output STAR; default is output_mrcs with a .star suffix",
    )
    parser.add_argument(
        "--tmp-dir",
        default="sim_tmp_images",
        help="Directory for restartable single-image MRC files",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Base simulator device (cuda, cuda:N, or cpu)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help=(
            "Maximum persistent workers. The historical default is 2. Set this "
            "to the number of GPUs you want to use; with --gpu-ids it is capped "
            "by the number of listed GPUs"
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        "--gpu-id",
        dest="gpu_ids",
        default=None,
        help=(
            "Comma-separated physical GPU IDs or UUIDs, e.g. 0,1,2,3. "
            "Each persistent process receives one value through "
            "CUDA_VISIBLE_DEVICES"
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Python executable for the integrated runner and all persistent "
            "workers; a different value re-executes this script once"
        ),
    )
    parser.add_argument("--start", type=int, default=0, help="Start row, 0-based")
    parser.add_argument(
        "--max-particles",
        type=int,
        default=None,
        help="Simulate at most this many selected rows",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary single-image MRC files after successful merge",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse nonempty temporary images already present",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print equivalent per-image simulator commands without importing torch",
    )
    parser.add_argument(
        "--defocus-min-um",
        type=float,
        default=None,
        help="Minimum randomly assigned defocus in microns",
    )
    parser.add_argument(
        "--defocus-max-um",
        type=float,
        default=None,
        help="Maximum randomly assigned defocus in microns",
    )
    parser.add_argument(
        "--defocus-seed",
        type=int,
        default=None,
        help="Random seed for defocus assignment",
    )
    parser.add_argument(
        "--inverse-contrast",
        action="store_true",
        help="Multiply images by -1 while assembling the final stack",
    )
    parser.add_argument(
        "--no-wrapper-defaults",
        action="store_true",
        help=(
            "Do not add the historical multislice/explicit-water/radiation-"
            "damage/per-frame defaults; pass every desired simulator option "
            "explicitly"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print one completion message per this many newly simulated images",
    )
    return parser


def build_tasks(
    rows_to_run: Sequence[Dict[str, str]],
    start_global_index: int,
    tmp_dir: Path,
    defocus_values_angstrom: Optional[np.ndarray],
) -> List[SimulationTask]:
    if defocus_values_angstrom is not None and len(defocus_values_angstrom) != len(
        rows_to_run
    ):
        raise ValueError("Defocus array length does not match selected STAR rows")

    tasks: List[SimulationTask] = []
    for local_index, row in enumerate(rows_to_run):
        global_index = start_global_index + local_index
        output_image = tmp_dir / f"sim_{global_index + 1:06d}.mrc"
        tasks.append(
            SimulationTask(
                local_index=local_index,
                global_index=global_index,
                output_image=str(output_image),
                rot=float(row["_rlnAngleRot"]),
                tilt=float(row["_rlnAngleTilt"]),
                psi=float(row["_rlnAnglePsi"]),
                defocus_angstrom=(
                    None
                    if defocus_values_angstrom is None
                    else float(defocus_values_angstrom[local_index])
                ),
            )
        )
    return tasks


def _print_dry_run(
    tasks: Sequence[SimulationTask],
    worker_specs: Sequence[WorkerSpec],
    simulator_script: str,
    pdb_path: str,
    simulator_args: Sequence[str],
) -> None:
    for i, task in enumerate(tasks):
        spec = worker_specs[i % len(worker_specs)]
        argv = _build_simulator_argv(
            task,
            pdb_path,
            simulator_args,
            spec.simulator_device,
        )
        equivalent = [sys.executable, simulator_script, *argv]
        env_prefix = ""
        if spec.cuda_visible_devices is not None:
            env_prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(spec.cuda_visible_devices)} "
        print(
            f"[DRY RUN {spec.label}] {env_prefix}{shlex.join(equivalent)}",
            flush=True,
        )


def main(argv: Optional[Iterable[str]] = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    wrapper_argv, post_separator_simulator_args = _split_wrapper_and_simulator_argv(
        raw_argv
    )
    wrapper_argv, legacy_simulator_script = _extract_legacy_simulator_positional(
        wrapper_argv
    )

    parser = build_argument_parser()
    args, pre_separator_unknown = parser.parse_known_args(wrapper_argv)
    simulator_args = list(pre_separator_unknown) + list(post_separator_simulator_args)
    simulator_args = apply_wrapper_simulator_defaults(
        simulator_args,
        disabled=bool(args.no_wrapper_defaults),
    )

    if _option_is_present(simulator_args, "--save-frames"):
        raise ValueError(
            "--save-frames is incompatible with a one-image-per-STAR-row MRCS. "
            "Remove it or run the single-image simulator directly."
        )
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.max_particles is not None and args.max_particles < 1:
        raise ValueError("--max-particles must be >= 1")

    _maybe_reexec_with_requested_python(args.python, raw_argv)

    script_default = Path(__file__).resolve().with_name(DEFAULT_SIMULATOR_FILENAME)
    simulator_script = Path(
        args.simulator_script or legacy_simulator_script or script_default
    ).expanduser().resolve()
    if not simulator_script.is_file():
        raise FileNotFoundError(f"Simulator script not found: {simulator_script}")

    pdb_path = Path(args.pdb).expanduser().resolve()
    input_star = Path(args.input_star).expanduser().resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    if not input_star.is_file():
        raise FileNotFoundError(f"STAR file not found: {input_star}")

    output_mrcs = Path(args.output_mrcs).expanduser().resolve()
    output_star = (
        Path(args.output_star).expanduser().resolve()
        if args.output_star
        else output_mrcs.with_suffix(".star")
    )
    tmp_dir = Path(args.tmp_dir).expanduser().resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    box, pixel_size = infer_box_and_pixel_size(simulator_args)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    worker_specs = build_worker_specs(args.device, gpu_ids, args.jobs)

    if args.device.lower().startswith("cuda") and args.jobs > len(worker_specs):
        print(
            f"CUDA worker count capped at {len(worker_specs)} to keep one "
            "simultaneous image per selected GPU. Provide --gpu-ids or a "
            "multi-device CUDA_VISIBLE_DEVICES allocation to use more GPUs.",
            flush=True,
        )
    print(
        "Persistent workers: " + ", ".join(spec.label for spec in worker_specs),
        flush=True,
    )
    print(
        f"Effective simulator geometry: box={box}, pixel_size={pixel_size:g} A/pixel",
        flush=True,
    )
    if not args.no_wrapper_defaults:
        print(
            "Wrapper defaults: multislice, cache-atom, no-center, explicit-water, "
            "radiation-damage(all), per-frame, shake-waters",
            flush=True,
        )

    rows, labels = parse_relion_star_angles(input_star)
    start = int(args.start)
    if start >= len(rows):
        raise ValueError(
            f"--start {start} is outside the STAR table containing {len(rows)} rows"
        )
    end = len(rows)
    if args.max_particles is not None:
        end = min(end, start + int(args.max_particles))
    rows_to_run = rows[start:end]
    if not rows_to_run:
        raise ValueError("No STAR rows were selected")

    defocus_values_angstrom: Optional[np.ndarray] = None
    random_defocus_requested = (
        args.defocus_min_um is not None or args.defocus_max_um is not None
    )
    if random_defocus_requested:
        if args.defocus_min_um is None or args.defocus_max_um is None:
            raise ValueError(
                "Provide both --defocus-min-um and --defocus-max-um"
            )
        defocus_values_angstrom = generate_random_defocus_values_angstrom(
            n_particles=len(rows_to_run),
            defocus_min_um=float(args.defocus_min_um),
            defocus_max_um=float(args.defocus_max_um),
            box=box,
            pixel_size=pixel_size,
            seed=args.defocus_seed,
        )
        print(
            "Random defocus enabled: "
            f"{args.defocus_min_um:.4f} to {args.defocus_max_um:.4f} um; "
            f"added box-center offset={0.5 * box * pixel_size:.3f} A",
            flush=True,
        )
        if args.skip_existing and args.defocus_seed is None:
            print(
                "Warning: --skip-existing with random defocus but no "
                "--defocus-seed can make reused images disagree with newly "
                "written STAR defocus values.",
                flush=True,
            )

    all_tasks = build_tasks(
        rows_to_run=rows_to_run,
        start_global_index=start,
        tmp_dir=tmp_dir,
        defocus_values_angstrom=defocus_values_angstrom,
    )

    completed: Dict[int, str] = {}
    tasks_to_run: List[SimulationTask] = []
    if args.skip_existing and not args.dry_run:
        for task in all_tasks:
            path = Path(task.output_image)
            if path.is_file() and path.stat().st_size > 0:
                completed[task.local_index] = task.output_image
            else:
                tasks_to_run.append(task)
        print(
            f"skip-existing: {len(completed)} existing, "
            f"{len(tasks_to_run)} remaining",
            flush=True,
        )
    else:
        tasks_to_run = list(all_tasks)

    print(
        f"Selected STAR rows: {len(rows_to_run)}; new simulations: {len(tasks_to_run)}",
        flush=True,
    )

    if args.dry_run:
        _print_dry_run(
            tasks=tasks_to_run,
            worker_specs=worker_specs,
            simulator_script=str(simulator_script),
            pdb_path=str(pdb_path),
            simulator_args=simulator_args,
        )
        print("Dry run finished; no simulator/PyTorch import and no output written.")
        return

    newly_completed = run_persistent_workers(
        worker_specs=worker_specs,
        tasks=tasks_to_run,
        simulator_script=str(simulator_script),
        pdb_path=str(pdb_path),
        simulator_args=simulator_args,
        progress_every=int(args.progress_every),
    )
    completed.update(newly_completed)

    image_paths_in_order: List[str] = []
    for local_index in range(len(rows_to_run)):
        output_image = completed.get(local_index)
        if output_image is None:
            raise RuntimeError(f"Missing completed output for selected row {local_index}")
        image_paths_in_order.append(output_image)

    print("Merging temporary images into the final MRCS stack...", flush=True)
    write_mrcs_stack_streaming(
        output_mrcs=output_mrcs,
        image_paths_in_order=image_paths_in_order,
        pixel_size=pixel_size,
        inverse_contrast=bool(args.inverse_contrast),
    )
    write_output_star(
        output_star=output_star,
        rows=rows_to_run,
        labels=labels,
        output_mrcs_name=output_mrcs,
        defocus_values_angstrom=defocus_values_angstrom,
    )

    print(f"Saved stack: {output_mrcs}", flush=True)
    print(f"Saved STAR:  {output_star}", flush=True)

    if not args.keep_tmp:
        for image_path in image_paths_in_order:
            try:
                Path(image_path).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            # Leave the directory if it contains unrelated/restart files.
            pass


if __name__ == "__main__":
    main()
