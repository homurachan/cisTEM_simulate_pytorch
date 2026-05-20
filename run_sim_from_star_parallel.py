#!/usr/bin/env python3
"""
Parallel wrapper for running a multislice simulation program from RELION STAR angles.

Workflow:
    input.star
    -> read _rlnAngleRot / _rlnAngleTilt / _rlnAnglePsi
    -> launch multiple simulator subprocesses in parallel
    -> collect single-image temporary .mrc files
    -> merge into one .mrcs stack
    -> write output .star with corrected _rlnImageName entries
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mrcfile
except Exception:
    mrcfile = None


def parse_relion_star_angles(star_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Minimal RELION STAR parser for the data_particles loop."""
    with open(star_path, "r", errors="replace") as f:
        lines = f.readlines()

    rows: List[Dict[str, str]] = []
    labels: List[str] = []
    in_particles = False
    in_loop = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("data_particles"):
            in_particles = True
            in_loop = False
            labels = []
            continue

        if not in_particles:
            continue

        if stripped == "":
            if in_loop and rows:
                break
            continue

        if stripped.startswith("data_") and not stripped.startswith("data_particles"):
            if rows:
                break
            continue

        if stripped.startswith("loop_"):
            in_loop = True
            labels = []
            continue

        if in_loop and stripped.startswith("_rln"):
            labels.append(stripped.split()[0])
            continue

        if in_loop and labels:
            parts = stripped.split()
            if len(parts) < len(labels):
                continue
            rows.append({label: value for label, value in zip(labels, parts)})

    required = ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]
    missing = [x for x in required if x not in labels]
    if missing:
        raise ValueError(f"Missing required STAR labels: {missing}")
    if not rows:
        raise ValueError(f"No particle rows found in {star_path}")
    return rows, labels
def get_arg_value_from_extra_args(
    extra_args: List[str],
    name: str,
    default=None,
):
    """
    Read a value from simulator extra_args.

    Supports:
        --box 256
        --box=256
        --pixel-size 1.04
        --pixel-size=1.04
    """
    prefix = name + "="

    for i, x in enumerate(extra_args):
        if x == name and i + 1 < len(extra_args):
            return extra_args[i + 1]
        if x.startswith(prefix):
            return x[len(prefix):]

    return default


def infer_box_and_pixel_size_from_extra_args(
    extra_args: List[str],
) -> Tuple[int, float]:
    """
    Infer --box and --pixel-size from simulator arguments.

    These are needed because STAR defocus should include:
        box / 2 * pixel_size

    Units:
        box: pixels
        pixel_size: Angstrom / pixel
    """
    box = get_arg_value_from_extra_args(extra_args, "--box", None)
    pixel_size = get_arg_value_from_extra_args(extra_args, "--pixel-size", None)

    if box is None:
        raise ValueError(
            "Cannot infer --box from simulator arguments. "
            "Please pass --box after the wrapper '--'."
        )

    if pixel_size is None:
        raise ValueError(
            "Cannot infer --pixel-size from simulator arguments. "
            "Please pass --pixel-size after the wrapper '--'."
        )

    return int(box), float(pixel_size)
def generate_random_defocus_values_angstrom(
    n_particles: int,
    defocus_min_um: float,
    defocus_max_um: float,
    box: int,
    pixel_size: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate random defocus values for particles.

    Input defocus range is in microns.
    Output defocus values are in Angstrom.

    Final value:
        defocus_A = random_uniform(defocus_min_um, defocus_max_um) * 10000
                    + box / 2 * pixel_size

    The extra box/2*pixel_size term follows your requested convention.
    """
    if defocus_max_um < defocus_min_um:
        raise ValueError("--defocus-max-um must be >= --defocus-min-um")

    rng = np.random.default_rng(seed)

    defocus_um = rng.uniform(
        float(defocus_min_um),
        float(defocus_max_um),
        size=int(n_particles),
    ).astype(np.float64)

    defocus_a = defocus_um * 10000.0 + 0.5 * float(box) * float(pixel_size)

    return defocus_a.astype(np.float64)
def read_mrc_2d(path: str) -> np.ndarray:
    if mrcfile is None:
        raise ImportError("mrcfile is required. Install with: pip install mrcfile")
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError(f"{path} is 3D with shape {data.shape}; expected one image.")
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"{path} has shape {data.shape}; expected 2D image.")
    return data.astype(np.float32, copy=False)


def write_mrcs_stack(output_mrcs: str, image_paths_in_order: List[str], pixel_size: Optional[float] = None,inverse_contrast = False) -> None:
    if mrcfile is None:
        raise ImportError("mrcfile is required. Install with: pip install mrcfile")
    stack = [read_mrc_2d(p) for p in image_paths_in_order]
    arr = np.stack(stack, axis=0).astype(np.float32)
    if inverse_contrast:
        arr = arr*-1.0
    with mrcfile.new(output_mrcs, overwrite=True) as mrc:
        mrc.set_data(arr)
        if pixel_size is not None:
            mrc.voxel_size = float(pixel_size)


def write_output_star(
    output_star: str,
    rows: List[Dict[str, str]],
    labels: List[str],
    output_mrcs_name: str,
    defocus_values_angstrom: Optional[np.ndarray] = None,
) -> None:
    """
    Write RELION-style STAR preserving particle labels.

    _rlnImageName is replaced by:
        1@output.mrcs
        2@output.mrcs
        ...

    If defocus_values_angstrom is provided, write:
        _rlnDefocusU
        _rlnDefocusV
        _rlnDefocusAngle

    RELION defocus unit is Angstrom.
    """
    labels_out = list(labels)

    if "_rlnImageName" not in labels_out:
        labels_out.append("_rlnImageName")

    for label in ["_rlnDefocusU", "_rlnDefocusV", "_rlnDefocusAngle"]:
        if label not in labels_out:
            labels_out.append(label)

    if defocus_values_angstrom is not None:
        if len(defocus_values_angstrom) != len(rows):
            raise ValueError(
                "Length of defocus_values_angstrom does not match number of rows."
            )

    with open(output_star, "w") as f:
        f.write("# relion simulated\n\n")
        f.write("data_particles\n\n")
        f.write("loop_\n")

        for i, label in enumerate(labels_out, start=1):
            f.write(f"{label} #{i}\n")

        mrcs_base = Path(output_mrcs_name).name

        for i, row in enumerate(rows, start=1):
            values = []

            if defocus_values_angstrom is not None:
                defocus_u = float(defocus_values_angstrom[i - 1])
                defocus_v = defocus_u
                defocus_angle = 0.0
            else:
                defocus_u = float(row.get("_rlnDefocusU", 0.0))
                defocus_v = float(row.get("_rlnDefocusV", defocus_u))
                defocus_angle = float(row.get("_rlnDefocusAngle", 0.0))

            for label in labels_out:
                if label == "_rlnImageName":
                    values.append(f"{i}@{mrcs_base}")
                elif label == "_rlnDefocusU":
                    values.append(f"{defocus_u:.6f}")
                elif label == "_rlnDefocusV":
                    values.append(f"{defocus_v:.6f}")
                elif label == "_rlnDefocusAngle":
                    values.append(f"{defocus_angle:.6f}")
                else:
                    values.append(str(row.get(label, "0")))

            f.write("\t".join(values) + "\n")


def _run_one_simulation_worker(
    task: Tuple[
        int,
        str,
        str,
        str,
        str,
        float,
        float,
        float,
        Optional[float],
        str,
        List[str],
        Optional[str],
        bool,
    ],
) -> Tuple[int, str, float]:
    """
    Worker function run inside ProcessPoolExecutor.

    Returns
    -------
    index
        0-based selected particle index.
    output_image
        Output image path.
    elapsed
        Wall-clock seconds.
    """
    (
        index,
        python_exe,
        simulator_script,
        pdb_path,
        output_image,
        rot,
        tilt,
        psi,
        defocus_angstrom,
        
        device,
        extra_args,
        env_cuda_visible_devices,
        dry_run,
    ) = task

    cmd = [
        python_exe,
        simulator_script,
        pdb_path,
        output_image,
        "--rot",
        str(rot),
        "--tilt",
        str(tilt),
        "--psi",
        str(psi),
        "--device",
        device,
    ]

    if defocus_angstrom is not None:
        cmd.extend(
            [
                "--defocus",
                f"{float(defocus_angstrom):.6f}",
            #    "--defocus-v",
            #    f"{float(defocus_angstrom):.6f}",
            #    "--defocus-angle",
            #    "0.0",
            ]
        )

    cmd.extend(extra_args)
    cmd.extend(["--mode", "multislice","--explicit-water","--per-frame","--use-cache-atom","--no-center","--shake-waters"])
#    cmd.extend(["--mode", "projection","--no-center"])
   
    env = os.environ.copy()
    if env_cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = env_cuda_visible_devices

    t0 = time.time()

    if dry_run:
        print("[DRY RUN]", " ".join(cmd), flush=True)
        return index, output_image, 0.0
#    print(cmd)
    subprocess.run(cmd, check=True, env=env)
    elapsed = time.time() - t0

    return index, output_image, elapsed


def parse_gpu_ids(gpu_ids_str: Optional[str]) -> Optional[List[str]]:
    if gpu_ids_str is None or gpu_ids_str.strip() == "":
        return None
    ids = [x.strip() for x in gpu_ids_str.split(",") if x.strip()]
    return ids if ids else None


def build_tasks(
    rows_to_run: List[Dict[str, str]],
    start_global_index: int,
    python_exe: str,
    simulator_script: str,
    pdb_path: str,
    tmp_dir: Path,
    device: str,
    extra_args: List[str],
    gpu_ids: Optional[List[str]],
    dry_run: bool,
    defocus_values_angstrom: Optional[np.ndarray] = None,
) -> List[Tuple]:
    tasks = []

    if defocus_values_angstrom is not None:
        if len(defocus_values_angstrom) != len(rows_to_run):
            raise ValueError(
                "Length of defocus_values_angstrom does not match rows_to_run."
            )

    for local_i, row in enumerate(rows_to_run):
        global_i_0based = start_global_index + local_i
        global_i_1based = global_i_0based + 1

        rot = float(row["_rlnAngleRot"])
        tilt = float(row["_rlnAngleTilt"])
        psi = float(row["_rlnAnglePsi"])

        output_image = str(tmp_dir / f"sim_{global_i_1based:06d}.mrc")

        env_cuda_visible_devices = None
        task_device = device

        if gpu_ids:
            env_cuda_visible_devices = gpu_ids[local_i % len(gpu_ids)]
            if device.startswith("cuda"):
                task_device = "cuda"

        defocus_angstrom = None
        if defocus_values_angstrom is not None:
            defocus_angstrom = float(defocus_values_angstrom[local_i])

        tasks.append(
            (
                local_i,
                python_exe,
                simulator_script,
                pdb_path,
                output_image,
                rot,
                tilt,
                psi,
                defocus_angstrom,
                task_device,
                extra_args,
                env_cuda_visible_devices,
                dry_run,
            )
        )

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel multislice simulations from RELION STAR Euler angles.")
    parser.add_argument("simulator_script", help="Path to simulation script.")
    parser.add_argument("pdb", help="Input PDB file.")
    parser.add_argument("input_star", help="Input RELION STAR file.")
    parser.add_argument("output_mrcs", help="Output simulated stack, e.g. sim.mrcs.")

    parser.add_argument("--output-star", default=None, help="Output STAR file. Default: output_mrcs with .star suffix.")
    parser.add_argument("--tmp-dir", default="sim_tmp_images", help="Temporary directory for single-image MRC files.")
    parser.add_argument("--device", default="cuda", help="Device passed to simulator, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--jobs", type=int, default=2, help="Number of simulations to run in parallel.")
    parser.add_argument("--gpu-ids",default=None,help=("Optional comma-separated GPU IDs for multi-GPU runs, e.g. 0,1,2,3. ""Each subprocess gets one ID through CUDA_VISIBLE_DEVICES."),)
    parser.add_argument("--python", default=sys.executable, help="Python executable used to call simulator.")
#   parser.add_argument("--pixel-size", type=float, default=1.04, help="Optional pixel size written into output .mrcs header.")
#   parser.add_argument("--box",type=int,default=256,help="Boxsize of the simulated images.",)

    parser.add_argument("--start", type=int, default=0, help="Start particle index, 0-based.")
    parser.add_argument("--max-particles", type=int, default=None, help="Only simulate this many particles.")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary single-image MRC files.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse temporary images that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print simulator commands without running them.")
    parser.add_argument("--defocus-min-um",type=float,default=None,help="Minimum random defocus in microns.",)
    parser.add_argument("--defocus-max-um",type=float,default=None,help="Maximum random defocus in microns.",)
    parser.add_argument("--defocus-seed",type=int,default=None,help="Random seed for defocus assignment.",)
    parser.add_argument("--inverse-contrast",action="store_true", help="Inverse the simulation contrast. By default the protein is black.")
    args, unknown = parser.parse_known_args()
    print("Please remember to add --box and --pixel-size !!!!!")
    # Arguments after "--" are passed to the simulator.
    if unknown and unknown[0] == "--":
        extra_args = unknown[1:]
    else:
        extra_args = unknown

    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if gpu_ids:
        print(f"Using GPU IDs by subprocess: {gpu_ids}", flush=True)
        print("Each subprocess sees its assigned GPU as cuda:0 via CUDA_VISIBLE_DEVICES.", flush=True)

    rows, labels = parse_relion_star_angles(args.input_star)

    start = int(args.start)
    end = len(rows)
    if args.max_particles is not None:
        end = min(end, start + int(args.max_particles))
    rows_to_run = rows[start:end]
    if not rows_to_run:
        raise ValueError("No particles selected.")
    defocus_values_angstrom = None

    if args.defocus_min_um is not None or args.defocus_max_um is not None:
        if args.defocus_min_um is None or args.defocus_max_um is None:
            raise ValueError(
                "Please provide both --defocus-min-um and --defocus-max-um."
            )

    #    box_for_defocus, pixel_size_for_defocus = args.box,args.pixel_size
        box_for_defocus, pixel_size_for_defocus = infer_box_and_pixel_size_from_extra_args(extra_args)
        
        defocus_values_angstrom = generate_random_defocus_values_angstrom(
            n_particles=len(rows_to_run),
            defocus_min_um=float(args.defocus_min_um),
            defocus_max_um=float(args.defocus_max_um),
            box=box_for_defocus,
            pixel_size=pixel_size_for_defocus,
            seed=args.defocus_seed,
        )

    print(
        "Random defocus enabled: "
        f"{args.defocus_min_um:.4f} to {args.defocus_max_um:.4f} um, "
        f"box={box_for_defocus}, pixel_size={pixel_size_for_defocus}, "
        f"added offset={0.5 * box_for_defocus * pixel_size_for_defocus:.3f} A",
        flush=True,
    )
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(
        rows_to_run=rows_to_run,
        start_global_index=start,
        python_exe=args.python,
        simulator_script=args.simulator_script,
        pdb_path=args.pdb,
        tmp_dir=tmp_dir,
        device=args.device,
        extra_args=extra_args,
        gpu_ids=gpu_ids,
        dry_run=args.dry_run,
        defocus_values_angstrom=defocus_values_angstrom,
    )

    completed: Dict[int, str] = {}
    if args.skip_existing and not args.dry_run:
        remaining = []
        for task in tasks:
            local_i = task[0]
            output_image = task[4]
            if Path(output_image).exists() and Path(output_image).stat().st_size > 0:
                completed[local_i] = output_image
            else:
                remaining.append(task)
        tasks_to_submit = remaining
        print(f"skip-existing: {len(completed)} existing, {len(tasks_to_submit)} remaining.", flush=True)
    else:
        tasks_to_submit = tasks

    print(
        f"Selected particles: {len(rows_to_run)}; jobs={args.jobs}; tasks to run={len(tasks_to_submit)}",
        flush=True,
    )
    
    if tasks_to_submit:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_run_one_simulation_worker, task) for task in tasks_to_submit]
            n_done = 0
            for future in as_completed(futures):
                local_i, output_image, elapsed = future.result()
                completed[local_i] = output_image
                n_done += 1
                print(
                    f"Finished {n_done}/{len(tasks_to_submit)} "
                    f"(selected row {local_i}, {elapsed:.2f} s)",
                    flush=True,
                )
    
    if args.dry_run:
        print("Dry run finished. No .mrcs/.star written.")
        return

    image_paths_in_order = []
    for local_i in range(len(rows_to_run)):
        if local_i not in completed:
            raise RuntimeError(f"Missing completed output for selected row {local_i}")
        image_paths_in_order.append(completed[local_i])

    print("Merging temporary images into stack...", flush=True)
    write_mrcs_stack(args.output_mrcs, image_paths_in_order, pixel_size=pixel_size_for_defocus,inverse_contrast=args.inverse_contrast)

    output_star = args.output_star or str(Path(args.output_mrcs).with_suffix(".star"))
    write_output_star(output_star, rows_to_run, labels, args.output_mrcs,defocus_values_angstrom=defocus_values_angstrom,)

    print(f"Saved stack: {args.output_mrcs}", flush=True)
    print(f"Saved STAR:  {output_star}", flush=True)

    if not args.keep_tmp:
        for p in image_paths_in_order:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
