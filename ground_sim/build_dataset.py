#!/data/public/renhaoye/miniforge-pypy3/envs/ai4galaxy/bin/python
"""Batch dataset builder for the ground-simulation VAE experiment.

Physics parameters are the ones frozen in ground_simulation_tuning.ipynb
(2026-07-19 review).  Per approved observation, 4 purely random 64x64 cutouts
are taken (user decision: no GT admission conditions; background-only crops
are legitimate samples).

Layout (matches sgm.data.dataset.Dataset with image_folder=BGSUB,
error_folder=INVVAR, error_type=invvar):

    <root>/{train,valid,test}/GROUND_R/BGSUB/<name>.fits    HDU0 = OBS (nmgy)
    <root>/{train,valid,test}/GROUND_R/INVVAR/<name>.fits   HDU0 = 1/SIGMA^2

SIGMA is estimated from the noisy science exposures, fitted background and
detector read noise.  GT is never used to construct INVVAR.

Contract enforcement at the data level:
  * train/valid BGSUB files carry ONLY the OBS plane -- GT is physically
    absent, so it cannot enter any training/validation loss even by accident
    (the loader's ``gt`` field falls back to the image itself).
  * test BGSUB files additionally carry HDU1 "GT" and HDU2 "COVERAGE";
    both are used exclusively by the final SNR-vs-|rec-GT| evaluation, with
    COVERAGE excluding zero-padded pixels of sub-500px sources.

Splits are by *subhalo* (all orientations of one galaxy stay together) to
prevent leakage: 70/15/15.

Runs are non-reproducible by default: the master seed comes from OS entropy
and is recorded in the manifest for post-hoc debugging only.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import secrets
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from astropy.io import fits

from ground_sim.forward_model import (
    Params, SKIRT_PIXEL_SCALE, bin_planes, load_r_band, simulate,
)

# ---- frozen physics (source of truth: notebook cell ea0321ee) -------------
FROZEN = Params(
    z_source=0.01,
    z_target=0.04,
    n_exposures=3,
    exposure_time=60.0,
    sky_mag=22.0,
    zeropoint=24.6,
    seeing_fwhm=1.5,
    read_noise=3.0,
    bkg_box_size=8,
    apply_angular_scaling=True,
    pixel_scale=SKIRT_PIXEL_SCALE,
)
BIN = 1
PAD_SIZE = 500
CROP_SIZE = 64
N_CROPS = 4

RAW_GLOB = "/data1/public/csst_mock/dataset/raw/*.fits"
NAME_RE = re.compile(r"^(?P<snap>\d+)_Subhalo_(?P<subhalo>\d+)_O(?P<orient>\d+)\.fits$")

SPLIT_FRACTIONS = {"train": 0.70, "valid": 0.15, "test": 0.15}


def pad_to_square(rate, target=PAD_SIZE):
    ny, nx = rate.shape
    if ny >= target and nx >= target:
        return rate, np.ones_like(rate)
    out = np.zeros((max(ny, target), max(nx, target)), dtype=rate.dtype)
    cov = np.zeros_like(out)
    oy, ox = (out.shape[0] - ny) // 2, (out.shape[1] - nx) // 2
    out[oy:oy + ny, ox:ox + nx] = rate
    cov[oy:oy + ny, ox:ox + nx] = 1.0
    return out, cov


def split_by_subhalo(paths, rng):
    """Assign each subhalo (with all its orientations) to one split."""
    subhalos = {}
    for p in paths:
        m = NAME_RE.match(p.name)
        if not m:
            continue
        subhalos.setdefault(m.group("subhalo"), []).append(p)
    ids = sorted(subhalos)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * SPLIT_FRACTIONS["train"]))
    n_valid = int(round(n * SPLIT_FRACTIONS["valid"]))
    assignment = {}
    for i, sid in enumerate(ids):
        split = "train" if i < n_train else ("valid" if i < n_train + n_valid else "test")
        for p in subhalos[sid]:
            assignment[str(p)] = split
    return assignment


def process_one(task):
    """Simulate one observation and write its crops.  One task == one core."""
    path_str, split, out_root, obs_seed = task
    try:
        rng = np.random.default_rng(obs_seed)
        rate = load_r_band(path_str)
        rate_p, cov_native = pad_to_square(rate)
        obs, gt, sigma, info, cov = simulate(rate_p, FROZEN, rng, coverage=cov_native)
        obs, gt, sigma, cov = bin_planes(obs, gt, sigma, BIN, coverage=cov)

        side = obs.shape[0]
        hi = side - CROP_SIZE
        if hi < 0:
            return (path_str, split, "canvas_too_small", [])
        ys = rng.integers(0, hi + 1, size=N_CROPS)
        xs = rng.integers(0, hi + 1, size=N_CROPS)

        stem = Path(path_str).stem
        bgsub_dir = Path(out_root) / split / "GROUND_R" / "BGSUB"
        invvar_dir = Path(out_root) / split / "GROUND_R" / "INVVAR"
        rows = []
        for j, (y0, x0) in enumerate(zip(ys.tolist(), xs.tolist())):
            name = f"{stem}_c{j}.fits"
            o = np.asarray(obs[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE], dtype=np.float32)
            s = np.asarray(sigma[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE], dtype=np.float64)
            iv = np.asarray(1.0 / np.square(s), dtype=np.float32)

            hdus = [fits.PrimaryHDU(o)]
            hdus[0].header["UNIT"] = "nanomaggy"
            hdus[0].header["SRCFILE"] = Path(path_str).name
            hdus[0].header["CROPY"] = y0
            hdus[0].header["CROPX"] = x0
            hdus[0].header["PADDED"] = bool(cov_native.mean() < 1.0)
            hdus[0].header["SIGMETH"] = "OBS+BKG"
            if split == "test":
                g = np.asarray(gt[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE], dtype=np.float32)
                c = np.asarray(cov[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE], dtype=np.float32)
                hdus.append(fits.ImageHDU(g, name="GT"))
                hdus.append(fits.ImageHDU(c, name="COVERAGE"))
            fits.HDUList(hdus).writeto(bgsub_dir / name, overwrite=True)
            iv_hdu = fits.PrimaryHDU(iv)
            iv_hdu.header["BUNIT"] = "1/nanomaggy^2"
            iv_hdu.header["SIGMETH"] = "OBS+BKG"
            iv_hdu.writeto(invvar_dir / name, overwrite=True)
            rows.append((name, split, Path(path_str).name, y0, x0,
                         float(cov[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE].mean()),
                         bool(cov_native.mean() < 1.0), obs_seed))
        return (path_str, split, "ok", rows)
    except Exception as exc:  # noqa: BLE001 -- worker must not kill the pool
        return (path_str, split, f"error: {exc!r}", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        default="/data1/public/renhaoye/dataset/ground_sim_v2_obsivar",
    )
    ap.add_argument("--workers", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N sources")
    ap.add_argument("--seed", type=int, default=None,
                    help="master seed; omitted => entropy (non-reproducible)")
    args = ap.parse_args()

    master_seed = args.seed if args.seed is not None else secrets.randbits(63)
    seed_seq = np.random.SeedSequence(master_seed)

    paths = sorted(Path("/data1/public/csst_mock/dataset/raw").glob("*.fits"))
    if args.limit:
        paths = paths[:args.limit]
    split_rng = np.random.default_rng(seed_seq.spawn(1)[0])
    assignment = split_by_subhalo(paths, split_rng)

    out_root = Path(args.out_root)
    for split in SPLIT_FRACTIONS:
        (out_root / split / "GROUND_R" / "BGSUB").mkdir(parents=True, exist_ok=True)
        (out_root / split / "GROUND_R" / "INVVAR").mkdir(parents=True, exist_ok=True)

    obs_seeds = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(len(paths))]
    tasks = [(str(p), assignment.get(str(p), "train"), str(out_root), obs_seeds[i])
             for i, p in enumerate(paths) if str(p) in assignment]

    print(f"master_seed={master_seed}  sources={len(tasks)}  workers={args.workers}")
    n_ok = n_fail = 0
    manifest_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(process_one, tasks, chunksize=4):
            path_str, split, status, rows = res
            if status == "ok":
                n_ok += 1
                manifest_rows.extend(rows)
            else:
                n_fail += 1
                print(f"FAIL {Path(path_str).name}: {status}")
            if (n_ok + n_fail) % 1000 == 0:
                print(f"  progress {n_ok + n_fail}/{len(tasks)}")

    manifest = out_root / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["crop_file", "split", "source_file", "crop_y", "crop_x",
                    "coverage_mean", "was_padded", "obs_seed"])
        w.writerows(manifest_rows)
    with (out_root / "GENERATION_INFO.txt").open("w") as fh:
        fh.write(f"master_seed={master_seed}\nfrozen_params={FROZEN}\n"
                 f"BIN={BIN} PAD_SIZE={PAD_SIZE} CROP_SIZE={CROP_SIZE} N_CROPS={N_CROPS}\n"
                 "sigma_method=observed_source_plus_fitted_background; GT_not_used\n"
                 f"crops=pure-random (no GT admission)\nsplit=by-subhalo 70/15/15\n")
    print(f"done: ok={n_ok} fail={n_fail} crops={len(manifest_rows)} -> {out_root}")


if __name__ == "__main__":
    main()
