#!/usr/bin/env python
"""Verify DA3 outputs from run_da3_on_tapvid3d.py."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import trimesh


def verify_subset(subset_dir: Path, input_dir: Path) -> dict:
    out_clips = [p for p in subset_dir.iterdir() if p.is_dir() and (p / "done.flag").exists()]
    in_clips = sorted(input_dir.glob("*.npz")) if input_dir.is_dir() else []
    summary = {"input_clips": len(in_clips), "out_clips": len(out_clips), "sample": None}
    if not out_clips:
        return summary

    sample = out_clips[0]
    sample_info = {"clip": sample.name}
    depth_npz = sample / "depth.npz"
    if depth_npz.exists():
        d = np.load(depth_npz)
        sample_info["shapes"] = {k: tuple(d[k].shape) for k in d.files}
        sample_info["dtypes"] = {k: str(d[k].dtype) for k in d.files}

    glb = sample / "scene.glb"
    if glb.exists():
        try:
            scene = trimesh.load(glb)
            geom_count = len(scene.geometry) if hasattr(scene, "geometry") else 0
            point_count = 0
            for g in scene.geometry.values() if hasattr(scene, "geometry") else []:
                if hasattr(g, "vertices"):
                    point_count += len(g.vertices)
            sample_info["glb"] = {"geometries": geom_count, "vertices": point_count}
        except Exception as exc:
            sample_info["glb"] = f"load_failed: {exc}"

    summary["sample"] = sample_info
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tapvid3d-root", default=os.path.expanduser("~/data/tapvid3d"))
    p.add_argument("--output-root", default=os.path.expanduser("~/data/tapvid3d_da3_out"))
    p.add_argument("--subsets", nargs="+", default=["drivetrack", "pstudio", "adt"])
    args = p.parse_args()

    tapvid_root = Path(args.tapvid3d_root)
    out_root = Path(args.output_root)

    print(f"== output root: {out_root}")
    for sub in args.subsets:
        sub_out = out_root / sub
        sub_in = tapvid_root / sub
        if not sub_out.is_dir():
            print(f"\n[{sub}] no output dir")
            continue
        info = verify_subset(sub_out, sub_in)
        print(f"\n[{sub}] input={info['input_clips']} output={info['out_clips']}")
        s = info["sample"]
        if s:
            print(f"  sample clip: {s['clip']}")
            for k, v in s.get("shapes", {}).items():
                print(f"    {k}: shape={v} dtype={s['dtypes'][k]}")
            if "glb" in s:
                print(f"    glb: {s['glb']}")

    oom_log = out_root / "oom.log"
    if oom_log.exists():
        lines = oom_log.read_text().strip().splitlines()
        print(f"\nfailures logged: {len(lines)} (see {oom_log})")
        for line in lines[:5]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
