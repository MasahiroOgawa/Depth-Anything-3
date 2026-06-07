#!/usr/bin/env python
"""Run Depth-Anything-3 over TAPVID3D minival clips for 4D reconstruction.

Each TAPVID3D .npz holds frames in `images_jpeg_bytes`. For every clip we:
  1. Decode JPEGs to a temp dir.
  2. Run DA3 once on the full sorted frame list (single model load is amortized
     across all clips by the outer loop).
  3. Export depth + poses + colored point cloud to <output_root>/<subset>/<clip>/.
  4. Delete the temp frame dir to keep disk flat.

OOM and other per-clip failures are logged but never abort the run.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3

LOG = logging.getLogger("da3_tapvid3d")

DEFAULT_MODEL = "depth-anything/DA3-LARGE-1.1"
DEFAULT_GS_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE"
DEFAULT_EXPORT_FORMAT = "glb-mini_npz"
DEFAULT_GS_EXPORT_FORMAT = "glb-mini_npz-gs_ply"


def decode_clip_frames(npz_path: Path, frames_dir: Path) -> list[str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    # allow_pickle: TAPVID3D npz stores variable-length jpeg byte arrays as
    # dtype=object. Trusted source (official TAPVID3D buckets).
    data = np.load(npz_path, allow_pickle=True)
    jpeg_bytes = data["images_jpeg_bytes"]
    paths: list[str] = []
    for i, byts in enumerate(jpeg_bytes):
        out = frames_dir / f"frame_{i:05d}.jpg"
        out.write_bytes(bytes(byts))
        paths.append(str(out))
    return paths


def process_clip(
    model: DepthAnything3,
    npz_path: Path,
    out_root: Path,
    tmp_root: Path,
    export_format: str,
    infer_gs: bool,
    process_res: int,
) -> tuple[bool, str]:
    clip_id = npz_path.stem
    subset = npz_path.parent.name
    out_dir = out_root / subset / clip_id
    if (out_dir / "done.flag").exists():
        return True, "skip-already-done"

    frames_dir = tmp_root / subset / clip_id
    try:
        frame_paths = decode_clip_frames(npz_path, frames_dir)
        if not frame_paths:
            return False, "no-frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred = model.inference(
            frame_paths,
            export_dir=str(out_dir),
            export_format=export_format,
            infer_gs=infer_gs,
            process_res=process_res,
        )
        np.savez_compressed(
            out_dir / "depth.npz",
            depth=pred.depth,
            extrinsics=pred.extrinsics,
            intrinsics=pred.intrinsics,
            conf=pred.conf,
        )
        (out_dir / "done.flag").write_text(f"frames={len(frame_paths)}\n")
        return True, f"ok-{len(frame_paths)}"
    except torch.cuda.OutOfMemoryError as exc:
        return False, f"oom: {exc!s}"
    except Exception as exc:
        LOG.exception("clip %s failed", clip_id)
        return False, f"err: {exc!s}"
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tapvid3d-root", default=os.path.expanduser("~/data/tapvid3d"))
    p.add_argument("--output-root", default=os.path.expanduser("~/data/tapvid3d_da3_out"))
    p.add_argument("--tmp-root", default=os.path.expanduser("~/data/tapvid3d_tmp"))
    p.add_argument("--subsets", nargs="+", default=["drivetrack", "pstudio", "adt"])
    p.add_argument("--model", default=None, help="Default: LARGE, or GIANT if --infer-gs is set")
    p.add_argument("--export-format", default=None)
    p.add_argument("--limit", type=int, default=None, help="Cap clips per subset (debug)")
    p.add_argument(
        "--infer-gs",
        action="store_true",
        help="Predict 3D Gaussians; switches to GIANT model and adds gs_ply to export",
    )
    p.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Model processing resolution; lower for OOM-prone clips (e.g. 392)",
    )
    args = p.parse_args()
    if args.model is None:
        args.model = DEFAULT_GS_MODEL if args.infer_gs else DEFAULT_MODEL
    if args.export_format is None:
        args.export_format = DEFAULT_GS_EXPORT_FORMAT if args.infer_gs else DEFAULT_EXPORT_FORMAT

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tapvid_root = Path(args.tapvid3d_root)
    out_root = Path(args.output_root)
    tmp_root = Path(args.tmp_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    oom_log = out_root / "oom.log"

    LOG.info("Loading model %s", args.model)
    t0 = time.time()
    model = DepthAnything3.from_pretrained(args.model).to("cuda")
    model.eval()
    LOG.info("Model loaded in %.1fs", time.time() - t0)

    clips: list[Path] = []
    for sub in args.subsets:
        sub_dir = tapvid_root / sub
        if not sub_dir.is_dir():
            LOG.warning("subset dir missing: %s", sub_dir)
            continue
        sub_clips = sorted(sub_dir.glob("*.npz"))
        if args.limit:
            sub_clips = sub_clips[: args.limit]
        clips.extend(sub_clips)
        LOG.info("queued %d clips from %s", len(sub_clips), sub)

    if not clips:
        LOG.error("no clips found under %s", tapvid_root)
        return 2

    n_ok = n_fail = 0
    with torch.inference_mode():
        for i, clip in enumerate(clips, 1):
            t = time.time()
            ok, msg = process_clip(
                model,
                clip,
                out_root,
                tmp_root,
                args.export_format,
                args.infer_gs,
                args.process_res,
            )
            dt = time.time() - t
            tag = "OK " if ok else "ERR"
            LOG.info("[%d/%d] %s %s (%.1fs) %s", i, len(clips), tag, clip.name, dt, msg)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                with oom_log.open("a") as f:
                    f.write(f"{clip}\t{msg}\n")

    LOG.info("done: %d ok, %d failed, log=%s", n_ok, n_fail, oom_log)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
