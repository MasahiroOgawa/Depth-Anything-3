#!/usr/bin/env python
"""Export a clip's 4D reconstruction to a Rerun .rrd file.

The .rrd opens in the Rerun viewer on any machine — rotate the 3D scene with
the mouse and scrub the time slider to step frames. Designed so you can run
this on the GPU box, rsync the small .rrd file to your laptop, and inspect
the 4D scene interactively without X forwarding or a live network port.

The same logic also supports live streaming: pass --serve to host the data
over Rerun's gRPC port; connect the Rerun viewer to it from your laptop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image

from render_4d_animation import load_rgb_frames, unproject_frame

LOG = logging.getLogger("export_rerun")


def export_clip(
    clip_dir: Path,
    tapvid_npz: Path | None,
    rrd_out: Path | None,
    serve: bool,
    fps: float,
) -> None:
    arr = np.load(clip_dir / "depth.npz")
    depth_all = arr["depth"]
    extr_all = arr["extrinsics"]
    intr_all = arr["intrinsics"]
    conf_all = arr["conf"] if "conf" in arr.files else None
    n_frames = depth_all.shape[0]

    rgb_all = load_rgb_frames(tapvid_npz) if tapvid_npz else None
    if rgb_all is not None and rgb_all.shape[0] != n_frames:
        LOG.warning("rgb count %d != depth count %d; ignoring", rgb_all.shape[0], n_frames)
        rgb_all = None
    LOG.info("clip %s: %d frames, rgb=%s", clip_dir.name, n_frames, rgb_all is not None)

    app_id = f"da3_4d_{clip_dir.parent.name}_{clip_dir.name}"
    rr.init(app_id, spawn=False)
    if serve:
        rr.serve_grpc()
        LOG.info("Rerun gRPC server running. From your laptop, run:")
        LOG.info("  rerun --connect rerun+http://localhost:9876/proxy")
    elif rrd_out:
        rr.save(rrd_out)

    rr.log("/world", rr.ViewCoordinates.RDF, static=True)

    for t in range(n_frames):
        rr.set_time("frame", sequence=t)
        rr.set_time("time", duration=t / fps)

        pts, cols = unproject_frame(
            depth_all[t],
            intr_all[t],
            extr_all[t],
            conf_all[t] if conf_all is not None else None,
            rgb_all[t] if rgb_all is not None else None,
        )
        if pts.size:
            rr.log(
                "/world/points",
                rr.Points3D(pts, colors=(cols * 255).astype(np.uint8), radii=0.005),
            )
        # Camera frustum at this frame so you can see the trajectory too
        R = extr_all[t, :3, :3]
        tvec = extr_all[t, :3, 3]
        cam_center = -R.T @ tvec
        cam_rot_c2w = R.T  # camera -> world rotation
        rr.log(
            "/world/cam",
            rr.Transform3D(translation=cam_center, mat3x3=cam_rot_c2w),
        )
        rr.log(
            "/world/cam/image",
            rr.Pinhole(
                image_from_camera=intr_all[t],
                resolution=[depth_all[t].shape[1], depth_all[t].shape[0]],
            ),
        )
        if rgb_all is not None:
            ih, iw = depth_all[t].shape
            rgb_resized = (
                rgb_all[t]
                if rgb_all[t].shape[:2] == (ih, iw)
                else np.asarray(Image.fromarray(rgb_all[t]).resize((iw, ih), Image.BILINEAR))
            )
            rr.log("/world/cam/image", rr.Image(rgb_resized))

        if t == 0 or (t + 1) % 25 == 0:
            LOG.info("  frame %d/%d", t + 1, n_frames)

    if serve:
        LOG.info("Streaming. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
    elif rrd_out:
        LOG.info("wrote %s", rrd_out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clip-dir", required=True, type=Path, help="A clip dir under tapvid3d_da3_out/"
    )
    p.add_argument(
        "--rrd", type=Path, default=None, help="Output .rrd path (default: <clip-dir>/4d.rrd)"
    )
    p.add_argument("--serve", action="store_true", help="Stream over gRPC instead of writing .rrd")
    p.add_argument("--tapvid3d-root", type=Path, default=Path.home() / "data/tapvid3d")
    p.add_argument("--no-rgb", action="store_true")
    p.add_argument("--fps", type=float, default=24.0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    clip_dir = args.clip_dir.resolve()
    if not (clip_dir / "depth.npz").exists():
        LOG.error("no depth.npz under %s", clip_dir)
        return 2
    subset = clip_dir.parent.name
    clip_id = clip_dir.name
    tapvid_npz = None if args.no_rgb else (args.tapvid3d_root / subset / f"{clip_id}.npz")
    rrd_out = None if args.serve else (args.rrd or clip_dir / "4d.rrd")

    t0 = time.time()
    export_clip(clip_dir, tapvid_npz, rrd_out, args.serve, args.fps)
    LOG.info("done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
