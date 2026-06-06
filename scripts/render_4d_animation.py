#!/usr/bin/env python
"""Render a clip's 4D reconstruction as an MP4.

Loads <clip-dir>/depth.npz and, for every frame, unprojects the depth map into
world-space points using the matching per-frame intrinsics/extrinsics. Each
frame is rendered from a single fixed camera (computed once from the aggregate
scene bounding box) so the only thing changing in the MP4 is the time-varying
3D scene itself — that's the "4D" part.

If the matching TAPVID3D source npz is available it is used to color points
with the original RGB; otherwise points are colored by depth.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image

LOG = logging.getLogger("render_4d")

CONF_PERCENTILE = 40.0
MAX_POINTS_PER_FRAME = 80_000


def load_rgb_frames(tapvid_npz: Path) -> np.ndarray | None:
    if not tapvid_npz.exists():
        return None
    # allow_pickle: TAPVID3D npz stores variable-length jpeg byte arrays as
    # dtype=object. Trusted source (official TAPVID3D buckets).
    data = np.load(tapvid_npz, allow_pickle=True)
    if "images_jpeg_bytes" not in data.files:
        return None
    frames = [
        np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB"))
        for b in data["images_jpeg_bytes"]
    ]
    return np.stack(frames)


def unproject_frame(
    depth: np.ndarray,
    intr: np.ndarray,
    extr: np.ndarray,
    conf: np.ndarray | None,
    rgb: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]

    mask = depth > 0
    if conf is not None:
        mask &= conf >= np.percentile(conf, CONF_PERCENTILE)

    v_idx, u_idx = np.where(mask)
    if v_idx.size == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)

    d = depth[v_idx, u_idx]
    x_c = (u_idx - cx) / fx * d
    y_c = (v_idx - cy) / fy * d
    z_c = d
    pts_cam = np.stack([x_c, y_c, z_c], axis=1).astype(np.float32)

    # extrinsics are world->camera (3x4 [R | t]); invert to get camera->world.
    R = extr[:3, :3]
    t = extr[:3, 3]
    pts_world = (pts_cam - t) @ R

    if rgb is not None:
        rgb_resized = (
            rgb
            if rgb.shape[:2] == depth.shape
            else np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
        )
        colors = rgb_resized[v_idx, u_idx].astype(np.float32) / 255.0
    else:
        d_min, d_max = float(d.min()), float(d.max())
        norm = (d - d_min) / max(d_max - d_min, 1e-6)
        colors = np.stack([norm, 0.5 * (1 - norm), 1 - norm], axis=1).astype(np.float32)

    if pts_world.shape[0] > MAX_POINTS_PER_FRAME:
        idx = np.random.default_rng(0).choice(
            pts_world.shape[0], MAX_POINTS_PER_FRAME, replace=False
        )
        pts_world = pts_world[idx]
        colors = colors[idx]
    return pts_world, colors


def compute_camera(all_points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.concatenate(all_points, axis=0)
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    center = (lo + hi) / 2
    extent = float(np.linalg.norm(hi - lo))
    # Place camera offset along an oblique direction so we see depth.
    direction = np.array([1.0, -0.5, -1.5])
    direction /= np.linalg.norm(direction)
    eye = center + direction * extent * 1.2
    up = np.array([0.0, -1.0, 0.0])  # image-y is down → world up flips
    return eye, center, up


def render_clip(
    clip_dir: Path,
    output: Path,
    tapvid_npz: Path | None,
    fps: int,
    image_size: tuple[int, int],
) -> None:
    depth_npz = clip_dir / "depth.npz"
    arr = np.load(depth_npz)
    depth_all = arr["depth"]
    extr_all = arr["extrinsics"]
    intr_all = arr["intrinsics"]
    conf_all = arr["conf"] if "conf" in arr.files else None
    n_frames = depth_all.shape[0]

    rgb_all = load_rgb_frames(tapvid_npz) if tapvid_npz else None
    if rgb_all is not None and rgb_all.shape[0] != n_frames:
        LOG.warning(
            "rgb frame count %d != depth count %d; ignoring rgb", rgb_all.shape[0], n_frames
        )
        rgb_all = None
    LOG.info("clip %s: %d frames, rgb=%s", clip_dir.name, n_frames, rgb_all is not None)

    per_frame = []
    all_points = []
    for t in range(n_frames):
        pts, cols = unproject_frame(
            depth_all[t],
            intr_all[t],
            extr_all[t],
            conf_all[t] if conf_all is not None else None,
            rgb_all[t] if rgb_all is not None else None,
        )
        per_frame.append((pts, cols))
        all_points.append(pts)

    eye, center, up = compute_camera(all_points)

    w, h = image_size
    renderer = rendering.OffscreenRenderer(w, h)
    renderer.scene.set_background([0.05, 0.05, 0.08, 1.0])
    eye_l, center_l, up_l = eye.tolist(), center.tolist(), up.tolist()

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 3.0

    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8) as writer:
        for t, (pts, cols) in enumerate(per_frame):
            renderer.scene.clear_geometry()
            if pts.size:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
                renderer.scene.add_geometry(f"f{t}", pcd, mat)
            # Re-set camera each frame: Open3D auto-fits near/far to the
            # current scene contents, so calling this before geometry yields
            # a degenerate clip range and a blank render.
            renderer.setup_camera(60.0, center_l, eye_l, up_l)
            img = renderer.render_to_image()
            writer.append_data(np.asarray(img))
            if t == 0 or (t + 1) % 25 == 0:
                LOG.info("  frame %d/%d", t + 1, n_frames)
    renderer = None
    LOG.info("wrote %s (%d frames @ %d fps)", output, n_frames, fps)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clip-dir", required=True, type=Path, help="A clip dir under tapvid3d_da3_out/"
    )
    p.add_argument(
        "--output", type=Path, default=None, help="Output MP4 path (default: <clip-dir>/4d.mp4)"
    )
    p.add_argument("--tapvid3d-root", type=Path, default=Path.home() / "data/tapvid3d")
    p.add_argument(
        "--no-rgb", action="store_true", help="Color by depth, skip TAPVID3D npz lookup"
    )
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--size", default="960x720", help="WxH render resolution")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    clip_dir = args.clip_dir.resolve()
    if not (clip_dir / "depth.npz").exists():
        LOG.error("no depth.npz under %s", clip_dir)
        return 2

    subset = clip_dir.parent.name
    clip_id = clip_dir.name
    tapvid_npz = None if args.no_rgb else (args.tapvid3d_root / subset / f"{clip_id}.npz")
    output = args.output or (clip_dir / "4d.mp4")
    w, h = (int(x) for x in args.size.lower().split("x"))

    t0 = time.time()
    render_clip(clip_dir, output, tapvid_npz, args.fps, (w, h))
    LOG.info("done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
