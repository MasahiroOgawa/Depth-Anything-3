# Depth Anything 3 → TAPVID3D 4D Reconstruction Pipeline

End-to-end recipe to: install Depth-Anything-3 in an isolated `uv` venv,
download the TAPVID3D `minival` benchmark, run DA3 over every clip's frames,
and view the resulting 4D reconstruction (per-frame depth + camera pose +
fused point cloud + camera trajectory) in a browser.

Validated 2026-06 on:

- GPU: NVIDIA RTX 5060 Ti, 16 GB, Blackwell sm_120, driver 595.71.05
- CUDA: bundled with PyTorch wheels (no system CUDA toolkit required for
  `[app]` install path; `[gs]` extra needs system `nvcc`)
- OS: Linux, bash, `uv 0.11.19`

## 1. Install

### 1.1 Create the DA3 venv

```bash
cd /home/mas/proj/study/Depth-Anything-3
uv venv --python 3.12 .venv
```

> **Why 3.12 even though `pyproject.toml` says `<=3.13`?** `open3d 0.19` ships
> no `cp313` wheel — the effective ceiling is 3.12. Higher Python (3.13/3.14)
> fails dependency resolution.

### 1.2 Install PyTorch (Blackwell-capable build)

```bash
uv pip install --torch-backend=auto torch torchvision
```

`uv` auto-detects the local CUDA stack and resolves the `cu130` wheel
(`torch==2.12.0+cu130`). This is the wheel Blackwell (sm_120) needs.

### 1.3 Install DA3 editable

```bash
uv pip install hatchling hatch-vcs editables
uv pip install -e ".[app]" --no-build-isolation
```

`hatchling`/`hatch-vcs`/`editables` are required at build time but missing
from DA3's `build-system.requires`. `--no-build-isolation` reuses the venv's
torch so CUDA-extension builds (`pycolmap` etc.) can find it.

### 1.4 Smoke test

```bash
.venv/bin/python -c "
import torch
from depth_anything_3.api import DepthAnything3
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('DA3 imports OK')
"
```

Expected: `torch 2.12.0+cu130`, `cuda True`, `(12, 0)` capability,
`DA3 imports OK`. A yellow `WARN Dependency gsplat is required for rendering
3DGS` line is harmless if you're skipping the `[gs]` extra.

### 1.5 Skipped: `[gs]` Gaussian-Splat extra

`uv pip install -e ".[gs]"` fails because `gsplat` compiles CUDA extensions
and needs `nvcc` on PATH. The PyTorch-bundled `nvidia/cu13/` directory has
includes + libs but no `nvcc` binary. To unlock this path:

```bash
sudo apt install nvidia-cuda-toolkit       # or any package providing nvcc
uv pip install -e ".[gs]" --no-build-isolation
```

Skip this section if you only need depth + poses + point clouds.

## 2. Download TAPVID3D minival

TAPVID3D has three subsets — Aria (ADT), DriveTrack, Panoptic Studio. Use a
separate venv: the official tooling pulls TensorFlow + JAX, which conflicts
with the torch-only DA3 venv.

```bash
mkdir -p ~/data/tapvid3d && cd ~/data/tapvid3d
uv venv --python 3.11 .venv-tapvid3d
uv pip install --python .venv-tapvid3d/bin/python \
    "tapnet[tapvid3d_eval,tapvid3d_generation] @ git+https://github.com/google-deepmind/tapnet.git"
```

### 2.1 DriveTrack (fully automatic, ~1 GB)

```bash
.venv-tapvid3d/bin/python -m tapnet.tapvid3d.annotation_generation.generate_drivetrack \
    --output_dir /home/mas/data/tapvid3d/drivetrack --split minival
```

Pulls 50 npz files (with embedded JPEG frames) from a public GCS bucket. No
Waymo TOS gate for `minival`.

### 2.2 Panoptic Studio (auto, ~1.8 GB after extract)

```bash
.venv-tapvid3d/bin/python -m tapnet.tapvid3d.annotation_generation.generate_pstudio \
    --output_dir /home/mas/data/tapvid3d/pstudio --split minival
```

The script auto-downloads a 589 MB `data.zip` from RWTH Aachen's
`Dynamic3DGaussians` mirror, extracts the source videos, and merges them
with the per-clip annotations into 50 `.npz` files.

### 2.3 Aria Digital Twin (gated — manual)

Skipped in the validated run. Requires:

1. Accept Meta's Aria Digital Twin license.
2. Use Meta's `aria_dataset_downloader` to fetch raw ADT scenes locally.
3. Then run:
   ```bash
   .venv-tapvid3d/bin/python -m tapnet.tapvid3d.annotation_generation.generate_adt \
       --output_dir /home/mas/data/tapvid3d/adt --split minival \
       --adt_base_path /path/to/downloaded/adt
   ```

## 3. Run DA3 over every clip

A driver script handles JPEG extraction → inference → export → cleanup, with
per-clip OOM isolation:

```bash
cd /home/mas/proj/study/Depth-Anything-3
.venv/bin/python scripts/run_da3_on_tapvid3d.py \
    --subsets drivetrack pstudio adt
```

The driver:

1. Loads the model once (amortized across all clips).
2. For each `~/data/tapvid3d/{subset}/*.npz`:
   - Decodes `images_jpeg_bytes` to `~/data/tapvid3d_tmp/{subset}/{clip}/frame_*.jpg`.
   - Calls `model.inference(...)` with `export_format="glb-mini_npz"`.
   - Writes `~/data/tapvid3d_da3_out/{subset}/{clip}/{depth.npz, scene.glb, scene.jpg, depth_vis/, exports/, done.flag}`.
   - Deletes the tmp JPEG dir.
3. Clips that OOM are logged to `~/data/tapvid3d_da3_out/oom.log` and the
   run continues. Re-running the driver skips clips with a `done.flag`.

Default model: `depth-anything/DA3-LARGE-1.1` (0.35B params, fits in 16 GB
for clips up to ~130 frames at 504 px). To switch:

```bash
.venv/bin/python scripts/run_da3_on_tapvid3d.py \
    --model depth-anything/DA3NESTED-GIANT-LARGE
```

Note: GIANT only meaningfully helps if you've installed `[gs]` (Gaussian
Splats); otherwise LARGE is the better quality/throughput trade.

## 4. Verify outputs

```bash
.venv/bin/python scripts/verify_da3_outputs.py
```

Reports per-subset coverage, sample-clip array shapes, and `scene.glb` vertex
counts. Healthy output looks like:

```
[drivetrack] input=50 output=49
  sample clip: tapvid3d_5459113827443493510_...
    depth: shape=(85, 336, 504) dtype=float32
    extrinsics: shape=(85, 3, 4) dtype=float32
    intrinsics: shape=(85, 3, 3) dtype=float32
    conf: shape=(85, 336, 504) dtype=float32
    glb: {'geometries': 86, 'vertices': 1001360}
```

## 5. View the 4D reconstructions

### 5.1 Local machine

```bash
.venv/bin/python -m depth_anything_3.cli gallery \
    --gallery-dir /home/mas/data/tapvid3d_da3_out \
    --host 127.0.0.1 --port 8007 --open-browser
```

Opens a browser at http://127.0.0.1:8007 with a thumbnail grid of every
processed clip. Click a clip to spin the 3D point cloud + camera frustums
with mouse, and scroll the per-frame depth visualizations.

### 5.2 Remote machine over SSH (this is the common case)

If DA3 lives on a remote box reachable via a bastion (`yamalab-server`)
plus a ProxyJump alias for the GPU host (`ogawa-galleria` in the validated
run):

**On the GPU host**, start the gallery bound to loopback only:

```bash
nohup .venv/bin/python -m depth_anything_3.cli gallery \
    --gallery-dir /home/mas/data/tapvid3d_da3_out \
    --host 127.0.0.1 --port 8007 \
    > /home/mas/data/tapvid3d/gallery.log 2>&1 &
disown
```

**On your laptop**, open an SSH session with port forwarding:

```bash
ssh -L 8007:localhost:8007 ogawa-galleria
```

Then browse to **http://localhost:8007** on the laptop. Stop the gallery on
the GPU host with `pkill -f "depth_anything_3.cli gallery"`.

`scene.glb` is a self-contained glTF file: drag any single
`{subset}/{clip}/scene.glb` into https://gltf-viewer.donmccurdy.com/,
Blender (`File → Import → glTF 2.0`), or VS Code's "glTF Tools" extension
for offline viewing without the gallery server.

### 5.3 True 4D playback (time-varying point cloud → MP4)

`scene.glb` is time-aggregated — points from every frame fused into one
static cloud. To see the scene *evolve over time* from a fixed viewpoint,
render a per-frame point cloud animation:

```bash
.venv/bin/python scripts/render_4d_animation.py \
    --clip-dir /home/mas/data/tapvid3d_da3_out/pstudio/tennis_23
# writes <clip-dir>/4d.mp4
```

The renderer unprojects `depth[t]` to world space using `extrinsics[t]` and
`intrinsics[t]`, applies a per-frame confidence threshold (40th percentile)
to drop noisy points, computes a single oblique camera from the aggregate
scene bbox, and emits an MP4 (default 960×720 @ 24 fps) using Open3D's EGL
headless `OffscreenRenderer`. Each frame is colored with the original
TAPVID3D RGB if `~/data/tapvid3d/{subset}/{clip-id}.npz` is reachable
(default); pass `--no-rgb` to fall back to a depth colormap.

Render speed on the validated box: ~30 ms/frame, so a 150-frame PStudio
clip renders in ~4 s.

### 5.4 Interactive 4D viewing (rotate + zoom + scrub time)

The MP4 above is a fixed-viewpoint render — useful for sharing but you
can't rotate the scene to confirm the 3D structure. For genuinely
interactive 4D inspection, export the clip to a **Rerun** `.rrd` file
and open it with the Rerun viewer:

```bash
# On the GPU box, export a clip
.venv/bin/python scripts/export_4d_to_rerun.py \
    --clip-dir /home/mas/data/tapvid3d_da3_out/pstudio/tennis_23
# writes <clip-dir>/4d.rrd  (~200 MB for a 150-frame PStudio clip)
```

The .rrd contains, per frame: the world-space point cloud (RGB-colored
when the source npz is reachable), the camera frustum, the camera's
intrinsics+extrinsics, and the original RGB image projected through the
frustum.

**Option A — download the `.rrd` and view locally (simplest)**:

```bash
# On your laptop
brew install rerun-io/rerun/rerun     # or pip install rerun-sdk
rsync -avz mas-galleria:/home/mas/data/tapvid3d_da3_out/pstudio/tennis_23/4d.rrd ~/Downloads/
rerun ~/Downloads/4d.rrd
```

Drag the 3D view with mouse to rotate, scroll to zoom, drag the time
slider at the bottom to scrub frames.

**Option B — stream live over an SSH tunnel** (no large file copy):

```bash
# On the GPU box
.venv/bin/python scripts/export_4d_to_rerun.py \
    --clip-dir /home/mas/data/tapvid3d_da3_out/pstudio/tennis_23 --serve
# Keeps running; serves on port 9876
```

```bash
# On your laptop — tunnel + connect
ssh -L 9876:localhost:9876 mas-galleria  # in one terminal, keep open
rerun --connect rerun+http://localhost:9876/proxy  # in another
```

### 5.5 Quick depth-only MP4 (no 3D)

For a quick MP4 of just the depth maps over time (no 3D rendering),
ffmpeg the existing `depth_vis/` PNGs directly:

```bash
.venv/bin/python -c "
import imageio.v2 as iio, glob
frames = sorted(glob.glob('/home/mas/data/tapvid3d_da3_out/pstudio/tennis_23/depth_vis/*.jpg'))
with iio.get_writer('depth.mp4', fps=24) as w:
    for f in frames: w.append_data(iio.imread(f))
"
```

## 6. Outputs reference

```
~/data/tapvid3d_da3_out/
├── drivetrack/<clip-id>/
│   ├── depth.npz       # depth(N,H,W) + extrinsics(N,3,4) + intrinsics(N,3,3) + conf(N,H,W)
│   ├── scene.glb       # fused colored point cloud (~1M verts) + N camera frustums
│   ├── scene.jpg       # first-frame preview
│   ├── depth_vis/      # per-frame depth visualization PNGs
│   ├── exports/        # DA3's raw mini_npz export buffers
│   └── done.flag       # completion marker; presence skips on re-runs
├── pstudio/<clip-id>/  # same layout
└── oom.log             # one line per failed clip with the GPU memory diagnostic
```

`scene.glb` is **time-aggregated** — points from every frame fused into one
static cloud. For true time-varying playback (per-frame point cloud
animating), use the arrays in `depth.npz` directly:

```python
import numpy as np
d = np.load("scene-dir/depth.npz")
depth, ex, ix = d["depth"], d["extrinsics"], d["intrinsics"]
# unproject depth[t] with ix[t], transform with ex[t], render frame t
```

## 7. Known caveats from the validated run

- 1 / 100 clips OOM on DA3-LARGE at 16 GB
  (`tapvid3d_6638427309837298695_220_000_240_000_1_SbPukwTCEiap4DzMUklocw.npz`,
  162 frames). Workarounds: drop `process_res` to 392, or chunk the clip
  into sliding windows.
- `xformers` shows a `WARNING[XFORMERS]: xFormers can't load C++/CUDA
  extensions` line at startup. The wheel is built for `pt2.10+cu128/Py3.10`,
  we run `pt2.12+cu130/Py3.12`. Attention falls back to PyTorch SDPA;
  outputs are correct, throughput is slightly lower.
- The `da3 auto` CLI works on a frame directory but loads the model per
  invocation. The driver script bypasses it specifically to amortize the
  model load (~13–32 s) across all clips.
