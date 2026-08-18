# Monocular Driving Perception & Bird's-Eye-View Mapping

Turkish version: [README.tr.md](README.tr.md)

A perception pipeline that takes a **single forward-facing camera** feed and answers:

- What objects are in the scene? (vehicles, pedestrians, cyclists, traffic lights)
- How do they move across frames? (multi-object tracking)
- How far away are they? (monocular depth)
- Where is the drivable area and the lane structure? (segmentation)
- Is a collision likely, and how soon? (time-to-collision)

The output is a two-panel video: the annotated camera view on the left, a live
bird's-eye-view map of vehicles and lanes on the right.

![Detection, tracking and bird's-eye-view projection running on dashcam footage from Maltepe, Istanbul](docs/assets/demo.gif)

*Own dashcam footage, Maltepe/Istanbul. Left: detections with persistent track
IDs, motion trails, and relative depth. Right: ground-plane positions through a
homography. Distances on the map are measured from the calibration reference
row, not from the vehicle's nose — see [Calibrating the bird's-eye view](#calibrating-the-birds-eye-view).*

> **Status: 6 of 8 weeks done** — detection, tracking, depth, bird's-eye-view
> projection, TTC, and drivable-area segmentation all run. Remaining: a live
> demo and the failure analysis. The [Roadmap](#roadmap) is the authority on
> what works today.

---

## Why this project

An autonomous-driving perception stack packs nearly every core computer-vision
problem under one roof — detection, tracking, depth, segmentation, and geometric
projection. One project, a wide competence surface.

The deliverable is also directly visual: someone can see what was built in ten
seconds without reading a line of code.

---

## Architecture

```
                          Video frame
                               |
        +----------------------+----------------------+
        v                      v                      v
  +-----------+        +--------------+        +--------------+
  | Detection |        |    Depth     |        | Segmentation |
  |  (YOLO)   |        |(Depth Any.)  |        | (lane/road)  |
  +-----+-----+        +------+-------+        +------+-------+
        |                     |                       |
        v                     |                       |
  +-----------+               |                       |
  | Tracking  |               |                       |
  |(ByteTrack)|               |                       |
  +-----+-----+               |                       |
        |                     |                       |
        +----------+----------+-----------------------+
                   v
            +--------------+
            | Fusion layer |
            | (box + depth)|
            +------+-------+
                   |
        +----------+----------+
        v                     v
  +-----------+        +--------------+
  |    BEV    |        |     Risk     |
  | projection|        | (speed, TTC) |
  +-----+-----+        +------+-------+
        |                     |
        +----------+----------+
                   v
          Rendered output video
```

### Layer notes

| Layer | Role |
|---|---|
| **Detection** | Pre-trained YOLO marks objects with bounding boxes. COCO classes are sufficient; no custom training. |
| **Tracking** | ByteTrack assigns a persistent ID across frames. Speed and TTC depend on this ID continuity, which makes tracker stability the most critical technical point in the project. |
| **Depth** | Depth Anything produces a relative depth value per pixel. An object's depth is the **median** over the lower-centre region of its box — median, not mean, so a few outlier pixels cannot drag the estimate. |
| **Fusion** | Boxes and depth are merged into `(id, class, image position, depth)` per tracked object. |
| **Projection** | The bottom edge of a box is taken as the object's ground-contact point and mapped onto the ground plane through a homography. |
| **Risk** | Relative speed comes from track history; TTC follows from distance over closing speed. Objects below a threshold are highlighted. |

---

## Install

Requires Python 3.10+.

```bash
git clone <repo-url> otonomarac
cd otonomarac
python -m venv .venv
```

Activate the environment — `.venv\Scripts\activate` on Windows,
`source .venv/bin/activate` elsewhere — then install PyTorch for your hardware
**before** the other dependencies:

```bash
# NVIDIA Pascal GPUs (GTX 10xx) or driver older than 527:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Recent NVIDIA GPUs with an up-to-date driver:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# No GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
pip install -e .
python scripts/smoke_test.py
```

Model weights download automatically on first run.

> **If `pip install -e .` fails with `WinError 5` on Windows**, setuptools is
> being blocked while writing its metadata (usually antivirus real-time
> scanning). Put the sources on the path directly instead — functionally
> equivalent for development:
>
> ```bash
> python -c "import sysconfig,pathlib; pathlib.Path(sysconfig.get_paths()['purelib'], 'perception_src.pth').write_text(str(pathlib.Path('src').resolve()))"
> ```

---

## Usage

```bash
# Simplest run — writes outputs/result.mp4
python -m perception.cli --input data/drive.mp4

# Quick smoke test on the first 100 frames
python -m perception.cli --input data/drive.mp4 --max-frames 100

# Larger model, half precision, explicit device
python -m perception.cli --input data/drive.mp4 --model yolov8s.pt --half --device cuda

# Config file, overridden by any flag you also pass
python -m perception.cli --config configs/default.yaml --input data/drive.mp4
```

Run `python -m perception.cli --help` for the full flag list. Every run prints a
per-stage timing table and a tracking report at the end.

### Tracking parameter sweep

Tracker settings are measured, not guessed:

```bash
python scripts/tracking_sweep.py data/drive.mp4 --config configs/default.yaml
```

Each experiment differs from the baseline along a single axis, so a change in
the numbers can be attributed. The script reports unique IDs, median track
length, fragmentation rate, and throughput. Results and the reasoning behind
the shipped defaults are in [docs/proje-gunlugu.md](docs/proje-gunlugu.md).

Two findings worth knowing before you tune anything:

- **`detection.conf` must stay below `tracking.track_low_thresh`.** ByteTrack's
  whole point is reusing low-confidence detections in a second association pass
  to recover lost tracks. If the detector filters them out first, that pass gets
  nothing and the algorithm silently degrades to plain IoU tracking. Raising the
  detector threshold to 0.15 pushed fragmentation from 44% to 53%. `Config.validate()`
  warns about this.
- **Raising `match_thresh` loosens matching, it does not tighten it.** The cost
  matrix is `1 - IoU` and the threshold is an upper bound (`lap.lapjv(cost_limit=…)`),
  so `0.9` means "accept IoU ≥ 0.1". On 10 Hz footage the 0.8 default is too
  strict and breaks tracks.

### Depth is relative, and the numbers say so

Object labels show `~2.0`, never `2.0 m`. Absolute distance is not recoverable
from one camera, and printing a unit would claim a precision the system does not
have. The value is unitless and only meaningful for comparing objects; scale
calibration against a fixed reference is week 5's job.

The stored depth map is the model's **raw** output, deliberately not normalised.
Rescaling each frame to `[0,1]` looks better but makes values incomparable
between frames — one nearby object entering the scene shifts the whole map's
scale, so every object's "distance" changes while nothing has moved. Speed
estimation depends on exactly that frame-to-frame difference. Normalisation
happens only in the drawing layer, only for the colour map.

### Calibrating the bird's-eye view

The homography needs four points on the road plane, and they are camera- and
mount-specific:

```bash
python scripts/calibrate_bev.py data/drive.mp4 --frame 900 \
    --quad 0.414,0.839 0.654,0.839 0.579,0.728 0.493,0.728 --verify
```

`--verify` checks the calibration against something it was **not** built from:
the projected width of detected vehicles. Two numbers matter — the median should
land near a typical vehicle width (~1.8 m), and the width must be uncorrelated
with distance. The second is the real test; a non-zero correlation means the
perspective was not properly removed.

An earlier version of this check measured whether the yellow centre line came
out vertical after warping. That was circular — the quad was built from that
same line, so the answer was always perfect, and dozens of different horizon
heights scored identically. A check that rejects no configuration is not a check.

`quad_depth_m` is an **assumption, not a measurement**. Varying it from 6 m to
22 m leaves both the median width and the distance correlation unchanged while
scaling every absolute distance on the map. Lateral scale can be pinned to a
reference; longitudinal scale cannot, with this test. That is the monocular
scale ambiguity, stated plainly rather than hidden behind a number.

### Tuning for slow hardware

| Flag | Effect |
|---|---|
| `--frame-stride 2` | Process every other frame; output FPS is divided to match, so playback stays real-time |
| `--resize-width 960` | Shrink frames before inference |
| `--imgsz 480` | Smaller model input |
| `--model yolov8n.pt` | Smallest weights (default) |

---

## Layout

```
configs/         YAML configuration
src/perception/
  config.py      Dataclass config, YAML loading, CLI overrides
  video_io.py    Frame reading/writing, stride, resizing
  detection.py   YOLO wrapper -> Detection objects
  tracking.py    ByteTrack wrapper, track stats, motion trails
  visualize.py   All OpenCV drawing
  pipeline.py    Layer orchestration
  cli.py         Command-line entry point
  utils.py       Device selection, per-stage profiler
scripts/         Smoke test, KITTI conversion, parameter sweep
data/            Input videos (git-ignored)
outputs/         Rendered results (git-ignored)
docs/            Project report and engineering log
```

---

## Roadmap

| Week | Milestone | Status |
|---|---|---|
| 1 | Repo skeleton, video I/O, YOLO detection | **done** — 50.7 FPS on GTX 1080 |
| 2 | ByteTrack integration, IDs, motion trails | **done** — 17% track fragmentation |
| 3 | Depth Anything, box–depth fusion | **done** — relative depth per object |
| 4 | Homography, bird's-eye-view map, two-panel render | **done** — 0.1 ms/frame |
| 5 | Relative speed, scale calibration, TTC | **done** — TTC is scale-invariant |
| 6 | Lane / drivable-area segmentation | **done** — visible free space on the map |
| 7 | Gradio UI, Hugging Face Spaces deploy | |
| 8 | Documentation, failure analysis, performance table | |

The rule: **every week ends with something that runs.** The next layer starts
only after the current one works end to end, so the project stays presentable
whenever it stops.

---

## Design decisions

**Bird's-eye-view projection — two routes.** Method A assumes the road is a flat
plane and maps four image points to four ground points through a homography.
It needs no depth model, and it is fast and stable, but it breaks on slopes and
crests. Method B lifts the object into 3D using depth plus camera intrinsics and
projects from above; more accurate on non-flat roads, but monocular depth is
relative, so scale is ambiguous and calibration is required. **Method A ships
first, Method B follows, and the two are compared in a dedicated section** —
"I tried both, here is where they differ" is a far stronger claim than one method
presented alone.

**Scale ambiguity, and why TTC escapes it.** Absolute distance is not
recoverable from a single camera. Worse than the initial plan assumed: a
*lateral* reference such as lane or vehicle width cannot fix the *longitudinal*
scale at all, because focal length cancels out of the lateral relation and does
not cancel out of the depth one:

```
X = (u − u_c) · h / (v − v_h)      ← f cancels
Z = f · h / (v − v_h)              ← f remains
```

So the lateral calibration pins the camera height — it comes out at 1.43 m over
107 vehicle samples, right for a windscreen mount — while `bev.quad_depth_m`
stays an assumption. Map distances in metres are uncertain up to one factor.

**Time-to-collision is not.** Scale every distance by an unknown *k* and the
closing speed scales by the same *k*, so `TTC = d / v` is unchanged. Measured:
doubling `quad_depth_m` doubles the reported distance (1.3 → 2.6 → 5.2 m) and
leaves TTC at 1.29 s every time. The most useful output turned out to be
independent of the weakest assumption.

**No model training.** Pre-trained weights are a prioritisation, not laziness.
The skills this project exists to demonstrate are **system integration, geometry,
and failure analysis** — not retraining a classifier.

---

## Explicitly out of scope

Deliberately excluded, because they need hardware or would make the project
unfinishable:

- Running on a real vehicle
- Planning or control (steering, braking commands)
- LiDAR, radar, or multi-sensor fusion
- Simulator (CARLA) integration
- Training models from scratch
- Metric (absolute) depth measurement

Knowing what you did not build, and why, is engineering maturity — not a gap.

---

## Success criteria

- [ ] An arbitrary driving video processes end to end without errors
- [ ] The BEV map shows relative vehicle positions consistently
- [ ] The live demo link works and is publicly reachable
- [ ] The README states what the system does, how it works, and **where it fails**
- [ ] Performance measurements are reported

Note the absence of an accuracy metric (mAP and friends). This is a systems
integration project, not a model training project; it is judged on whether it
runs and how clearly it is explained.

---

## License

MIT
