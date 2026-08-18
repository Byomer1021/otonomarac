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

*Own dashcam footage, Maltepe/Istanbul. **Left:** detections with persistent
track IDs, motion trails and relative depth; the vehicle ahead is outlined in
red once its time-to-collision drops below two seconds. **Right:** ground-plane
positions through a homography, over the drivable area and lane paint the
segmentation layer found. Gaps in the map are directions no camera saw — every
vehicle casts an occlusion shadow. Map distances are measured from the
calibration reference row, not the vehicle's nose; see
[Calibrating the bird's-eye view](#calibrating-the-birds-eye-view).*

**[▶ Try it in the browser](https://huggingface.co/spaces/byomer1021/otonomarac)** — upload a clip, get the annotated video and a measurement report. No install.

> **Status: complete.** All eight weeks are done — detection, tracking, depth,
> bird's-eye-view projection, TTC and drivable-area segmentation run end to end,
> the demo is live, and [Where it fails](#where-it-fails) reports what breaks and
> by how much.

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

## Live demo

A Gradio interface runs the whole stack on an uploaded clip:

```bash
pip install -r requirements-app.txt
python app.py
```

It is built for the free CPU tier, so it does not pretend to be real-time: a
clip goes in, the first 15 seconds are processed offline, the annotated video
and a measurement report come back.

**Tuning for CPU is measured, not guessed.** Dropping to `yolov8n` at 480 px,
processing at 960 px wide, and recomputing depth every 5 frames and segmentation
every 10 takes the pipeline from 342 ms/frame to 121 ms — a 15-second clip
finishes in about a minute. One result worth recording: restricting torch to two
threads instead of six changed almost nothing, so at these model sizes the
bottleneck is per-operation overhead rather than parallel compute. Full settings
and reasoning: [configs/spaces.yaml](configs/spaces.yaml).

**The bird's-eye map stays off for uploaded video.** A homography needs four
points on the road plane, specific to the camera and how it is mounted, and that
information does not exist for a stranger's clip. Shipping a plausible-looking
default would produce distances that read as measured and are not. Detection,
tracking and depth are camera-agnostic and run on anything; the bundled Maltepe
example has a real calibration, so it shows the map too.

### Deploying to Hugging Face Spaces

Authenticate once, then run the deploy script:

```bash
hf auth login                      # token from huggingface.co/settings/tokens (Write)
python scripts/deploy_space.py     # creates <user>/otonomarac and uploads
```

`--dry-run` stages the upload without touching the Hub, so you can inspect
exactly what would be sent (20 files, 1.8 MB — sources, configs, the app and the
example clip; no data or outputs).

The script generates the Space's `requirements.txt` rather than copying this
repo's, for two reasons. The repo file deliberately omits torch because the
right wheel depends on the machine; left as-is, pip on the Space would resolve
torch through ultralytics and pull the **CUDA** build — several gigabytes onto a
CPU-only container. The generated file asks for the CPU wheel explicitly. It also
writes `sdk_version` from the locally installed gradio, so the Space starts on
the version the app was actually tested against instead of a hand-maintained
number that drifts.

Model weights download on first run, so the first request is slower than the
timings above.

> **Spaces pricing changed in 2026.** Hosting a Gradio Space on the free
> `cpu-basic` tier now requires a PRO subscription; free accounts get static
> Spaces and ZeroGPU. The deploy script defaults to `cpu-basic`; pass
> `--hardware zero-a10g` for the free GPU tier, which also needs a `@spaces.GPU`
> decorator in `app.py`.


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
| 7 | Gradio UI, Hugging Face Spaces deploy | **done** — 342 → 121 ms/frame on CPU |
| 8 | Documentation, failure analysis, performance table | **done** |

The rule: **every week ends with something that runs.** The next layer starts
only after the current one works end to end, so the project stays presentable
whenever it stops.

---

## Where it fails

Measured, not asserted — [`scripts/failure_analysis.py`](scripts/failure_analysis.py)
produces every number in [docs/hata-analizi.md](docs/hata-analizi.md), comparing
a dense city clip against an almost-empty rural one from the same camera.

Four findings worth knowing before trusting any output:

**Projection degrades with range.** Frame-to-frame distance noise for a tracked
object grows 3.7× from the near band to the far one: median 3.9 m/s at 0-8 m
against 14.6 m/s at 25-40 m, with a 90th percentile of 54 m/s — 196 km/h, which
nothing in a city street does. The cause is geometric: ground distance 40 m sits
at image row 497.6 and 60 m at 492.3, so a five-pixel strip covers twenty
metres. Anything past 25 m on the map is indicative, not measured.

**Occlusion breaks the ground-contact assumption.** The system assumes a box's
bottom edge is where the object meets the road. Partly hide the object and the
visible bottom sits above the true contact point, so the homography places it
too far away. Occluded boxes carry double the distance noise of clear ones
(10.0 against 5.2 m/s median). A fit-quality gate suppresses the resulting false
warnings, at the cost of thinning genuine ones from 39 observations to 21. The
underlying distance is still wrong; only the alarm is suppressed.

**An empty scene turns the bonnet into phantom vehicles.** `detection.conf` sits
at 0.05 because ByteTrack's second association pass needs it. With no real
objects in frame, that threshold labels reflections on the bonnet as cars — 71%
of detections on the rural clip, median confidence 0.13, against 4% in the city.
Detections more than half below the bonnet line are now dropped before tracking:
rural raw detections fall from 890 to 330 and usable ground positions rise from
20% to 55%, while the city scene moves by 2%.

**Tracking difficulty comes from occlusion, not object count.** The rural clip
has 5 tracks against the city's 110, yet zero gapped tracks against 25% — with
nothing to hide behind, no track is ever lost and re-found.

**Rain does not break detection; visibility does.** A second recording — heavy
rain in Istanbul — puts median confidence on the open road at 0.75, *above* the
0.63 of the dry city clip, because motorway vehicles are larger and better
separated than city traffic. The controlled comparison is within that same
drive: as mist closes in, median confidence falls 0.75 → 0.57, fragmentation
rises 19% → 31%, and median track length halves from 16 frames to 9. Rain's real
cost is continuity — gapped tracks reach 42% on the wet road against 23% dry,
which is exactly what speed estimation is sensitive to.

Untested rather than working: night (both recordings are daytime), and sloped
roads (Maltepe is flat, so the flat-plane assumption was never stressed). The
rain camera was not calibrated, so its bird's-eye map and TTC are unmeasured.

### Performance

Per frame, 1280 px wide, GTX 1080 for detection and CPU for the two heavy models:

| Stage | ms | Note |
|---|---|---|
| depth | 117.0 | CPU, recomputed every 3rd frame |
| segmentation | 57.2 | CPU, every 10th frame |
| detection | 20.4 | GPU |
| drawing | 9.5 | two panels |
| tracking | 2.8 | |
| fusion | 0.8 | |
| risk | 0.4 | |
| projection | 0.1 | |
| **end to end** | **220.8** | **4.5 FPS** |

On the free CPU tier with the tuned profile: 121 ms/frame, about 8 FPS.
Warm-up is excluded — the first CUDA inference costs 4 s and would otherwise
have reported the week-1 pipeline as 22.4 FPS instead of 50.7.

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

- [x] An arbitrary driving video processes end to end without errors
- [x] The BEV map shows relative vehicle positions consistently — cross-checked
      against monocular depth, which shares no input beyond the frame itself
- [x] The live demo link works and is publicly reachable —
      [huggingface.co/spaces/byomer1021/otonomarac](https://huggingface.co/spaces/byomer1021/otonomarac)
- [x] The README states what the system does, how it works, and **where it fails**
      — [Where it fails](#where-it-fails), measured in [docs/hata-analizi.md](docs/hata-analizi.md)
- [x] Performance measurements are reported — per stage, warm-up excluded

Note the absence of an accuracy metric (mAP and friends). This is a systems
integration project, not a model training project; it is judged on whether it
runs and how clearly it is explained.

---

## License

MIT
