# Finger Joints Temperature Analyzer

A web app that measures the temperature at each of the 21 hand joints from three recordings of the same hand: an RGB video, a thermal video and radiometric (per-pixel temperature) data.

**Live demo:** https://finger-joints-temperature-analyzer-76jskclbshfj9o9feyjiw9.streamlit.app/

You upload the recordings and fill in a short form. The app then runs the analysis pipeline automatically, shows the hand being tracked while it works, and presents the results with downloadable files. Nothing is stored after the session (see [Privacy](#privacy)).

---

## Contents
1. [How it works](#how-it-works)
2. [Using the app](#using-the-app)
3. [Inputs and outputs](#inputs-and-outputs)
4. [Project structure](#project-structure)
5. [Run locally](#run-locally)
6. [Deploy for free](#deploy-for-free)
7. [Configuration](#configuration)
8. [Privacy](#privacy)
9. [Limitations](#limitations)
10. [Changes made to the original scripts](#changes-made-to-the-original-scripts)
11. [Troubleshooting](#troubleshooting)

---

## How it works

The analysis is a chain of standalone scripts. The app runs them in order, answers their terminal questions from the form, and passes each script's output folder to the next one, so no paths are typed by hand.

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `1_split_frames.py` | Takes one frame every *N* seconds from the RGB video, thermal video and radiometric data | `rgb/`, `thermal_vis/`, `thermal_rad/` |
| 2 | `2_hand_analysis_frames.py` | Detects 21 hand landmarks per frame with MediaPipe | `joint_coordinates.csv`, annotated frames |
| 3 | `3_calibrate_alignment.py` | Fits the RGB-to-thermal alignment (scale, rotation, offset) once per camera rig | `rig_calibration.json`, overlay images |
| 4 | `4_add_radiometric.py` | Maps each joint into the thermal image, corrects small per-frame shifts, reads its temperature | `joint_coordinates_radiometric.csv` |
| 5 | `5_joint_temperature_report.py` | Builds a report per hand | `lefthand_` / `righthand_temperature_report/` |
| 7 | `7_evaluate_pipeline.py` (optional) | Computes quality and performance metrics | `evaluation_report/` |

Notes on the method:
- **Alignment.** The two cameras sit about 10 mm apart. A 4-parameter similarity transform maps RGB pixels to thermal pixels, then a bounded per-frame shift (up to 6 px) refines it against that frame's warm hand region.
- **Measured values only.** A temperature is reported only where a real reading exists. A joint that falls just off the hand can be "snapped" to the nearest warm pixel (up to 6 px) and is flagged as such. Missing values are never filled in the statistics.
- **Live tracking view.** `live_shim.py` runs `hand_analysis_system_2.py` unchanged on the server and captures its display as an image that the app refreshes every second.

---

## Using the app

1. **Enter the inputs**: patient ID, RGB video, thermal video, radiometric data (one stacked `.npy` or several), and seconds between frames.
2. **Choose a calibration**: calibrate from this recording (about a minute), or upload an existing `rig_calibration.json`.
3. **Optional**: open *Advanced options* for every remaining setting of the original scripts (detection source, left/right swap, region of interest, local re-alignment, snapping, units, field of view, quality metrics).
4. **Start analysis.** Section 1 shows the status (Initializing, Tracking in progress, Processing completed, or Error) and the live tracking frame.
5. **Read the results** in Section 2 and download what you need.

---

## Inputs and outputs

**Inputs**

| Input | Notes |
|---|---|
| RGB video | Any format OpenCV can read (`.mp4`, `.avi`, `.mov`) |
| Thermal video | Same moment in time as the RGB video |
| Radiometric data | `.npy` of shape `(N, H, W)`, or several per-frame `.npy` files |
| Calibration (optional) | `rig_calibration.json` from an earlier run |
| Ground truth, reference temperatures (optional) | CSVs, used only for quality metrics |

**Results dashboard**
- Summary table per joint: max, min, mean, median, standard deviation and variance of temperature, plus joint location and where the maximum occurred
- Temperature-versus-frame plots
- Heatmap of all joints across all frames
- Joint locations, alignment check images, and evaluation metrics when requested
- Quality notes (reading coverage, alignment overlap, temperature range, warnings)

**Downloads:** a ZIP of all outputs including intermediate files, or a ZIP of key results only, and the calibration file.

---

## Project structure

```
streamlit_app.py        Web interface (inputs, live tracking, results, downloads)
core.py                 Runs the scripts, answers their prompts, passes paths between steps
live_shim.py            Lets hand_analysis_system_2.py run on a server with no screen
pipeline/               The analysis scripts (1, 2, 3, 4, 5, 7) and hand_analysis_system_2.py
models/                 hand_landmarker.task (MediaPipe model)
.streamlit/config.toml  Upload limit (1000 MB per file) and theme
requirements.txt        Python libraries
packages.txt            System libraries needed on the server
DEPLOY.md               Step-by-step free deployment
```

---

## Run locally

Requires Python 3.10 or newer (3.11 recommended).

```bash
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at http://localhost:8501. Running locally has no memory or upload limits beyond your own computer.

---

## Deploy for free

The app is built for Streamlit Community Cloud: push the repository to GitHub, create an app on https://share.streamlit.io pointing at `streamlit_app.py`, choose Python 3.11 in Advanced settings, and optionally set `APP_PASSWORD` in Secrets. Full steps are in [DEPLOY.md](DEPLOY.md).

---

## Configuration

Set these as environment variables, or in Streamlit Secrets when deployed.

| Name | Meaning | Default |
|---|---|---|
| `APP_PASSWORD` | If set, visitors must enter it | none (public) |
| `HTA_ABANDON_SECONDS` | Delete a run this many seconds after the browser disconnects | 90 |
| `HTA_DATA` | Folder for temporary uploads and results | system temp folder |

---

## Privacy

Recordings may contain patient data, so the app keeps nothing:
- A run exists only while its browser page is open.
- Clicking a download, choosing **Finish without downloading**, or refreshing/closing the page deletes the uploads and results from the server. After a download click, deletion happens after a short 30-second grace period so the file can finish transferring.
- A server restart wipes everything.

Download what you need before leaving the page. If the app is public, set `APP_PASSWORD` and only upload data you are permitted to share. The hosting provider's own infrastructure is outside the app's control.

---

## Limitations

- **No browser webcam.** The "live" view is server-side processing of your uploaded recordings, not a camera feed.
- **Free-tier resources.** Streamlit Community Cloud provides roughly 2.7 GB of RAM and 2 CPU cores, so large radiometric stacks (several hundred MB) can exhaust memory because step 1 loads the stack fully. Mitigations: switch off the live view, use a larger frame interval, or upload several smaller `.npy` files. Run locally for large datasets.
- **One analysis at a time.** Additional runs wait in a queue.
- **Idle apps sleep** on the free tier and wake when opened.
- **Ground-truth annotation** (`6_annotate_ground_truth.py`) needs a desktop window and mouse clicks, so it is not part of the web app. Create the ground-truth CSV on your own computer and upload it in the optional quality section.
- Calibration belongs to one fixed camera setup. Recalibrate if the cameras move.

---

## Changes made to the original scripts

The analysis logic is unchanged. The only edit is the model path in `2_hand_analysis_frames.py` and `hand_analysis_system_2.py`: the hard-coded `N:\AIML\...` path now reads the `HAND_MODEL_PATH` environment variable, falling back to `models/hand_landmarker.task`. The scripts' terminal prompts are answered automatically by `core.py`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No hand was detected" | The hand is not clearly visible in the RGB video, or the region of interest is wrong |
| Left and right hands reversed | Tick *Swap Left/Right labels* in Advanced options |
| Low overlap or alignment warning | Choose *Calibrate from this recording* and check the alignment images |
| App restarts on large uploads | Memory limit reached: turn off the live view, increase the frame interval, or run locally |
| Build fails on `libGL.so.1` | Make sure `packages.txt` is in the repository root |
| Build fails installing mediapipe | Select Python 3.11 in Advanced settings and redeploy |

The MediaPipe model `hand_landmarker.task` is a public Google file.
