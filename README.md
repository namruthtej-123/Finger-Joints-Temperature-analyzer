# Hand Temperature Analyzer (Streamlit)
live Demo :https://finger-joints-temperature-analyzer-76jskclbshfj9o9feyjiw9.streamlit.app/
Web version of the hand-joint temperature pipeline. Upload an RGB video, a thermal video and the radiometric
data; the app runs your five scripts one after another, shows the hand tracking while it runs, then shows the
results (summary table, temperature-vs-frame plot, heatmap, quality check) and offers everything as a ZIP.

## What is in this folder

| File / folder | Purpose |
|---|---|
| `streamlit_app.py` | The web interface (inputs, Section 1 live tracking, Section 2 results + downloads) |
| `core.py` | Runs the pipeline scripts, answers their terminal questions, hands each output path to the next script |
| `live_shim.py` | Lets `hand_analysis_system_2.py` run on a server with no screen (see below) |
| `pipeline/1_split_frames.py` ... `5_joint_temperature_report.py`, `7_evaluate_pipeline.py` | **Your scripts, logic unchanged** |
| `pipeline/hand_analysis_system_2.py` | Your live tracking script, logic unchanged |
| `models/hand_landmarker.task` | MediaPipe model |
| `.streamlit/config.toml` | Upload limit (1000 MB per file), light theme |
| `requirements.txt`, `packages.txt` | Python libraries / system libraries for the server |

### The only edit made inside your scripts
`MODEL_PATH` in `2_hand_analysis_frames.py` and `hand_analysis_system_2.py` pointed to `N:\AIML\...` on your PC.
It now reads the environment variable `HAND_MODEL_PATH` (set by the app) or `models/hand_landmarker.task`.
Nothing else in them was touched.

## Every terminal input is now in the interface
| Script question | Where it is in the app |
|---|---|
| 1: patient ID, RGB/thermal video paths, radiometric file or folder, output folder, seconds between frames | Sections 1-2 (uploads; folders are automatic) |
| 2: swap Left/Right, participant, frames folder, detect on RGB or thermal, interval, ROI, output folder | Patient ID, Advanced options (rest automatic) |
| 3: frames folder, CSV, rig file location, field of view, camera spacing, distance, frames used, orientation | Section 3 + Advanced options (paths automatic) |
| 4: CSV, thermal folder, calibration file, local re-alignment, snapping, units, "proceed anyway" | Advanced options (paths automatic; "proceed" is answered yes and shown as a warning) |
| 5: CSV | Automatic |
| 7: CSV, patient folder, calibration, ground-truth CSV, reference CSV, mm per pixel | Advanced options > "Also compute quality metrics" |

`6_annotate_ground_truth.py` is not included: it needs a desktop window and mouse clicks, which a web server cannot show.
Create the ground-truth CSV on your PC with that script and upload it in the optional quality section.

## Live hand tracking: how it works on a server
A cloud server has no screen and no camera, so the original OpenCV window cannot open. `live_shim.py` starts
`hand_analysis_system_2.py` exactly as it is, but replaces `cv2.imshow` with "save this picture". The app shows
that picture, refreshing every second, with the status Initializing / Tracking in progress / Processing completed /
Error. The operator therefore sees the skeleton being tracked on the uploaded video.
It does **not** use a live webcam in the browser. If you want that later, it needs a different component (WebRTC).
You can switch the live view off under Advanced options to save CPU and memory.

## Run it on your own computer
```
python -m venv venv
venv\Scripts\activate          (Windows)      source venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
streamlit run streamlit_app.py
```
It opens at http://localhost:8501. Outputs are written to a temporary folder (set `HTA_DATA` to choose another).

## Deploy it free on the internet
See `DEPLOY.md`.

## Optional settings (environment variables or Streamlit "Secrets")
| Name | Meaning | Default |
|---|---|---|
| `APP_PASSWORD` | If set, visitors must enter it | none (public) |
| `HTA_ABANDON_SECONDS` | Delete a run this many seconds after the browser disconnects | 90 |
| `HTA_DATA` | Folder for uploads / results | system temp folder |
