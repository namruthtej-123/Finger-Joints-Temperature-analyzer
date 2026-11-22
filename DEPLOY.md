# Deploying for free: Streamlit Community Cloud (public web address, no cost)

Streamlit Community Cloud runs the app straight from a GitHub repository and gives it a public `https://...streamlit.app` address.

## Steps
1. **GitHub account** (free): https://github.com
2. **Create a new repository**, e.g. `hand-temperature-analyzer` (Public is the simplest; Community Cloud also supports private repos for your own account).
3. **Upload this whole folder** into the repository (GitHub web page > *Add file* > *Upload files*, drag the folder's contents, *Commit*).
   Keep the structure: `streamlit_app.py`, `core.py`, `live_shim.py`, `requirements.txt`, `packages.txt`, `pipeline/`, `models/`, `.streamlit/`.
   (The model file is 7.8 MB, well within GitHub's limits.)
4. Go to https://share.streamlit.io and sign in with GitHub.
5. **Create app** > *Deploy a public app from GitHub* and choose:
   - Repository: yours
   - Branch: `main`
   - Main file path: `streamlit_app.py`
6. Open **Advanced settings**:
   - **Python version: 3.11** (the safest choice for MediaPipe).
   - **Secrets** (optional but recommended, so strangers cannot use your app): paste
     ```
     APP_PASSWORD = "choose-a-password"
     ```
7. Click **Deploy**. The first build takes a few minutes (it installs MediaPipe and OpenCV). Then you get your public address; share it with anyone.

Updating later: change files in GitHub; the app redeploys by itself.

## Things to know before relying on it (free hosting limits)
- **Memory and CPU are small.** Community Cloud gives an app roughly 2.7 GB RAM at most and up to 2 CPU cores (Streamlit documentation; limits can change). Processing will be slower than on your PC.
  Script 1 loads a stacked radiometric `.npy` completely into memory, so a stack of several hundred MB, plus the live view, may exceed the limit and the app shows "Oh no" / restarts.
  If that happens: switch off the live view, use a larger "one frame every N seconds", or upload the radiometric data as several smaller `.npy` files, or use one of the options below.
- **Upload size:** `.streamlit/config.toml` sets 1000 MB per file. The platform may enforce a lower cap; if a big upload fails, reduce the size.
- **Nothing is stored:** each run lives only while its browser page is open. Clicking a download, choosing "Finish without downloading", refreshing or closing the page deletes the uploads and results (within about 90 seconds), and everything is wiped when the server restarts. Download the ZIP and `rig_calibration.json` before leaving. Idle apps go to sleep and wake when someone opens the address.
- **One analysis at a time:** extra jobs wait in a queue.
- **Privacy:** uploaded recordings are held on a third-party server. Use the password, delete runs with *Delete this run's data*, and do not upload identifiable patient data unless you are allowed to.
- **Public address means public use** unless you set `APP_PASSWORD`.

## If the free limits are too small
- **Run it on a lab PC and open it from other computers** on the same network: `streamlit run streamlit_app.py --server.address 0.0.0.0` (no hosting cost, no size limits).
- **A free always-on virtual machine** (for example Oracle Cloud's "Always Free" tier, which needs a card for sign-up verification) can run the same commands as on your own computer and gives much more memory. This needs some server setup.
- Hugging Face Spaces: its documentation currently says Docker/Gradio Spaces that run on compute need a paid plan, so it is not a zero-cost route to rely on. Check their page before choosing it.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Build fails on `libGL.so.1` | Make sure `packages.txt` is in the repo root (it installs `libgl1`) |
| Build fails installing mediapipe | Choose Python 3.11 in Advanced settings and redeploy |
| "No hand was detected" | Hand not visible in the RGB video, or the detection area is wrong |
| Quality notes about low overlap | Choose "Calibrate from this recording" |
| Left and Right reversed | Tick "Swap Left/Right labels" in Advanced options |
