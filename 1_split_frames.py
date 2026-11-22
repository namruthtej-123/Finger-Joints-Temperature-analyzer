# split_frames.py
""""
# split_frames.py — Video & Radiometric Frame Extractor

A small CLI script that splits an **RGB video**, a **thermal video**, and a
**radiometric `.npy`** dataset into per-frame files, sampled at a fixed time
interval (e.g. 1 frame every 5 seconds), and organises the output into a
**per-patient folder**.

---

## 1. What this script does

Given three inputs that are aligned in time —

- an **RGB video** (e.g. `static_RGB.mp4`),
- a **thermal video** (e.g. `static_thermal.mp4`),
- **radiometric data** as either a single stacked `.npy` of shape `(N, H, W)`
  or a **folder of per-frame `.npy` files**,

— the script extracts **one frame every *N* seconds** from each source and
writes the results into three sub-folders under a **patient-specific output
folder**.

The **same time interval** is applied to all three sources (converted into a
frame-step per source using that source's own fps), so:




all correspond to the **same moment** in the recording.

---

## 2. Inputs

All inputs are supplied **interactively at runtime** via `input()` — nothing
is hard-coded.

| # | Prompt | Type | Example | Notes |
|---|--------|------|---------|-------|
| 1 | `Enter patient ID` | string | `P042` | Used as the output folder name. Letters / digits / `_` / `-` only. |
| 2 | `Path to RGB video` | path | `N:\data\rgb.mp4` | Any format OpenCV can open (`.mp4`, `.avi`, `.mov`, …) |
| 3 | `Path to thermal video` | path | `N:\data\thermal.mp4` | Must be aligned in time with the RGB video |
| 4 | `Radiometric data is [1] a single .npy file or [2] a folder of .npy files ?` | `1` or `2` | `1` | Chooses storage format of the radiometric data |
| 5a | *(if `1`)* `Path to radiometric .npy file` | path | `N:\data\rad.npy` | Expects shape `(N, H, W)` or `(H, W, N)` |
| 5b | *(if `2`)* `Path to folder of .npy files` | path | `N:\data\rad\` | `.npy` files sorted lexicographically |
| 6 | `Output root folder` | path | `N:\AIML\frames_out` | Parent of all patient folders |
| 7 | `Extract 1 frame every how many seconds? [5]` | float | `5` | The sampling interval |

> **Note on paths:** the helper `ask_path()` strips Windows "Copy as path"
> quotes (`"…"`), surrounding single quotes, and stray zero-width characters,
> so you can paste a path directly from Explorer without editing it.

> **Note on radiometric fps:** the radiometric `.npy` carries no fps metadata.
> The script **derives** it automatically as:
>
> ```
> rad_fps = n_radiometric_frames ÷ thermal_video_duration_seconds
> ```
>
> If the radiometric and thermal frame counts differ by more than 5 %, a
> warning is printed and the alignment is assumed to be **by duration**, not
> by index. This is the correct behaviour when the two sources sample at
> different rates.

---

## 3. Outputs

For a patient with ID `P042` and output root `N:/AIML/frames_out`, the
following is produced:

"""

import os
import re
import cv2
import numpy as np
from pathlib import Path

print('########################')
print('GENERATES FRAMES FOR RGB , THERMAL , RADIOMETRIC DATA')
print('########################')


# -------------------- helpers --------------------
def ask_path(prompt: str) -> str:
    p = input(prompt).strip().strip('"').strip("'").strip()
    p = p.replace("\u200b", "").replace("\xa0", " ")
    return os.path.normpath(p) if p else p


def ask_patient_id(prompt: str) -> str:
    """Read a patient ID and validate it is safe to use as a folder name."""
    pid = input(prompt).strip()
    if not pid:
        raise SystemExit("Patient ID cannot be empty.")
    # Only letters, digits, '-' and '_' — this rules out '/', '\', '..', spaces, etc.
    if not re.match(r"^[A-Za-z0-9_-]+$", pid):
        raise SystemExit(
            "Patient ID must contain only letters, digits, '-' or '_' "
            "(no spaces, slashes, dots, etc.)."
        )
    return pid


def video_info(path):
    """Return (n_frames, fps, duration_sec) for a video file."""
    cap = cv2.VideoCapture(str(path))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    dur = n / fps if fps and fps > 0 else None
    return n, fps, dur


def count_npy_frames(rad_path, rad_kind):
    """How many radiometric frames are there, regardless of storage format?"""
    if rad_kind == "1":
        return int(np.load(rad_path, mmap_mode="r").shape[0])
    return len(list(Path(rad_path).glob("*.npy")))


# -------------------- extractors --------------------
def extract_video_frames(video_path, out_dir, prefix="frame", ext=".png",
                         interval_sec=5.0):
    """Write 1 frame every `interval_sec` seconds. No cap."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"fps unknown for {video_path}")

    step = max(1, int(round(fps * interval_sec)))
    print(f"[video] {video_path}")
    print(f"        {n} frames | {fps:.2f} fps | {w}x{h}")
    print(f"        step = {step}  (≈ 1 frame / {interval_sec}s)")

    out_idx, saved = 0, 0
    src_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if src_idx % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_{out_idx:06d}{ext}"), frame)
            out_idx += 1
            saved   += 1
        src_idx += 1
    cap.release()
    print(f"        -> {saved} frames written to {out_dir}")
    return saved


def extract_npy_stacked(npy_path, out_dir, prefix="th",
                        interval_sec=5.0, rad_fps=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    data = np.load(npy_path)
    print(f"[npy ] {npy_path}  shape={data.shape}  dtype={data.dtype}")

    if data.ndim == 2:
        np.save(os.path.join(out_dir, f"{prefix}_000000.npy"), data)
        return 1
    if data.ndim != 3:
        raise ValueError(f"Unsupported shape {data.shape}")
    if data.shape[-1] < 10 and data.shape[0] > 10:
        data = np.moveaxis(data, -1, 0)               # -> (N,H,W)

    N = data.shape[0]
    step = max(1, int(round(rad_fps * interval_sec)))
    idxs = list(range(0, N, step))

    for out_idx, src_idx in enumerate(idxs):
        np.save(os.path.join(out_dir, f"{prefix}_{out_idx:06d}.npy"), data[src_idx])
    print(f"        rad_fps={rad_fps:.2f}  step={step}  -> {len(idxs)} .npy frames in {out_dir}")
    return len(idxs)


def extract_npy_folder(npy_dir, out_dir, prefix="th",
                       interval_sec=5.0, rad_fps=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(npy_dir).glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in {npy_dir}")

    step = max(1, int(round(rad_fps * interval_sec)))
    idxs = list(range(0, len(files), step))

    for out_idx, src_idx in enumerate(idxs):
        np.save(os.path.join(out_dir, f"{prefix}_{out_idx:06d}.npy"),
                np.load(files[src_idx]))
    print(f"[npy ] rad_fps={rad_fps:.2f}  step={step}  "
          f"copied {len(idxs)} frames -> {out_dir}")
    return len(idxs)


# -------------------- main --------------------
if __name__ == "__main__":

    # ---- 1. Patient ID FIRST ----
    patient_id = ask_patient_id("Enter patient ID (letters/digits/_/- only): ")

    # ---- 2. Paths ----
    rgb_video = ask_path("Path to RGB video        (e.g. N:/data/rgb.mp4): ")
    th_video  = ask_path("Path to thermal video    (e.g. N:/data/thermal.mp4): ")

    rad_kind = input("Radiometric data is [1] a single .npy file  or [2] a folder of .npy files ? ").strip()
    if rad_kind == "1":
        rad_path = ask_path("Path to radiometric .npy file   : ")
    elif rad_kind == "2":
        rad_path = ask_path("Path to folder of .npy files    : ")
    else:
        raise SystemExit("Choose 1 or 2.")

    out_root = ask_path("Output root folder       (e.g. N:/data/frames_out): ")

    # ---- 3. Interval ----
    int_str = input("Extract 1 frame every how many seconds? [5]: ").strip()
    interval_sec = float(int_str) if int_str else 5.0

    # ---- 4. Sanity-check inputs ----
    if not Path(rgb_video).is_file():
        raise SystemExit(f"RGB video not found: {rgb_video}")
    if not Path(th_video).is_file():
        raise SystemExit(f"Thermal video not found: {th_video}")
    if rad_kind == "1" and not Path(rad_path).is_file():
        raise SystemExit(f"Radiometric .npy not found: {rad_path}")
    if rad_kind == "2" and not Path(rad_path).is_dir():
        raise SystemExit(f"Radiometric folder not found: {rad_path}")

    # ---- 5. Build the patient-specific output folder ----
    OUT = Path(out_root) / patient_id
    if OUT.exists() and any(OUT.iterdir()):
        ans = input(
            f"\nOutput folder already exists and is not empty:\n  {OUT}\n"
            f"Overwrite contents? [y/N]: "
        ).strip().lower()
        if ans != "y":
            raise SystemExit("Aborted — choose a different patient ID or output root.")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 6. Auto-derive radiometric fps from npy-count ÷ thermal-duration ----
    th_n, th_fps, th_dur = video_info(th_video)
    n_npy = count_npy_frames(rad_path, rad_kind)
    if th_dur is None:
        raise SystemExit("Cannot read thermal video fps — cannot compute radiometric fps.")
    rad_fps = n_npy / th_dur

    print(f"\n[cfg ] patient_id  = {patient_id}")
    print(f"        output root = {OUT}")
    print(f"        interval    = {interval_sec}s")
    print(f"        thermal video     : {th_n} frames @ {th_fps:.2f} fps = {th_dur:.2f}s")
    print(f"        radiometric       : {n_npy} frames")
    print(f"        derived rad_fps   : {rad_fps:.2f}")
    if abs(n_npy - th_n) / th_n > 0.05:
        print(f"        WARNING: radiometric ({n_npy}) != thermal ({th_n}) frame count — "
              f"assuming alignment by duration, not by index.")

    # ---- 7. Extract (same interval on all three) ----
    n_rgb = extract_video_frames(rgb_video, OUT / "rgb",         prefix="rgb",
                                 interval_sec=interval_sec)
    n_th  = extract_video_frames(th_video,  OUT / "thermal_vis", prefix="th",
                                 interval_sec=interval_sec)

    if rad_kind == "1":
        n_rad = extract_npy_stacked(rad_path, OUT / "thermal_rad", prefix="th",
                                    interval_sec=interval_sec, rad_fps=rad_fps)
    else:
        n_rad = extract_npy_folder(rad_path,  OUT / "thermal_rad", prefix="th",
                                   interval_sec=interval_sec, rad_fps=rad_fps)

    # ---- 8. Report ----
    print(f"\nFrame counts -> rgb={n_rgb}  thermal_video={n_th}  radiometric={n_rad}")
    if n_rgb != n_rad:
        print("WARNING: rgb vs radiometric counts differ — check alignment.")
    print(f"Saved to: {OUT}")
    print("Done.")