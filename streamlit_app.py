"""
Hand Temperature Analyzer - Streamlit front end.

  Section 1  Live hand analysis and tracking   (status, stages, live tracking picture)
  Section 2  Results and downloads             (summary table, plots, quality, ZIP)

The existing scripts in /pipeline run unchanged. core.py answers their terminal questions and
passes every output path to the next script, so nobody types a path.
"""
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections import deque
from pathlib import Path

# Writable locations (cloud servers have a read-only or temporary app folder)
_BASE = Path(os.environ.get("HTA_DATA", Path(tempfile.gettempdir()) / "hta_data"))
os.environ.setdefault("DATA_DIR", str(_BASE))
RUNS = _BASE / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

import pandas as pd            # noqa: E402
import streamlit as st         # noqa: E402

import core                    # noqa: E402
from core import Job, STAGES, SAVED_RIG   # noqa: E402

core.CAL_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Hand Temperature Analyzer", page_icon=None, layout="wide")

PURGE_GRACE_S = 30      # seconds kept after a download click so the browser can finish receiving the file
ABANDON_S = float(os.environ.get("HTA_ABANDON_SECONDS", "90"))   # no browser contact for this long -> delete
STAGE_NAME = dict(STAGES)
ICON = {"wait": "[ ]", "run": "[>]", "ok": "[x]", "skip": "[-]", "fail": "[!]"}


# ----------------------------------------------------------------------------
# Shared server state (one per server process, not per browser tab)
# ----------------------------------------------------------------------------
@st.cache_resource
def registry():
    reg = {"runs": {}, "lock": threading.Semaphore(1)}

    for d in RUNS.iterdir():                      # nothing survives a server restart
        shutil.rmtree(d, ignore_errors=True)

    def cleaner():
        while True:
            try:
                now = time.time()
                for rid_, r in list(reg["runs"].items()):
                    if (r.get("purge_at") and now > r["purge_at"]) or \
                            now - r.get("seen", r["created"]) > ABANDON_S:     # browser closed / refreshed
                        if r.get("job") and r["state"] in ("queued", "running"):
                            r["state"] = "cancelled"
                            try:
                                r["job"].stop()
                            except Exception:
                                pass
                            time.sleep(3)
                        shutil.rmtree(RUNS / rid_, ignore_errors=True)
                        reg["runs"].pop(rid_, None)
                for d in RUNS.iterdir():
                    if d.name not in reg["runs"]:
                        shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
            time.sleep(15)
    threading.Thread(target=cleaner, daemon=True).start()
    return reg


REG = registry()


def new_run(rid, pid):
    return {"id": rid, "pid": pid, "state": "queued", "created": time.time(), "error": "",
            "stages": {k: {"status": "wait", "msg": ""} for k, _ in STAGES}, "frac": {},
            "log": deque(maxlen=200), "job": None, "evaluate": False, "live": False}


def make_emit(run):
    def emit(kind, *a):
        if kind == "log":
            run["log"].append(a[0])
        elif kind == "stage":
            key, status, msg = a
            run["stages"][key] = {"status": status, "msg": msg}
            run["frac"].pop(key, None) if status in ("ok", "skip", "fail") else None
        elif kind == "progress":
            run["frac"][a[0]] = a[1]
            run["stages"][a[0]]["msg"] = "frame " + a[2]
        elif kind == "done":
            run["state"] = "done"
        elif kind == "error":
            run["state"], run["error"] = "error", a[0]
        elif kind == "cancelled":
            run["state"] = "cancelled"
    return emit


def overall(run):
    keys = [k for k, _ in STAGES if not (k == "eval" and not run["evaluate"])]
    done = sum(1 for k in keys if run["stages"][k]["status"] in ("ok", "skip", "fail"))
    cur = sum(run["frac"].get(k, 0.0) for k in keys if run["stages"][k]["status"] == "run")
    return min(1.0, (done + cur) / max(1, len(keys)))


# ----------------------------------------------------------------------------
# Optional access password (set APP_PASSWORD in the app's Secrets / environment)
# ----------------------------------------------------------------------------
def gate():
    pw = os.environ.get("APP_PASSWORD", "")
    try:
        pw = pw or st.secrets.get("APP_PASSWORD", "")
    except Exception:
        pass
    if not pw or st.session_state.get("authed"):
        return
    st.title("Hand Temperature Analyzer")
    entered = st.text_input("Access password", type="password")
    if entered:
        if hashlib.sha256(entered.encode()).digest() == hashlib.sha256(pw.encode()).digest():
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
def save_upload(up, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    up.seek(0)
    with open(dest, "wb") as f:
        shutil.copyfileobj(up, f, 8 * 1024 * 1024)


def show_form():
    st.title("Hand Temperature Analyzer")
    st.caption("Upload the recordings, press Start, and follow the analysis live. "
               "Results appear below when processing is complete.")

    st.subheader("1. Patient and videos")
    c1, c2, c3 = st.columns(3)
    pid = c1.text_input("Patient / participant ID", help="Letters, digits, - and _ only. Used as the folder name.")
    rgb = c2.file_uploader("RGB video", type=["mp4", "avi", "mov", "mkv", "wmv"])
    thermal = c3.file_uploader("Thermal video", type=["mp4", "avi", "mov", "mkv", "wmv"])

    st.subheader("2. Radiometric data")
    kind = st.radio("Radiometric data is", ["One .npy file (all frames stacked)", "Several .npy files (one per frame)"],
                    horizontal=True)
    rad = st.file_uploader("Radiometric .npy", type=["npy"], accept_multiple_files=kind.startswith("Several"))
    interval = st.number_input("Take one frame every (seconds)", min_value=0.1, value=5.0, step=0.5,
                               help="The same interval is used for the RGB video, the thermal video and the radiometric data.")

    st.subheader("3. Camera calibration")
    opts = ["Calibrate from this recording (about a minute)", "Upload a calibration file (rig_calibration.json)"]
    calib = st.radio("Calibration", opts, label_visibility="collapsed")
    rig_up = None
    if calib.startswith("Upload"):
        rig_up = st.file_uploader("rig_calibration.json", type=["json"])
    st.caption("A calibration belongs to a fixed camera setup. After an analysis you can download "
               "rig_calibration.json and upload it next time to skip this step.")

    with st.expander("Advanced options"):
        a1, a2 = st.columns(2)
        with a1:
            detect = st.selectbox("Detect hands on", ["RGB frames (default)", "Thermal visual frames"])
            swap = st.checkbox("Swap Left/Right labels", help="Tick if the hand label is reversed in the annotated frames "
                                                              "(MediaPipe assumes a mirrored image).")
            roi = st.text_input("Detection area x,y,w,h (empty = whole frame)")
            local_realign = st.checkbox("Per-frame local re-alignment", value=True)
            snap = st.checkbox("Snap off-hand joints to the nearest warm pixel", value=True)
            units = st.selectbox("Radiometric units", ["auto", "celsius", "kelvin", "decikelvin", "centikelvin"])
        with a2:
            calib_frames = st.text_input("Frames used for calibration ('all' or a number)", value="all")
            orientation = st.selectbox("Thermal image orientation if it cannot be detected (0-7; 4 = transpose)",
                                       list(range(8)), index=4)
            fov = st.text_input("RGB,thermal horizontal field of view in degrees (optional, e.g. 60,45)")
            b1, b2 = st.columns(2)
            baseline = b1.number_input("Camera spacing (mm)", value=10.0, help="Used only if a field of view is given.")
            dist = b2.number_input("Distance to hand (mm)", value=310.0, help="Used only if a field of view is given.")
            save_default = False
            live = st.checkbox("Show live hand-tracking view", value=True,
                               help="Runs hand_analysis_system_2.py alongside the analysis. Uses extra CPU and memory.")
            evaluate = st.checkbox("Also compute quality / performance metrics")
        if evaluate:
            st.caption("Optional accuracy checks (leave empty to skip):")
            g1, g2, g3 = st.columns(3)
            gt_up = g1.file_uploader("Ground-truth clicks CSV", type=["csv"])
            ref_up = g2.file_uploader("Reference temperatures CSV", type=["csv"])
            mm_px = g3.text_input("Millimetres per thermal pixel")
        else:
            gt_up = ref_up = None
            mm_px = ""

    st.info("Nothing is stored. Your recordings and results exist only while this page is open and are deleted "
            "from the server when you close or refresh it. Download what you need before leaving. "
            "Do not upload identifiable patient data unless you are permitted to.")

    errors = []
    if not pid.strip() or not all(ch.isalnum() or ch in "-_" for ch in pid.strip()):
        errors.append("Patient ID is required (letters, digits, - and _ only).")
    if rgb is None:
        errors.append("Upload the RGB video.")
    if thermal is None:
        errors.append("Upload the thermal video.")
    if not rad:
        errors.append("Upload the radiometric .npy data.")
    if calib.startswith("Upload") and rig_up is None:
        errors.append("Upload the calibration file or choose another calibration option.")
    ct = calib_frames.strip().lower()
    if ct != "all" and not ct.isdigit():
        errors.append("Frames used for calibration must be 'all' or a number.")
    if fov.strip():
        try:
            [float(v) for v in fov.split(",")][1]
        except Exception:
            errors.append("Field of view must look like 60,45.")
    if roi.strip():
        try:
            if len([int(v) for v in roi.replace(" ", "").split(",")]) != 4:
                raise ValueError
        except ValueError:
            errors.append("Detection area must be four whole numbers: x,y,w,h.")
    for e in errors:
        st.caption(":red[" + e + "]")

    if st.button("Start analysis", type="primary", disabled=bool(errors)):
        rid = uuid.uuid4().hex[:12]
        rdir = RUNS / rid
        up = rdir / "uploads"
        with st.spinner("Saving uploaded files..."):
            ext = lambda f: Path(f.name).suffix.lower() or ".mp4"     # noqa: E731
            save_upload(rgb, up / ("rgb_video" + ext(rgb)))
            save_upload(thermal, up / ("thermal_video" + ext(thermal)))
            if kind.startswith("One"):
                save_upload(rad, up / "rad_stack.npy")
                rad_path, is_file = up / "rad_stack.npy", True
            else:
                for f in rad:
                    save_upload(f, up / "rad_frames" / Path(f.name).name)
                rad_path, is_file = up / "rad_frames", False
            mode = "new"
            rig_file = ""
            if calib.startswith("Use"):
                mode = "saved"
            elif calib.startswith("Upload"):
                save_upload(rig_up, up / "rig_calibration.json")
                try:
                    json.loads((up / "rig_calibration.json").read_text())
                except Exception:
                    st.error("That calibration file is not valid JSON."); st.stop()
                mode, rig_file = "file", str(up / "rig_calibration.json")
            gt = ref = ""
            if gt_up is not None:
                save_upload(gt_up, up / "ground_truth_thermal.csv"); gt = str(up / "ground_truth_thermal.csv")
            if ref_up is not None:
                save_upload(ref_up, up / "reference_temperatures.csv"); ref = str(up / "reference_temperatures.csv")
        params = dict(
            pid=pid.strip(), rgb=str(next(up.glob("rgb_video.*"))), thermal=str(next(up.glob("thermal_video.*"))),
            rad=str(rad_path), rad_is_file=is_file, run_dir=str(rdir / "run"), interval=str(interval),
            detect_on="2" if detect.startswith("Thermal") else "1", swap=swap, roi=roi.strip(),
            calib_mode=mode, rig_file=rig_file, fov=fov.strip(), baseline_mm=baseline, dist_mm=dist,
            calib_frames=ct, orientation=orientation, save_default=save_default,
            local_realign=local_realign, snap=snap, units=units, live=live, evaluate=evaluate,
            gt_csv=gt, ref_csv=ref, mm_per_px=mm_px.strip())
        run = new_run(rid, pid.strip())
        run["evaluate"], run["live"] = evaluate, live
        REG["runs"][rid] = run
        job = Job(params, make_emit(run))
        run["job"] = job

        def work():
            with REG["lock"]:
                if run["state"] == "cancelled":
                    return
                run["state"] = "running"
                job.run()
                shutil.rmtree(up, ignore_errors=True)       # originals are no longer needed
            (RUNS / rid).touch()
        threading.Thread(target=work, daemon=True).start()
        st.session_state["rid"] = rid
        st.rerun()


# ----------------------------------------------------------------------------
# Section 1: live tracking
# ----------------------------------------------------------------------------
def tracking_status(run):
    job = run["job"]
    if run["state"] == "error":
        return "error", "Error or tracking failure"
    if run["state"] == "cancelled":
        return "error", "Stopped"
    if run["state"] == "done":
        return "ok", "Processing completed"
    if run["state"] == "queued":
        return "wait", "Initializing (waiting for the server)"
    det = run["stages"]["detect"]["status"]
    if det == "wait":
        return "wait", "Initializing"
    if det == "run" or (job and job.live_status() == "tracking"):
        return "run", "Tracking in progress"
    return "run", "Processing"


def latest_frame(run):
    """Newest live-tracking picture: the live view if running, else the newest annotated frame."""
    rid = run["id"]
    job = run["job"]
    if job is not None and job.live_frame is not None and Path(job.live_frame).is_file():
        try:
            return Path(job.live_frame).read_bytes(), "Live hand tracking"
        except OSError:
            pass
    ann = RUNS / rid / "run" / "analysis"
    if ann.is_dir():
        imgs = [f for f in ann.rglob("annotated/*") if f.suffix.lower() in (".png", ".jpg")]
        if imgs:
            f = max(imgs, key=lambda x: x.stat().st_mtime)
            try:
                return f.read_bytes(), "Latest analysed frame"
            except OSError:
                pass
    return None, ""


def draw_live(run, with_log=False):
    kind, text = tracking_status(run)
    if kind == "ok":
        st.success(text)
    elif kind == "error":
        st.error(text)
    else:
        st.info(text)
    st.progress(overall(run))
    left, right = st.columns([2, 3])
    with left:
        lines = []
        for k, name in STAGES:
            if k == "eval" and not run["evaluate"]:
                continue
            s = run["stages"][k]
            lines.append("%s  %s  %s" % (ICON[s["status"]], name, ("- " + s["msg"]) if s["msg"] else ""))
        st.code("\n".join(lines), language=None)
        if run["state"] == "queued":
            ahead = sum(1 for r in REG["runs"].values() if r["state"] == "queued" and r["created"] < run["created"])
            st.caption("Another analysis is running; %d ahead of yours." % ahead)
    with right:
        img, cap = latest_frame(run)
        if img:
            st.image(img, caption=cap, width="stretch")
        else:
            st.caption("The tracking picture appears here once hand detection starts.")
    if with_log:
        with st.expander("Details"):
            st.code("\n".join(list(run["log"])[-25:]) or "...", language=None)


@st.fragment(run_every=1.0)
def live_poll(rid):
    """Refreshes itself every second while the analysis runs; a full rerun shows the final state."""
    run = REG["runs"].get(rid)
    if run is None:
        st.rerun()
        return
    run["seen"] = time.time()
    draw_live(run, with_log=True)
    if run["state"] not in ("queued", "running"):
        st.rerun()


def show_run(run):
    st.title("Hand Temperature Analyzer")
    st.caption("Patient: %s   |   Run ID: %s" % (run["pid"], run["id"]))
    st.header("Section 1 - Live hand analysis and tracking")
    if run["state"] in ("queued", "running"):
        live_poll(run["id"])
        if st.button("Stop analysis"):
            if run["job"]:
                run["job"].stop()
            run["state"] = "cancelled"
            st.rerun()
        return
    draw_live(run)
    if run["state"] == "error":
        st.error(run["error"])
        with st.expander("Details"):
            st.code("\n".join(run["log"]), language=None)
        if st.button("Back to the form", type="primary"):
            st.session_state.pop("rid", None); st.rerun()
        return
    if run["state"] == "cancelled":
        if st.button("Back to the form", type="primary"):
            st.session_state.pop("rid", None); st.rerun()
        return
    show_results(run["id"], run["pid"])


# ----------------------------------------------------------------------------
# Section 2: results and downloads
# ----------------------------------------------------------------------------
def make_zip(rdir, scope):
    out = rdir / ("outputs_%s.zip" % scope)
    root = rdir / "run" / ("results" if scope == "key" else "")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in root.rglob("*"):
            if f.is_file() and f.name != out.name:
                z.write(f, f.relative_to(rdir / "run"))
    return out


@st.fragment(run_every=5)
def heartbeat(rid):
    r = REG["runs"].get(rid)
    if r is None:                                  # data already deleted -> back to the empty form
        st.session_state.pop("rid", None)
        st.rerun()
        return
    r["seen"] = time.time()


def schedule_purge(rid):
    r = REG["runs"].get(rid)
    if r is not None and not r.get("purge_at"):
        r["purge_at"] = time.time() + PURGE_GRACE_S


def show_results(rid, pid):
    rdir = RUNS / rid
    res = rdir / "run" / "results"
    qf = res / "quality_summary.json"
    st.header("Section 2 - Results and downloads")
    if not qf.is_file():
        st.error("Results are not available (they may have expired and been deleted).")
        if st.button("Start a new analysis"):
            st.session_state.pop("rid", None); st.rerun()
        return
    qc = json.loads(qf.read_text(encoding="utf-8"))

    m = st.columns(5)
    m[0].metric("Frames with a hand", "%s of %s" % (qc.get("frames", "?"), qc.get("frames_in_video", "?")))
    m[1].metric("Joints with a temperature", "%s %%" % qc.get("measured_pct", "?"))
    m[2].metric("Hand/thermal overlap (Dice)", qc.get("dice_final", "-"),
                help="1 = perfect. Before the per-frame fix: %s" % qc.get("dice_global", "-"))
    m[3].metric("Median temperature", "%s C" % qc.get("temp_median", "-"))
    m[4].metric("Temperature range", "%s to %s C" % (qc.get("temp_min", "-"), qc.get("temp_max", "-")))
    for w in qc.get("warnings", []):
        st.warning(w)
    if not qc.get("warnings"):
        st.success("Quality check passed.")

    reports = sorted(res.glob("*hand_temperature_report"))
    if not reports:
        st.error("No report was produced.")
    else:
        names = [r.name.replace("_temperature_report", "") for r in reports]
        tag = names[0] if len(names) == 1 else st.radio("Hand", names, horizontal=True)
        d = res / (tag + "_temperature_report")
        tabs = st.tabs(["Summary table", "Temperature vs frame", "Heatmap", "Joint locations",
                        "Alignment check", "Evaluation metrics"])
        sm = d / (tag + "_joint_temperature_summary.csv")
        df = pd.read_csv(sm) if sm.is_file() else pd.DataFrame()
        with tabs[0]:
            if not df.empty:
                show = df[["joint", "max_temp_c", "min_temp_c", "mean_temp_c", "median_temp_c",
                           "std_dev_temp_c", "variance_temp_c", "frames_with_reading", "frames_total"]].rename(columns={
                    "joint": "Joint", "max_temp_c": "Max (C)", "min_temp_c": "Min (C)", "mean_temp_c": "Mean (C)",
                    "median_temp_c": "Median (C)", "std_dev_temp_c": "Std dev", "variance_temp_c": "Variance",
                    "frames_with_reading": "Frames read", "frames_total": "Frames"})
                st.dataframe(show, hide_index=True, width="stretch", height=700)
        with tabs[1]:
            p = d / (tag + "_temperature_vs_frame.png")
            st.image(str(p), width="stretch") if p.is_file() else st.caption("No plot.")
        with tabs[2]:
            p = d / (tag + "_heatmap.png")
            st.image(str(p), width="stretch") if p.is_file() else st.caption("No heatmap.")
        with tabs[3]:
            cols = [c for c in ["joint", "mean_x_rgb", "mean_y_rgb", "mean_x_thermal", "mean_y_thermal",
                                "frame_at_max", "x_thermal_at_max", "y_thermal_at_max"] if c in df.columns]
            if cols:
                st.dataframe(df[cols], hide_index=True, width="stretch", height=700)
        with tabs[4]:
            imgs = sorted((res / "alignment_check").glob("*.png"))
            if imgs:
                st.caption("White dots = mapped joints, green outline = warm hand region. The dots should sit on the hand.")
                cols = st.columns(3)
                for i, f in enumerate(imgs[:6]):
                    cols[i % 3].image(str(f), width="stretch")
            else:
                st.caption("Calibration pictures are only produced when you calibrate from the recording.")
        with tabs[5]:
            mt = res / "evaluation_report" / "metrics.csv"
            if mt.is_file():
                st.dataframe(pd.read_csv(mt), hide_index=True, width="stretch")
                for f in sorted((res / "evaluation_report").glob("*.png")):
                    st.image(str(f), width="stretch")
            else:
                st.caption("Not computed. Tick 'Also compute quality / performance metrics' under Advanced options.")

    st.subheader("Download")
    c1, c2 = st.columns(2)
    for col, scope, label, help_ in ((c1, "all", "Prepare ZIP of all outputs",
                                      "Every file the pipeline produced: frames, joints, radiometric CSV, reports, logs."),
                                     (c2, "key", "Prepare ZIP of key results only",
                                      "Reports, summary CSVs, plots, quality summary and calibration (small).")):
        with col:
            if st.button(label, help=help_, key="prep_" + scope):
                with st.spinner("Creating ZIP..."):
                    st.session_state["zip_" + scope] = str(make_zip(rdir, scope))
            zp = st.session_state.get("zip_" + scope)
            if zp and Path(zp).is_file() and rid in zp:
                with open(zp, "rb") as f:
                    st.download_button("Download " + ("all outputs" if scope == "all" else "key results") + " (ZIP)",
                                       f, file_name="%s_%s.zip" % (pid, scope), mime="application/zip",
                                       key="dl_" + scope, on_click=schedule_purge, args=(rid,))
    rig = res / "rig_calibration.json"
    if rig.is_file():
        st.download_button("Download calibration (rig_calibration.json)", rig.read_bytes(),
                           file_name="rig_calibration.json", key="dl_rig",
                           on_click=schedule_purge, args=(rid,))
    run_ = REG["runs"].get(rid)
    if run_ and run_.get("purge_at"):
        st.warning("Download started. Your data is being deleted from the server now. "
                   "To download a different file, run the analysis again.")
    else:
        st.caption("Results are not stored. Choose a download, or finish without downloading. "
                   "In both cases your data is deleted from the server afterwards.")
    heartbeat(rid)
    if st.button("Finish without downloading (delete my data now)"):
        shutil.rmtree(rdir, ignore_errors=True)
        REG["runs"].pop(rid, None)
        st.session_state.pop("rid", None); st.rerun()


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
def main():
    gate()
    rid = st.session_state.get("rid")
    if rid:
        run = REG["runs"].get(rid)
        if run is not None:
            run["seen"] = time.time()
            return show_run(run)
        st.session_state.pop("rid", None)
    show_form()


main()
