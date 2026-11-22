"""
core.py - pipeline runner used by the web server.
Runs the unchanged scripts in /pipeline one after another and hands each script's output
path to the next (their questions are answered automatically).
Also usable as the script launcher:  python core.py --run <script>
"""
import codecs
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import threading
import time
import queue
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
FROZEN = getattr(sys, "frozen", False)
APP_FILE = Path(sys.executable if FROZEN else __file__).resolve()
RES_DIR = Path(getattr(sys, "_MEIPASS", APP_FILE.parent))      # bundled, read-only
DATA_DIR = Path(os.environ.get("DATA_DIR", str(APP_FILE.parent)))   # writable
PIPE = RES_DIR / "pipeline"
MODEL = RES_DIR / "models" / "hand_landmarker.task"
CAL_DIR = DATA_DIR / "calibration"
SAVED_RIG = CAL_DIR / "rig_calibration.json"
DEFAULT_RUNS = DATA_DIR / "runs"
SETTINGS = DATA_DIR / "settings.json"


# ----------------------------------------------------------------------------
# Script-runner mode: "python app.py --run <script> " executes one pipeline script.
# (Used for every stage so it also works when the app is packaged as an .exe.)
# ----------------------------------------------------------------------------
if len(sys.argv) >= 3 and sys.argv[1] == "--run":
    _script = sys.argv[2]
    sys.argv = [_script] + sys.argv[3:]
    runpy.run_path(_script, run_name="__main__")
    sys.exit(0)

import numpy as np          # noqa: E402  (after runner mode on purpose)
import pandas as pd         # noqa: E402

STAGES = [("split", "Extract frames"),
          ("detect", "Detect hand joints"),
          ("calib", "Camera calibration"),
          ("temp", "Read temperatures"),
          ("report", "Build report"),
          ("eval", "Quality metrics")]
STAGE_KEYS = [k for k, _ in STAGES]
ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]|processed (\d+)/(\d+)")


class Cancelled(Exception):
    pass


class StageError(Exception):
    pass


def load_settings():
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:
        return {}


def save_settings(d):
    try:
        SETTINGS.write_text(json.dumps(d, indent=2))
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Quality summary computed from the final CSV
# ----------------------------------------------------------------------------
def compute_qc(radio_csv, rig_json=None):
    df = pd.read_csv(radio_csv)
    qc = {"rows": int(len(df)), "frames": int(df["frame"].nunique()),
          "hands": sorted(str(h) for h in df["handedness"].unique()), "warnings": []}
    rm = df["reading_method"] if "reading_method" in df else pd.Series("direct", index=df.index)
    for m in ("direct", "snapped", "none"):
        qc[m + "_pct"] = round(100.0 * float((rm == m).mean()), 1)
    qc["measured_pct"] = round(qc["direct_pct"] + qc["snapped_pct"], 1)
    if "align_dice" in df:
        qc["dice_global"] = round(float(df["align_dice_global"].median()), 3)
        qc["dice_final"] = round(float(df["align_dice"].median()), 3)
    if "temperature_c" in df and df["temperature_c"].notna().any():
        t = df["temperature_c"].dropna()
        qc["temp_min"], qc["temp_median"], qc["temp_max"] = (round(float(t.min()), 1),
                                                              round(float(t.median()), 1),
                                                              round(float(t.max()), 1))
        if not (20 <= qc["temp_median"] <= 45):
            qc["warnings"].append("Median temperature is outside 20-45 C: check the radiometric units.")
    else:
        qc["warnings"].append("No temperature values were produced.")
    if qc["measured_pct"] < 70:
        qc["warnings"].append("Only %.0f%% of joints have a temperature reading (target 70%% or more)."
                              % qc["measured_pct"])
    if qc.get("dice_final") is not None and qc["dice_final"] < 0.6:
        qc["warnings"].append("Hand/thermal overlap is low (Dice %.2f). The camera calibration may not "
                              "fit this recording; check the alignment_check images." % qc["dice_final"])
    if rig_json and Path(rig_json).is_file():
        try:
            qc["calibration_dice"] = round(float(json.loads(Path(rig_json).read_text())["mean_dice"]), 3)
        except Exception:
            pass
    return qc


# ----------------------------------------------------------------------------
# The pipeline job (no GUI code in here)
# ----------------------------------------------------------------------------
class Job:
    """emit(kind, *args): kinds = log, stage, progress, done, error, cancelled"""

    def __init__(self, params, emit):
        self.p = dict(params)
        for k in ("rgb", "thermal", "rad", "out_root", "run_dir", "rig_file", "gt_csv", "ref_csv"):   # absolute paths
            if self.p.get(k):
                self.p[k] = str(Path(self.p[k]).expanduser().resolve())
        self.emit = emit
        self.proc = None
        self.live = None
        self.live_frame = None
        self.cancel = threading.Event()

    # ---- helpers -------------------------------------------------------
    def log(self, text):
        self.emit("log", text)

    def stage(self, key, status, msg=""):
        if status == "run":
            self.current = key
        self.emit("stage", key, status, msg)

    def _env(self):
        env = dict(os.environ)
        env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                   HAND_MODEL_PATH=str(MODEL), MPLBACKEND="Agg")
        return env

    def _cmd(self, script):
        exe = sys.executable
        if exe.lower().endswith("pythonw.exe"):
            exe = exe[:-len("pythonw.exe")] + "python.exe"
        if FROZEN:
            return [exe, "--run", str(script)]
        return [exe, str(APP_FILE), "--run", str(script)]

    def _flags(self):
        return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0

    def stop(self):
        self.cancel.set()
        for pr in (self.proc, self.live):
            try:
                if pr and pr.poll() is None:
                    pr.kill()
            except Exception:
                pass

    def run_script(self, key, script, answers, cwd=None, logfile=None):
        """Run one pipeline script, answering its questions from `answers`."""
        if self.cancel.is_set():
            raise Cancelled()
        self.proc = subprocess.Popen(
            self._cmd(PIPE / script), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=self._env(), cwd=str(cwd) if cwd else None,
            bufsize=0, creationflags=self._flags())
        try:
            self.proc.stdin.write(("\n".join(str(a) for a in answers) + "\n").encode("utf-8"))
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        buf, full = "", []
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                break
            buf += dec.decode(chunk).replace("\r\n", "\n")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                self._line(key, line, full)
        if buf.strip():
            self._line(key, buf, full)
        rc = self.proc.wait()
        text = "\n".join(full)
        if logfile:
            Path(logfile).write_text(text, encoding="utf-8")
        if self.cancel.is_set():
            raise Cancelled()
        return rc, text

    def _line(self, key, line, full):
        segs = line.split("\r")
        for s in segs:
            m = PROGRESS_RE.search(s)
            if m:
                a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
                if int(b) > 0:
                    self.emit("progress", key, int(a) / int(b), "%s of %s" % (a, b))
        last = segs[-1]
        full.append(last)
        self.log(last)

    def need(self, rc, text, key, what):
        if rc != 0:
            tail = [l for l in text.splitlines() if l.strip()][-8:]
            raise StageError("%s failed.\n\n%s" % (what, "\n".join(tail)))

    # ---- live tracking window -----------------------------------------
    def start_live(self, run_dir):
        """Run hand_analysis_system_2.py unchanged; live_shim.py turns its cv2.imshow() into a JPEG file."""
        try:
            live_dir = run_dir / "live_monitor"
            live_dir.mkdir(exist_ok=True)
            self.live_frame = live_dir / "latest.jpg"
            env = self._env()
            env["HTA_LIVE_FRAME"] = str(self.live_frame)
            env["HTA_LIVE_SCRIPT"] = str(PIPE / "hand_analysis_system_2.py")
            lf = open(run_dir / "logs" / "live_monitor.txt", "wb")
            self.live = subprocess.Popen(
                self._cmd(RES_DIR / "live_shim.py"), stdin=subprocess.PIPE,
                stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(live_dir),
                creationflags=self._flags())
            self.live.stdin.write(("\n".join([self.p["pid"], "2", self.p["rgb"]]) + "\n").encode("utf-8"))
            self.live.stdin.close()
            self.log("Live hand-tracking started.")
        except Exception as e:                                      # never block the pipeline
            self.log("Live tracking could not be started: %s" % e)
            self.live = None

    def live_status(self):
        """not_started | tracking | finished | failed"""
        if self.live is None:
            return "not_started"
        rc = self.live.poll()
        return "tracking" if rc is None else ("finished" if rc == 0 else "failed")

    def stop_live(self):
        try:
            if self.live and self.live.poll() is None:
                self.live.terminate()
        except Exception:
            pass

    # ---- main ----------------------------------------------------------
    def run(self):
        p = self.p
        try:
            if not MODEL.is_file():
                raise StageError("Model file missing: %s" % MODEL)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(p["run_dir"]) if p.get("run_dir") else Path(p["out_root"]) / ("%s_%s" % (p["pid"], ts))
            for sub in ("frames", "analysis", "logs", "results"):
                (run_dir / sub).mkdir(parents=True, exist_ok=True)
            (run_dir / "run_info.json").write_text(json.dumps(
                {"started": ts, "parameters": p}, indent=2), encoding="utf-8")
            self.run_dir = run_dir
            self.log("Run folder: %s" % run_dir)
            if p.get("live"):
                self.start_live(run_dir)
            try:
                self._pipeline(run_dir)
            finally:
                self.stop_live()
            self.emit("done", str(run_dir))
        except Cancelled:
            self.emit("cancelled")
        except StageError as e:
            if getattr(self, "current", None):
                self.emit("stage", self.current, "fail", "")
            self.emit("error", str(e))
        except Exception as e:                                       # unexpected
            self.emit("error", "Unexpected error: %r" % (e,))

    def _pipeline(self, run_dir):
        p = self.p
        logs = run_dir / "logs"
        frames_root, analysis = run_dir / "frames", run_dir / "analysis"
        pid = p["pid"]

        # 1 ---- split --------------------------------------------------
        self.stage("split", "run")
        ans = [pid, p["rgb"], p["thermal"], "1" if p["rad_is_file"] else "2", p["rad"],
               frames_root, p["interval"]]
        rc, txt = self.run_script("split", "1_split_frames.py", ans, logfile=logs / "1_split_frames.txt")
        self.need(rc, txt, "split", "Frame extraction")
        patient_dir = frames_root / pid
        n_rgb = len(list((patient_dir / "rgb").glob("*")))
        n_rad = len(list((patient_dir / "thermal_rad").glob("*.npy")))
        if n_rgb == 0 or n_rad == 0:
            raise StageError("Frame extraction produced no frames (rgb=%d, radiometric=%d)." % (n_rgb, n_rad))
        self.stage("split", "ok", "%d frames" % n_rgb)

        # 2 ---- detect --------------------------------------------------
        self.stage("detect", "run")
        ans = ["y" if p.get("swap") else "n", pid, patient_dir, str(p.get("detect_on", "1")), p["interval"],
               p.get("roi") or "", analysis]
        rc, txt = self.run_script("detect", "2_hand_analysis_frames.py", ans,
                                  logfile=logs / "2_hand_analysis_frames.txt")
        self.need(rc, txt, "detect", "Hand detection")
        found = sorted(analysis.rglob("joint_coordinates.csv"), key=lambda f: f.stat().st_mtime)
        if not found:
            raise StageError("Hand detection did not create joint_coordinates.csv.")
        csv_path = found[-1]
        session_dir = csv_path.parent
        jc = pd.read_csv(csv_path)
        if jc.empty:
            raise StageError("No hand was detected in any frame.\n\nCheck that the RGB video shows the "
                             "hand clearly, or that the ROI (if set) contains the hand.")
        nf, nh = jc["frame"].nunique(), jc["handedness"].nunique()
        self.stage("detect", "ok", "hand found in %d of %d frames" % (nf, n_rgb))

        # 3 ---- calibration --------------------------------------------
        rig = run_dir / "rig_calibration.json"
        if p["calib_mode"] == "file":
            if not Path(p.get("rig_file") or "").is_file():
                raise StageError("The uploaded calibration file was not found.")
            shutil.copy2(p["rig_file"], rig)
            self.stage("calib", "skip", "using uploaded calibration")
        elif p["calib_mode"] == "saved":
            if not SAVED_RIG.is_file():
                raise StageError("No saved camera calibration found. Choose 'Calibrate from this recording'.")
            shutil.copy2(SAVED_RIG, rig)
            self.stage("calib", "skip", "using saved calibration")
        else:
            self.stage("calib", "run")
            fov = (p.get("fov") or "").strip()
            ans = ([patient_dir, csv_path, rig, fov]
                   + ([str(p.get("baseline_mm", 10)), str(p.get("dist_mm", 310))] if fov else [])
                   + [str(p.get("calib_frames", "all")), str(p.get("orientation", 4))])
            rc, txt = self.run_script("calib", "3_calibrate_alignment.py", ans,
                                      logfile=logs / "3_calibrate_alignment.txt")
            self.need(rc, txt, "calib", "Camera calibration")
            if not rig.is_file():
                raise StageError("Calibration did not create rig_calibration.json.")
            if p.get("save_default", True):
                CAL_DIR.mkdir(exist_ok=True)
                shutil.copy2(rig, SAVED_RIG)
            self.stage("calib", "ok", "saved for future runs" if p.get("save_default", True) else "done")

        # 4 ---- temperatures -------------------------------------------
        self.stage("temp", "run")
        ans = [csv_path, patient_dir / "thermal_rad", rig,
               "Y" if p.get("local_realign", True) else "n", "Y" if p.get("snap", True) else "n",
               p.get("units", "auto"), "y"]
        rc, txt = self.run_script("temp", "4_add_radiometric.py", ans,
                                  logfile=logs / "4_add_radiometric.txt")
        self.need(rc, txt, "temp", "Temperature reading")
        radio_csv = session_dir / "joint_coordinates_radiometric.csv"
        if not radio_csv.is_file():
            raise StageError("Temperature stage did not create joint_coordinates_radiometric.csv.\n\n"
                             + "\n".join([l for l in txt.splitlines() if l.strip()][-6:]))
        qc = compute_qc(radio_csv, rig)
        if qc["measured_pct"] == 0:
            raise StageError("No temperature could be read at any joint. The camera calibration probably "
                             "does not match this recording (try 'Calibrate from this recording').")
        self.stage("temp", "ok", "%.0f%% of joints read" % qc["measured_pct"])

        # 5 ---- report --------------------------------------------------
        self.stage("report", "run")
        rc, txt = self.run_script("report", "5_joint_temperature_report.py", [radio_csv],
                                  logfile=logs / "5_joint_temperature_report.txt")
        self.need(rc, txt, "report", "Report")
        reports = sorted(session_dir.glob("*hand_temperature_report"))
        if not reports:
            raise StageError("The report stage created no report folder.")
        self.stage("report", "ok", ", ".join(r.name.split("_")[0] for r in reports))

        # 6 ---- optional evaluation ------------------------------------
        if p.get("evaluate"):
            self.stage("eval", "run")
            rc, txt = self.run_script("eval", "7_evaluate_pipeline.py",
                                      [radio_csv, patient_dir, rig, p.get("gt_csv") or "", p.get("ref_csv") or "",
                                       p.get("mm_per_px") or ""],
                                      logfile=logs / "7_evaluate_pipeline.txt")
            self.stage("eval", "ok" if rc == 0 else "fail", "" if rc == 0 else "skipped (see logs)")
        else:
            self.stage("eval", "skip", "not requested")

        # ---- collect the important results ----------------------------
        res = run_dir / "results"
        for r in reports:
            shutil.copytree(r, res / r.name, dirs_exist_ok=True)
        shutil.copy2(radio_csv, res / "joint_coordinates_radiometric.csv")
        shutil.copy2(rig, res / "rig_calibration.json")
        ev = session_dir / "evaluation_report"
        if ev.is_dir():
            shutil.copytree(ev, res / "evaluation_report", dirs_exist_ok=True)
        chk = patient_dir / "alignment_check"
        if chk.is_dir():
            shutil.copytree(chk, res / "alignment_check", dirs_exist_ok=True)
        qc["frames_in_video"] = n_rgb
        (res / "quality_summary.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        info["finished"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        info["session_dir"] = str(session_dir)
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


