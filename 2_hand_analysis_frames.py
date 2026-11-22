"""
COMPONENT 2: hand_analysis_frames.py   (standalone)
Input : a folder of images (e.g. ...\\<patient_id>\\rgb made by split_frames.py)
Output: <out>/<participant>/<session>/
          landmark_data.csv      -> manual input for analyze_landmarks.py
          joint_coordinates.csv  -> one row per frame/hand/joint (x, y, z, angle)
          annotated/             -> frames with skeleton drawn
Needs : pip install opencv-python mediapipe
Does NOT import hand_analysis_system.py; the needed functions are copied here.
Radiometric values are added by a separate component (add_radiometric.py).
"""
import csv, math, re
from datetime import datetime
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import mediapipe as mp

import os
# Model location: set by the app via HAND_MODEL_PATH, otherwise models/ next to this folder.
MODEL_PATH = os.environ.get("HAND_MODEL_PATH") or str(
    Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task")
OUTPUT_DIR = Path("hand_analysis_output")

MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.5
PRESENCE_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5

# ROI edge margin used for cautious visibility assessment.
ROI_EDGE_MARGIN_FRACTION = 0.06

# ---------------------------- LANDMARK MAP -----------------------------

# MediaPipe hand landmark indices:
# 0 wrist; thumb 1-4; index 5-8; middle 9-12;
# ring 13-16; pinky 17-20.
FINGER_LANDMARKS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "little": [17, 18, 19, 20],
}

FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "little": 20,
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ---------------------------- GEOMETRY HELPERS -------------------------

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_degrees(a, b, c):
    """Angle ABC in degrees. A straight finger is approximately 180 degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    denom = math.hypot(*ba) * math.hypot(*bc)
    if denom < 1e-8:
        return None
    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / denom
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def normalized_to_frame_points(landmarks, roi_rect):
    """Convert crop-normalized landmarks into original-frame pixel points."""
    roi_x, roi_y, roi_w, roi_h = roi_rect
    points = []
    for lm in landmarks:
        x = roi_x + lm.x * roi_w
        y = roi_y + lm.y * roi_h
        points.append((int(round(x)), int(round(y))))
    return points


# ---------------------------- FINGER ANALYSIS --------------------------

def analyze_fingers(points, frame_width, frame_height, roi_rect):
    """
    Estimate finger posture, joint angles, and image-boundary status.

    IMPORTANT: MediaPipe predicts landmarks even when parts are occluded.
    Therefore these values cannot prove that a finger is anatomically present
    or absent. "landmarks_within_frame" means only that predicted points lie
    within the image/ROI, not that the finger is visually unobstructed.
    """
    roi_x, roi_y, roi_w, roi_h = roi_rect
    margin_x = max(4, int(roi_w * ROI_EDGE_MARGIN_FRACTION))
    margin_y = max(4, int(roi_h * ROI_EDGE_MARGIN_FRACTION))
    palm_width = max(distance(points[5], points[17]), 1.0)
    result = {}

    for finger, ids in FINGER_LANDMARKS.items():
        finger_points = [points[i] for i in ids]
        outside = any(
            x < 0 or y < 0 or x >= frame_width or y >= frame_height
            for x, y in finger_points
        )
        near_roi_edge = any(
            x < roi_x + margin_x or x > roi_x + roi_w - margin_x
            or y < roi_y + margin_y or y > roi_y + roi_h - margin_y
            for x, y in finger_points
        )

        if outside:
            visibility = "outside_frame"
        elif near_roi_edge:
            visibility = "near_roi_boundary"
        else:
            visibility = "landmarks_within_frame"

        if finger == "thumb":
            # Thumb landmarks: CMC=1, MCP=2, IP=3, TIP=4.
            joint_angles = {
                "CMC": angle_degrees(points[0], points[1], points[2]),
                "MCP": angle_degrees(points[1], points[2], points[3]),
                "IP": angle_degrees(points[2], points[3], points[4]),
            }
            extension_angle = joint_angles["MCP"]
            reach = distance(points[4], points[5]) / palm_width
            extended = (
                extension_angle is not None
                and extension_angle > 145
                and reach > 0.55
            )
        else:
            mcp, pip, dip, tip = ids
            joint_angles = {
                "MCP": angle_degrees(points[0], points[mcp], points[pip]),
                "PIP": angle_degrees(points[mcp], points[pip], points[dip]),
                "DIP": angle_degrees(points[pip], points[dip], points[tip]),
            }
            extension_angle = joint_angles["PIP"]
            reach = distance(points[tip], points[mcp]) / palm_width
            extended = (
                extension_angle is not None
                and extension_angle > 150
                and reach > 0.75
            )

        if extension_angle is None:
            posture = "uncertain"
        else:
            posture = "extended" if extended else "folded_or_bent"

        result[finger] = {
            "visibility_assessment": visibility,
            "posture_estimate": posture,
            "joint_angles_degrees": {
                joint: round(value, 1) if value is not None else None
                for joint, value in joint_angles.items()
            },
            # Kept for compatibility with the live overlay/CSV.
            "joint_angle_degrees": (
                round(extension_angle, 1)
                if extension_angle is not None else None
            ),
            "tip_pixel": list(points[FINGER_TIPS[finger]]),
            "anatomical_status": "unknown_not_inferred_from_landmarks",
        }

    return result


def classify_gesture(finger_analysis):
    """Simple geometric gesture heuristic; not a trained classifier."""
    states = {
        name: item["posture_estimate"] == "extended"
        for name, item in finger_analysis.items()
    }

    if any(v == "uncertain" for v in
           [item["posture_estimate"] for item in finger_analysis.values()]):
        return "UNCERTAIN"

    thumb = states["thumb"]
    index = states["index"]
    middle = states["middle"]
    ring = states["ring"]
    little = states["little"]

    if all([thumb, index, middle, ring, little]):
        return "OPEN PALM"
    if not any([thumb, index, middle, ring, little]):
        return "FIST"
    if index and middle and not ring and not little:
        return "PEACE"
    if index and not middle and not ring and not little:
        return "POINTING"
    if thumb and not index and not middle and not ring and not little:
        return "THUMBS UP"

    return "OTHER / UNKNOWN"

# ---------------------------- COMPONENT MAIN ---------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ask(prompt, default=""):
    v = input(prompt).strip().strip('"').strip("'").strip()
    return v or default


def joint_names(finger):
    return (["CMC", "MCP", "IP", "TIP"] if finger == "thumb"
            else ["MCP", "PIP", "DIP", "TIP"])


SWAP_HANDEDNESS = False


def main():
    global SWAP_HANDEDNESS
    print('MediaPipe assumes a mirrored (selfie) image. For normal, non-mirrored camera frames')
    print('its Left/Right label is reversed. Check one annotated frame and answer accordingly.')
    SWAP_HANDEDNESS = ask('Swap Left/Right labels? (y/n)', 'n').lower().startswith('y')
    participant = ask("Participant name or ID: ")
    if not participant:
        raise SystemExit("Participant cannot be empty.")
    # Patient folder written by split_frames.py, containing:
    #   rgb/  thermal_vis/  thermal_rad/
    patient_dir = Path(ask("Path to PATIENT frames folder (contains rgb, thermal_vis, thermal_rad): "))
    if not patient_dir.is_dir():
        raise SystemExit(f"Folder not found: {patient_dir}")
    choice = ask("Detect hands on [1] rgb (default) or [2] thermal_vis ? ", "1")
    sub = "thermal_vis" if choice == "2" else "rgb"
    img_dir = patient_dir / sub
    if not img_dir.is_dir():
        raise SystemExit(f"Sub-folder '{sub}' not found inside {patient_dir}")
    # thermal_rad holds .npy arrays, not images, so it is never used here.
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {img_dir}")
    print(f"Using {len(images)} images from: {img_dir}")
    interval = float(ask("Seconds between extracted frames [5]: ", "5"))

    first = cv2.imread(str(images[0]))
    fh, fw = first.shape[:2]
    roi_txt = ask(f"ROI x,y,w,h (frame {fw}x{fh}; Enter = full frame): ")
    roi = (0, 0, fw, fh)
    if roi_txt:
        try:
            x, y, w, h = [int(v) for v in roi_txt.replace(" ", "").split(",")]
            x = max(0, min(x, fw - 1)); y = max(0, min(y, fh - 1))
            roi = (x, y, max(1, min(w, fw - x)), max(1, min(h, fh - y)))
        except ValueError:
            print("Bad ROI; using full frame.")

    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", participant).strip("_") or "participant"
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print('Where to save ?')
    print('press Enter to save in the output folder OR give a specified path to save the output')
    out_dir = Path(ask("press ENTER or give path to save Output folder  [hand_analysis_output]: ",
                       "hand_analysis_output")) / safe / session_id
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=PRESENCE_CONFIDENCE,
    )
    lm_header = [
        "participant_name", "session_id", "frame", "timestamp_ms", "handedness",
        "hand_score", "gesture", "finger", "visibility_assessment",
        "posture_estimate", "joint_angle_degrees", "mcp_angle_degrees",
        "pip_angle_degrees", "dip_or_ip_angle_degrees", "anatomical_status",
        "landmark_id", "x_normalized_in_roi", "y_normalized_in_roi", "z_relative",
        "x_pixel_in_frame", "y_pixel_in_frame", "hand_center_x", "hand_center_y",
        "bbox_x", "bbox_y", "bbox_width", "bbox_height",
        "hand_velocity_px_per_sec", "finger_tip_speed_px_per_sec", "image_file",
    ]
    jt_header = [
        "participant_name", "frame", "image_file", "image_width", "image_height",
        "timestamp_s", "handedness", "finger", "joint", "landmark_id",
        "x_px", "y_px", "z_relative", "joint_angle_deg",
    ]
    prev, detections = {}, 0
    with open(out_dir / "landmark_data.csv", "w", newline="", encoding="utf-8") as f1, \
         open(out_dir / "joint_coordinates.csv", "w", newline="", encoding="utf-8") as f2, \
         mp.tasks.vision.HandLandmarker.create_from_options(options) as lmk:
        w1, w2 = csv.writer(f1), csv.writer(f2)
        w1.writerow(lm_header); w2.writerow(jt_header)

        for idx, path in enumerate(images):
            frame = cv2.imread(str(path))
            if frame is None:
                print("Skipping unreadable:", path.name); continue
            H, W = frame.shape[:2]
            r = roi if (W, H) == (fw, fh) else (0, 0, W, H)
            rx, ry, rw, rh = r
            t_s = idx * interval
            t_ms = int(round(t_s * 1000))
            rgb = cv2.cvtColor(frame[ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2RGB)
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=np.ascontiguousarray(rgb)))
            cur = {}
            for hi, landmarks in enumerate(res.hand_landmarks or []):
                detections += 1
                pts = normalized_to_frame_points(landmarks, r)
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                x0, y0 = max(0, min(xs)), max(0, min(ys))
                x1, y1 = min(W - 1, max(xs)), min(H - 1, max(ys))
                center = ((x0 + x1) // 2, (y0 + y1) // 2)
                info = res.handedness[hi][0]
                hand, score = info.category_name, float(info.score)
                if SWAP_HANDEDNESS:   # input is NOT mirrored -> MediaPipe label is flipped
                    hand = {'Left': 'Right', 'Right': 'Left'}.get(hand, hand)
                fdata = analyze_fingers(pts, W, H, r)
                gest = classify_gesture(fdata)

                p0, dt, vel = prev.get(hand), None, None
                if p0:
                    dt = max((t_ms - p0["t"]) / 1000, 1e-6)
                    vel = distance(center, p0["center"]) / dt
                cur[hand] = {"center": center, "t": t_ms,
                             "tips": {n: tuple(fdata[n]["tip_pixel"]) for n in FINGER_TIPS}}

                lm_info = {0: ("", "WRIST", None)}
                for fg, ids in FINGER_LANDMARKS.items():
                    angs = fdata[fg]["joint_angles_degrees"]
                    for lid, jn in zip(ids, joint_names(fg)):
                        lm_info[lid] = (fg, jn, angs.get(jn))

                for lid, lm in enumerate(landmarks):
                    px, py = pts[lid]
                    fg, jn, ang = lm_info[lid]
                    fd = fdata.get(fg)
                    angs = fd["joint_angles_degrees"] if fd else {}
                    tip_speed = ""
                    if p0 and fg and lid == FINGER_TIPS[fg] and p0["tips"].get(fg):
                        tip_speed = round(distance((px, py), p0["tips"][fg]) / dt, 2)
                    w1.writerow([
                        participant, session_id, idx, t_ms, hand, round(score, 4), gest, fg,
                        fd["visibility_assessment"] if fd else "",
                        fd["posture_estimate"] if fd else "",
                        fd["joint_angle_degrees"] if fd else "",
                        angs.get("MCP", ""), angs.get("PIP", ""),
                        angs.get("DIP", angs.get("IP", "")),
                        fd["anatomical_status"] if fd else "",
                        lid, round(lm.x, 6), round(lm.y, 6), round(lm.z, 6),
                        px, py, center[0], center[1], x0, y0,
                        max(0, x1 - x0), max(0, y1 - y0),
                        round(vel, 2) if vel is not None else "", tip_speed, path.name])
                    w2.writerow([participant, idx, path.name, W, H, t_s, hand,
                                 fg or "wrist", jn, lid, px, py, round(lm.z, 6),
                                 "" if ang is None else ang])

                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (255, 80, 0), 2)
                for px, py in pts:
                    cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"{hand} | {gest}", (x0, max(20, y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            prev = cur
            cv2.imwrite(str(out_dir / "annotated" / path.name), frame)
            print(f"[{idx + 1}/{len(images)}] {path.name}: {len(res.hand_landmarks or [])} hand(s)")

    print("\nDone.", detections, "detections.")
    print("landmark_data.csv     :", (out_dir / "landmark_data.csv").resolve())
    print("joint_coordinates.csv :", (out_dir / "joint_coordinates.csv").resolve())


if __name__ == "__main__":
    main()