"""
Hand Analysis System
- Select webcam or prerecorded video
- Draw/move an adjustable ROI with the mouse
- MediaPipe 21-landmark hand tracking
- Hand/finger coordinates, posture heuristics, movement tracking
- Live on-screen report
- CSV + JSON + optional PDF session report

Install:
    pip install opencv-python mediapipe reportlab

Controls:
    Mouse drag : draw/change ROI
    C          : clear ROI (detect over full frame)
    L          : toggle landmark ID/coordinate labels
    SPACE      : pause/resume analysis
    ESC or Q   : stop session, save collected data, and exit

Keyboard controls are read from the OpenCV video window. On Windows, Q, ESC,
and SPACE can also be pressed in the terminal running this script when it has focus.

Important:
MediaPipe Hand Landmarker returns 21 predicted landmarks, but does not provide
a reliable per-finger anatomical-presence or visibility score. The finger
"visibility" labels below are conservative image-boundary checks only. They
must not be interpreted as proof that a finger is anatomically absent or
occluded.
"""

import cv2
import mediapipe as mp
import csv
import json
import math
import time
import re
import os
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime


# ---------------------------- CONFIGURATION ----------------------------

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


# ---------------------------- ROI SELECTION ----------------------------

class ROISelector:
    """Mouse-drag ROI selector. None means use the complete frame."""

    def __init__(self):
        self.roi = None
        self.dragging = False
        self.start = None
        self.end = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.end = (x, y)

            x1 = min(self.start[0], self.end[0])
            y1 = min(self.start[1], self.end[1])
            x2 = max(self.start[0], self.end[0])
            y2 = max(self.start[1], self.end[1])

            if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                self.roi = (x1, y1, x2 - x1, y2 - y1)

    def clear(self):
        self.roi = None
        self.dragging = False
        self.start = None
        self.end = None

    def draw(self, frame):
        if self.roi is not None:
            x, y, w, h = self.roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame, "ROI", (x + 5, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if self.dragging and self.start and self.end:
            cv2.rectangle(frame, self.start, self.end, (0, 255, 255), 2)


def read_control_key(delay_ms=10):
    """Read controls from either the OpenCV window or the Windows console.

    OpenCV keyboard events are not delivered reliably by every IDE/GUI
    backend. The Windows console fallback means Q, ESC and SPACE also work
    when the terminal running this script has focus.
    """
    # First check the console. This is a non-blocking read and needs no
    # additional package on Windows.
    if os.name == "nt":
        try:
            import msvcrt
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in ("\x00", "\xe0"):
                    # Consume the second byte of extended keys (arrows, etc.).
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    return -1
                return ord(char) & 0xFF
        except (ImportError, OSError):
            pass

    # waitKeyEx pumps the OpenCV GUI event loop as well as reading its keys.
    try:
        key = cv2.waitKeyEx(max(1, int(delay_ms)))
    except (AttributeError, cv2.error):
        try:
            key = cv2.waitKey(max(1, int(delay_ms)))
        except cv2.error:
            return -1

    return -1 if key < 0 else key & 0xFF


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


# ---------------------------- REPORT HELPERS ---------------------------

def write_pdf_report(summary, pdf_path):
    """Create a PDF report. Requires reportlab."""
    try:
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, PageBreak
        )
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
    except ImportError:
        print("PDF skipped: install reportlab with: pip install reportlab")
        return False

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=landscape(letter),
        rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28
    )
    story = [
        Paragraph("Individual Hand Analysis Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph(f"Participant: {summary['participant_name']}", styles["BodyText"]),
        Paragraph(f"Session ID: {summary['session_id']}", styles["BodyText"]),
        Paragraph(f"Source: {summary['source']}", styles["BodyText"]),
        Paragraph(f"Frames processed: {summary['frames_processed']}", styles["BodyText"]),
        Paragraph(f"Hand detections: {summary['hand_detections']}", styles["BodyText"]),
        Paragraph(f"Duration: {summary['duration_seconds']} seconds", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Gesture counts (all detected hands)", styles["Heading2"]),
    ]

    gesture_rows = [["Gesture", "Count"]]
    gesture_rows.extend([[name, str(count)] for name, count in summary["gesture_counts"].items()])
    t = Table(gesture_rows, hAlign="LEFT")
    t.setStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 5),
    ])
    story.extend([t, Spacer(1, 12), Paragraph("Per-hand analysis", styles["Heading2"])])

    for hand_name, hand_summary in summary["per_hand"].items():
        story.append(Paragraph(
            f"{hand_name} — observed in {hand_summary['frames_seen']} frames",
            styles["Heading3"]
        ))
        rows = [["Finger", "Posture observations", "Image-boundary assessment",
                 "Mean MCP", "Mean PIP", "Mean DIP", "Anatomical status"]]
        for finger, detail in hand_summary["fingers"].items():
            means = detail["mean_joint_angles_degrees"]
            rows.append([
                finger,
                ", ".join(f"{k}: {v}" for k, v in detail["posture_counts"].items()) or "None",
                ", ".join(f"{k}: {v}" for k, v in detail["visibility_counts"].items()) or "None",
                str(means.get("MCP", "N/A")),
                str(means.get("PIP", "N/A")),
                str(means.get("DIP", "N/A")),
                "Unknown (not inferred)",
            ])
        table = Table(rows, repeatRows=1, hAlign="LEFT")
        table.setStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("PADDING", (0, 0), (-1, -1), 4),
        ])
        story.extend([table, Spacer(1, 10)])

    story.append(Paragraph("Interpretation limits", styles["Heading2"]))
    story.append(Paragraph(
        "A missing landmark or an undetected hand does NOT establish an "
        "anatomically absent finger. MediaPipe predicts 21 landmarks and does "
        "not provide a dependable anatomical-presence or per-landmark "
        "visibility score. Image-boundary labels only describe predicted "
        "landmark positions relative to the frame/ROI. Occlusion, unusual "
        "anatomy, camera angle, blur, and tracking errors can change results. "
        "Joint angles are 2D image-plane estimates, not clinical measurements.",
        styles["BodyText"]
    ))
    doc.build(story)
    return True


# ---------------------------- MAIN PROGRAM -----------------------------

def choose_video_source():
    print("\nChoose video source:")
    print("1. Live webcam")
    print("2. Recorded video file")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        cap = cv2.VideoCapture(0)
        source_name = "webcam"
    elif choice == "2":
        video_path = input("Enter full video path: ").strip().strip('"')
        cap = cv2.VideoCapture(video_path)
        source_name = video_path
    else:
        print("Invalid choice.")
        return None, None

    if not cap.isOpened():
        print("Could not open the selected video source.")
        cap.release()
        return None, None

    return cap, source_name


def main():
    participant_name = input("Enter participant name or ID: ").strip()
    if not participant_name:
        print("Participant name/ID cannot be empty.")
        return

    # Keep folder names safe while preserving the original participant label in reports.
    safe_participant = re.sub(r"[^A-Za-z0-9_-]+", "_", participant_name).strip("_")
    safe_participant = safe_participant or "participant"
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = OUTPUT_DIR / safe_participant / session_id
    session_dir.mkdir(parents=True, exist_ok=False)

    cap, source_name = choose_video_source()
    if cap is None:
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if not source_fps or source_fps <= 0 or math.isnan(source_fps):
        source_fps = 30.0

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    roi_selector = ROISelector()
    window_name = "Hand Analysis System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(frame_width, 1280), min(frame_height, 800))
    cv2.setMouseCallback(window_name, roi_selector.mouse_callback)

    show_landmark_labels = False
    frame_number = 0
    start_time = time.perf_counter()
    last_timestamp_ms = -1
    previous_tracks = {}
    trajectory = defaultdict(list)

    csv_path = session_dir / "landmark_data.csv"
    json_path = session_dir / "session_report.json"
    pdf_path = session_dir / "session_report.pdf"

    gesture_counts = Counter()
    finger_posture_counts = {
        finger: Counter()
        for finger in FINGER_LANDMARKS
    }
    finger_visibility_counts = {
        finger: Counter()
        for finger in FINGER_LANDMARKS
    }
    hand_detections = 0
    # Separate accumulators for Left and Right. These are handedness labels,
    # not biometric identity recognition.
    per_hand = defaultdict(lambda: {
        "frames_seen": 0,
        "gesture_counts": Counter(),
        "fingers": {
            finger: {
                "posture_counts": Counter(),
                "visibility_counts": Counter(),
                "angle_sums": Counter(),
                "angle_counts": Counter(),
            }
            for finger in FINGER_LANDMARKS
        },
    })

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "participant_name", "session_id", "frame", "timestamp_ms", "handedness", "hand_score", "gesture",
        "finger", "visibility_assessment", "posture_estimate",
        "joint_angle_degrees", "mcp_angle_degrees", "pip_angle_degrees",
        "dip_or_ip_angle_degrees", "anatomical_status", "landmark_id",
        "x_normalized_in_roi", "y_normalized_in_roi", "z_relative",
        "x_pixel_in_frame", "y_pixel_in_frame",
        "hand_center_x", "hand_center_y",
        "bbox_x", "bbox_y", "bbox_width", "bbox_height",
        "hand_velocity_px_per_sec", "finger_tip_speed_px_per_sec"
    ])

    paused = False
    paused_frame = None

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                # While paused, keep the webcam buffer fresh but do not analyse frames.
                if paused:
                    if source_name == "webcam":
                        cap.grab()
                    display = paused_frame.copy()
                    cv2.putText(display, "PAUSED - SPACE to resume | ESC to stop",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 165, 255), 2)
                    cv2.imshow(window_name, display)
                    key = read_control_key(30)
                    if key in (27, ord("q"), ord("Q")):
                        break
                    elif key == ord(" "):
                        paused = False
                    elif key in (ord("c"), ord("C")):
                        roi_selector.clear()
                    elif key in (ord("l"), ord("L")):
                        show_landmark_labels = not show_landmark_labels
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                frame_number += 1
                frame_height, frame_width = frame.shape[:2]

                # Full frame or selected ROI.
                if roi_selector.roi is None:
                    roi_rect = (0, 0, frame_width, frame_height)
                else:
                    rx, ry, rw, rh = roi_selector.roi
                    rx = max(0, min(rx, frame_width - 1))
                    ry = max(0, min(ry, frame_height - 1))
                    rw = max(1, min(rw, frame_width - rx))
                    rh = max(1, min(rh, frame_height - ry))
                    roi_rect = (rx, ry, rw, rh)

                rx, ry, rw, rh = roi_rect
                roi_image = frame[ry:ry + rh, rx:rx + rw]
                rgb_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_roi
                )

                # Use source frame time for recorded video; monotonic clock for webcam.
                if source_name == "webcam":
                    timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                else:
                    timestamp_ms = int((frame_number - 1) * 1000 / source_fps)
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                current_tracks = {}
                if result.hand_landmarks:
                    for hand_index, landmarks in enumerate(result.hand_landmarks):
                        hand_detections += 1
                        points = normalized_to_frame_points(landmarks, roi_rect)

                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        x_min = max(0, min(xs))
                        y_min = max(0, min(ys))
                        x_max = min(frame_width - 1, max(xs))
                        y_max = min(frame_height - 1, max(ys))
                        box_w = max(0, x_max - x_min)
                        box_h = max(0, y_max - y_min)
                        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

                        hand_info = result.handedness[hand_index][0]
                        hand_name = hand_info.category_name
                        hand_score = float(hand_info.score)
                        track_key = hand_name

                        finger_data = analyze_fingers(
                            points, frame_width, frame_height, roi_rect
                        )
                        gesture = classify_gesture(finger_data)
                        gesture_counts[gesture] += 1
                        per_hand[hand_name]["frames_seen"] += 1
                        per_hand[hand_name]["gesture_counts"][gesture] += 1

                        for finger, info in finger_data.items():
                            finger_posture_counts[finger][info["posture_estimate"]] += 1
                            finger_visibility_counts[finger][info["visibility_assessment"]] += 1
                            finger_bucket = per_hand[hand_name]["fingers"][finger]
                            finger_bucket["posture_counts"][info["posture_estimate"]] += 1
                            finger_bucket["visibility_counts"][info["visibility_assessment"]] += 1
                            for joint, angle_value in info["joint_angles_degrees"].items():
                                if angle_value is not None:
                                    finger_bucket["angle_sums"][joint] += angle_value
                                    finger_bucket["angle_counts"][joint] += 1

                        # Hand center velocity.
                        hand_velocity = None
                        previous = previous_tracks.get(track_key)
                        if previous is not None:
                            dt = max((timestamp_ms - previous["timestamp_ms"]) / 1000, 1e-6)
                            hand_velocity = distance(center, previous["center"]) / dt

                        current_tracks[track_key] = {
                            "center": center,
                            "timestamp_ms": timestamp_ms,
                            "tips": {
                                finger: tuple(finger_data[finger]["tip_pixel"])
                                for finger in FINGER_TIPS
                            }
                        }
                        trajectory[track_key].append({
                            "frame": frame_number,
                            "timestamp_ms": timestamp_ms,
                            "center_x": center[0],
                            "center_y": center[1],
                        })

                        # Draw hand skeleton and points.
                        for a, b in HAND_CONNECTIONS:
                            cv2.line(frame, points[a], points[b], (255, 80, 0), 2)
                        for landmark_id, (px, py) in enumerate(points):
                            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                            if show_landmark_labels:
                                cv2.putText(
                                    frame, str(landmark_id), (px + 4, py - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
                                )

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.circle(frame, center, 6, (0, 255, 255), -1)

                        label_y = max(20, y_min - 55)
                        cv2.putText(
                            frame, f"{hand_name} | {gesture}",
                            (x_min, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 2
                        )
                        cv2.putText(
                            frame, f"Center {center}  Score {hand_score:.2f}",
                            (x_min, label_y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 0), 1
                        )
                        if hand_velocity is not None:
                            cv2.putText(
                                frame, f"Hand speed {hand_velocity:.1f} px/s",
                                (x_min, label_y + 38), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 255, 0), 1
                            )

                        # Compact live finger status panel.
                        panel_x = min(frame_width - 230, max(5, x_max + 8))
                        panel_y = max(110, min(frame_height - 130, y_min))
                        for row, (finger, info) in enumerate(finger_data.items()):
                            text = f"{finger}: {info['posture_estimate']} / {info['visibility_assessment']}"
                            cv2.putText(
                                frame, text, (panel_x, panel_y + row * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                (255, 255, 255), 1
                            )

                        # One CSV row per landmark, with finger summary repeated.
                        for landmark_id, lm in enumerate(landmarks):
                            px, py = points[landmark_id]
                            finger_name = ""
                            visibility = ""
                            posture = ""
                            angle = ""
                            mcp_angle = pip_angle = distal_angle = ""
                            anatomical_status = ""
                            for name, ids in FINGER_LANDMARKS.items():
                                if landmark_id in ids:
                                    finger_name = name
                                    visibility = finger_data[name]["visibility_assessment"]
                                    posture = finger_data[name]["posture_estimate"]
                                    angle = finger_data[name]["joint_angle_degrees"]
                                    joint_values = finger_data[name]["joint_angles_degrees"]
                                    mcp_angle = joint_values.get("MCP", "")
                                    pip_angle = joint_values.get("PIP", "")
                                    distal_angle = joint_values.get("DIP", joint_values.get("IP", ""))
                                    anatomical_status = finger_data[name]["anatomical_status"]
                                    break

                            tip_speed = ""
                            if previous is not None and finger_name:
                                dt = max((timestamp_ms - previous["timestamp_ms"]) / 1000, 1e-6)
                                old_tip = previous["tips"].get(finger_name)
                                if old_tip is not None and landmark_id == FINGER_TIPS[finger_name]:
                                    tip_speed = distance(
                                        (px, py), old_tip
                                    ) / dt

                            csv_writer.writerow([
                                participant_name, session_id, frame_number, timestamp_ms, hand_name, hand_score,
                                gesture, finger_name, visibility, posture, angle,
                                mcp_angle, pip_angle, distal_angle, anatomical_status,
                                landmark_id, round(landmarks[landmark_id].x, 6),
                                round(landmarks[landmark_id].y, 6),
                                round(landmarks[landmark_id].z, 6),
                                px, py, center[0], center[1],
                                x_min, y_min, box_w, box_h,
                                round(hand_velocity, 2) if hand_velocity is not None else "",
                                round(tip_speed, 2) if tip_speed != "" else ""
                            ])

                previous_tracks = current_tracks

                # Draw ROI after annotations so it remains visible.
                roi_selector.draw(frame)

                elapsed = time.perf_counter() - start_time
                processing_fps = frame_number / max(elapsed, 1e-6)
                cv2.putText(
                    frame,
                    f"Frame {frame_number} | Processing FPS {processing_fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    "Keys: SPACE pause/resume | Q or ESC stop | C clear ROI | L labels | Drag: ROI",
                    (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1
                )

                cv2.imshow(window_name, frame)
                key = read_control_key(10)

                if key in (27, ord("q"), ord("Q")):
                    break
                elif key == ord(" "):
                    paused = True
                    paused_frame = frame.copy()
                elif key in (ord("c"), ord("C")):
                    roi_selector.clear()
                elif key in (ord("l"), ord("L")):
                    show_landmark_labels = not show_landmark_labels

    finally:
        cap.release()
        csv_file.close()
        cv2.destroyAllWindows()

    duration = round(time.perf_counter() - start_time, 2)
    per_hand_json = {}
    for hand_name, hand_data in per_hand.items():
        fingers_json = {}
        for finger, detail in hand_data["fingers"].items():
            mean_angles = {}
            for joint, total in detail["angle_sums"].items():
                count = detail["angle_counts"].get(joint, 0)
                mean_angles[joint] = round(total / count, 1) if count else None
            fingers_json[finger] = {
                "posture_counts": dict(detail["posture_counts"]),
                "visibility_counts": dict(detail["visibility_counts"]),
                "mean_joint_angles_degrees": mean_angles,
                "anatomical_status": "unknown_not_inferred_from_landmarks",
            }
        per_hand_json[hand_name] = {
            "frames_seen": hand_data["frames_seen"],
            "gesture_counts": dict(hand_data["gesture_counts"]),
            "fingers": fingers_json,
        }

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "participant_name": participant_name,
        "session_id": session_id,
        "source": str(source_name),
        "frames_processed": frame_number,
        "hand_detections": hand_detections,
        "duration_seconds": duration,
        "per_hand": per_hand_json,
        "gesture_counts": dict(gesture_counts),
        "finger_posture_counts": {
            finger: dict(counts)
            for finger, counts in finger_posture_counts.items()
        },
        "finger_visibility_counts": {
            finger: dict(counts)
            for finger, counts in finger_visibility_counts.items()
        },
        "hand_center_trajectories": dict(trajectory),
        "output_files": {
            "landmark_csv": str(csv_path),
            "session_json": str(json_path),
            "session_pdf": str(pdf_path),
        },
        "interpretation_limits": [
            "MediaPipe landmarks are predictions, not proof of anatomical finger presence.",
            "Visibility labels are based on image boundaries, not a trained occlusion detector.",
            "Finger posture and gesture labels are geometric estimates and may fail for rotated hands.",
            "Z is relative depth, not calibrated distance in centimeters.",
            "Hand speed is in image pixels per second, not physical-world speed."
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pdf_created = write_pdf_report(summary, pdf_path)

    print("\nAnalysis complete.")
    print("Participant:", participant_name)
    print("Session folder:", session_dir.resolve())
    print("Frames processed:", frame_number)
    print("Hand detections:", hand_detections)
    print("CSV:", csv_path.resolve())
    print("JSON:", json_path.resolve())
    print("PDF:", pdf_path.resolve() if pdf_created else "Not created (install reportlab)")


if __name__ == "__main__":
    main()
