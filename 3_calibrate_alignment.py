"""
COMPONENT 3a: calibrate_alignment.py   (run ONCE per camera rig, NOT per patient)
...
Output: <hand_analysis_output>/<patient>/<session>/rig_calibration.json
        <patient>/alignment_check/calib_*.png
...
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

FINGER_CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                 [13, 14, 15, 16], [17, 18, 19, 20]]
PALM = [0, 1, 5, 9, 13, 17]


# ---------------------- orientation detection ----------------------
def dihedral(a, k):
    if k >= 4:
        a = a.T
    return np.ascontiguousarray(np.rot90(a, k % 4))


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def detect_orientation(rad_frames, vis_images, size=(80, 60)):
    scores = np.zeros(8); counts = np.zeros(8)
    for rad, vis in zip(rad_frames, vis_images):
        g = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY) if vis.ndim == 3 else vis
        gs = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        rad = np.nan_to_num(rad.astype(np.float32))
        for k in range(8):
            cand = dihedral(rad, k)
            if abs(cand.shape[1] / cand.shape[0] - g.shape[1] / g.shape[0]) > 0.02:
                continue
            cs = cv2.resize(cand, size, interpolation=cv2.INTER_AREA)
            scores[k] += abs(_ncc(gs, cs)); counts[k] += 1
    valid = counts > 0
    if not valid.any():
        raise RuntimeError("No orientation candidate has the video's aspect ratio.")
    scores[valid] /= counts[valid]; scores[~valid] = -1
    order = np.argsort(scores)[::-1]
    return int(order[0]), scores, float(scores[order[0]] - scores[order[1]])


# ---------------------- 4-DOF similarity model ----------------------
def make_matrix(s, theta, tx, ty, rgb_size, th_size):
    cr = np.array(rgb_size, float) / 2.0
    ct = np.array(th_size, float) / 2.0
    c, sn = np.cos(theta), np.sin(theta)
    M2 = s * np.array([[c, -sn], [sn, c]], dtype=np.float64)
    t = ct - M2 @ cr + np.array([tx, ty], dtype=np.float64)
    return np.hstack([M2, t.reshape(2, 1)])


def apply_matrix(M, pts):
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ M[:, :2].T + M[:, 2]


# ---------------------- hand geometry ----------------------
def hand_mask(rad):
    v = np.nan_to_num(rad.astype(np.float32))
    lo, hi = np.percentile(v, [1, 99])
    u8 = np.clip((v - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (5, 5), 0)
    _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if m.mean() / 255 > 0.5:
        m = 255 - m
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m > 0


def silhouette(pts, shape, scale=1.0):
    p = np.asarray(pts, np.float32) * scale
    img = np.zeros(shape, np.uint8)
    palm = float(np.linalg.norm(p[5] - p[17]))
    thick = int(max(2, round(0.22 * palm)))
    hull = cv2.convexHull(np.round(p[PALM]).astype(np.int32))
    cv2.fillConvexPoly(img, hull, 1)
    for chain in FINGER_CHAINS:
        q = np.round(p[chain]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [q], False, 1, thick)
    return img > 0, palm


def dice_score(sil, mask, palm):
    palm = float(np.asarray(palm).item())
    if np.ndim(mask) < 2:
        raise TypeError(
            "dice_score: `mask` must be a 2-D boolean array, got "
            f"ndim={np.ndim(mask)} — check argument order "
            "(should be sil, mask, palm)."
        )
    k = int(max(3, round(0.35 * palm))) | 1
    near = cv2.dilate(sil.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    b = mask & near
    den = int(sil.sum()) + int(b.sum())
    return 2.0 * int((sil & b).sum()) / den if den else 0.0


def half(mask):
    return cv2.resize(mask.astype(np.uint8) * 255,
                      (mask.shape[1] // 2, mask.shape[0] // 2),
                      interpolation=cv2.INTER_AREA) > 127


def pattern_search(cost, p0, steps, min_steps, max_iter=400):
    p = np.array(p0, float); steps = np.array(steps, float)
    best = cost(p)
    for _ in range(max_iter):
        improved = False
        for i in range(len(p)):
            for sgn in (1, -1):
                q = p.copy(); q[i] += sgn * steps[i]
                c = cost(q)
                if c < best - 1e-7:
                    p, best, improved = q, c, True
        if not improved:
            steps = steps / 2
            if np.all(steps < min_steps):
                break
    return p, best


def fit_similarity(samples, rgb_size, th_size, s0=1.0, log=print):
    sub = [samples[i] for i in np.linspace(0, len(samples) - 1,
                                          min(30, len(samples))).astype(int)]
    half_s = [(p, half(m)) for p, m in sub]

    def make_cost(data, scale):
        def cost(q):
            s, theta, tx, ty = q
            M = make_matrix(s, theta, tx, ty, rgb_size, th_size)
            tot = 0.0
            for pts, m in data:
                sil, palm = silhouette(apply_matrix(M, pts), m.shape, scale)
                tot += dice_score(sil, m, palm)
            return 1.0 - tot / len(data)
        return cost

    cost_h = make_cost(half_s, 0.5)
    cost_f = make_cost(samples, 1.0)

    best = (9.0, None); n_evals = 0
    for s in s0 * np.array([0.85, 0.95, 1.0, 1.05, 1.15]):
        for theta in (-0.03, 0.0, 0.03):
            for tx in range(-24, 25, 6):
                for ty in range(-12, 13, 6):
                    c = cost_h((s, theta, tx, ty)); n_evals += 1
                    if c < best[0]:
                        best = (c, (s, theta, tx, ty))
    log(f"  coarse: {n_evals} evals, dice={1-best[0]:.3f}  "
        f"params s={best[1][0]:.4f} θ={np.degrees(best[1][1]):+.2f}° "
        f"tx={best[1][2]} ty={best[1][3]}")

    p, c = pattern_search(cost_f, best[1],
                          steps=(0.04 * best[1][0], 0.02, 4.0, 4.0),
                          min_steps=(0.001, 0.0005, 0.05, 0.05))
    return tuple(p), 1.0 - c


# ---------------------- data loading ----------------------
def load_groups(df):
    groups = {}
    for (fr, hand), g in df.groupby(["frame", "handedness"]):
        g = g.drop_duplicates("landmark_id").set_index("landmark_id")
        if all(i in g.index for i in range(21)):
            groups[(int(fr), str(hand))] = \
                g.loc[list(range(21)), ["x_px", "y_px"]].to_numpy(np.float32)
    return groups


# ---------------------- interactive helpers ----------------------
def ask(prompt, default=None):
    shown = f" [{default}]" if default is not None else ""
    v = input(f"{prompt}{shown}: ").strip().strip('"').strip("'").strip()
    return v if v else (None if default is None else str(default))


def ensure_json_path(p: Path) -> Path:
    if p.is_dir():
        return p / "rig_calibration.json"
    if p.suffix.lower() != ".json":
        return p.with_suffix(".json")
    return p


# ---------------------- main ----------------------
def main():
    print("\n=== 4-DOF rig calibration (s, theta, tx, ty) ===\n")

    patient = Path(ask("Path to a patient frames folder (rgb, thermal_vis, thermal_rad)"))
    csv_path = Path(ask("Path to its joint_coordinates.csv"))

    # ---- Derive default rig_path from the CSV location ----
    #   We want it INSIDE the session folder, right next to the CSV:
    #   <hand_analysis_output>/<patient>/<session>/rig_calibration.json
    session_dir = csv_path.parent
    if session_dir.is_dir():
        default_rig = session_dir / "rig_calibration.json"
    else:
        default_rig = Path("rig_calibration.json")

    rig_raw = Path(ask("Press ENTER to save the  RIG calibration file inside the FOLDER hand_analysis_output", str(default_rig)))
    rig_path = ensure_json_path(rig_raw)
    if rig_path != rig_raw:
        print(f"[cal ] resolved to file: {rig_path}")

    fov_hint = ask("RGB HFOV and thermal HFOV in degrees, e.g. 60,45 (Enter = unknown)", "")
    baseline_mm, dist_mm = 10.0, 310.0
    if fov_hint:
        baseline_mm = float(ask("Distance between the two cameras (mm)", 10))
        dist_mm = float(ask("Distance from cameras to hand (mm)", 310))

    # ---- load data ----
    rad_files = sorted((patient / "thermal_rad").glob("*.npy"))
    vis_files = sorted(p for p in (patient / "thermal_vis").glob("*")
                       if p.suffix.lower() in IMG_EXTS)
    if not rad_files:
        raise SystemExit("No .npy files in thermal_rad.")

    df = pd.read_csv(csv_path)
    rgb_size = (int(df["image_width"].iloc[0]), int(df["image_height"].iloc[0]))
    groups = load_groups(df)
    if not groups:
        raise SystemExit("No complete 21-landmark hands in the CSV.")
    keys = [k for k in sorted(groups) if k[0] < len(rad_files)]
    if not keys:
        raise SystemExit("No usable hands (need all 21 landmarks + matching .npy).")
    n_frames = len({k[0] for k in keys})
    print(f"\n{len(keys)} hand detections available "
          f"({n_frames} frames with a complete hand, of {len(rad_files)} radiometric frames).")
    while True:
        n_txt = ask(f"How many to use? 1..{len(keys)} or 'all'", "all").lower()
        if n_txt == "all":
            n_use = len(keys); break
        if n_txt.isdigit() and 1 <= int(n_txt) <= len(keys):
            n_use = int(n_txt); break
        print(f"  Please enter 'all' or a number between 1 and {len(keys)}.")
    pick = [keys[i] for i in np.linspace(0, len(keys) - 1, n_use).astype(int)]
    print(f"Using {len(pick)} of {len(keys)} hand detections.")

    # ---- 1. orientation ----
    raw0 = np.squeeze(np.load(rad_files[pick[0][0]]))
    if vis_files and len(vis_files) == len(rad_files):
        idxs = sorted({k[0] for k in pick})[:: max(1, len(pick) // 8)][:8]
        rads = [np.squeeze(np.load(rad_files[i])) for i in idxs]
        vis = [cv2.imread(str(vis_files[i])) for i in idxs]
        k, scores, margin = detect_orientation(rads, vis)
        print("Orientation scores (|corr| vs thermal video):",
              np.round(scores, 3).tolist())
        print(f"→ orientation k={k}  (margin to runner-up {margin:.3f})")
        if margin < 0.05:
            print("WARNING: orientation is ambiguous; check alignment_check images.")
    else:
        k = int(ask("thermal_vis not usable. Orientation 0-7 (4 = plain transpose)", 4))
    th_h, th_w = dihedral(raw0, k).shape
    th_size = (th_w, th_h)
    print(f"Raw .npy {raw0.shape} → oriented {th_h}×{th_w} (H×W);  "
          f"RGB {rgb_size[1]}×{rgb_size[0]}")

    # ---- 2. samples ----
    samples, meta = [], []
    for n_done, key in enumerate(pick, 1):
        rad = dihedral(np.squeeze(np.load(rad_files[key[0]])).astype(np.float32), k)
        samples.append((groups[key], hand_mask(rad)))
        meta.append(key)
        if n_done % 100 == 0:
            print(f"  loaded {n_done}/{len(pick)} frames")

    # ---- 3. baseline Dice ----
    s0 = 1.0
    if fov_hint:
        a, b = [float(v) for v in fov_hint.split(",")]
        s0 = (th_w / rgb_size[0]) * np.tan(np.radians(a / 2)) / np.tan(np.radians(b / 2))
        print(f"Scale hint from FOVs: {s0:.3f}")
    base = make_matrix(s0, 0.0, 0, 0, rgb_size, th_size)

    before_scores = []
    for p_, m_ in samples:
        sil, palm = silhouette(apply_matrix(base, p_), m_.shape)
        before_scores.append(dice_score(sil, m_, palm))
    before = float(np.mean(before_scores))
    print(f"Mean Dice with scale only, no rotation/parallax: {before:.3f}")

    # ---- 4. fit 4-DOF ----
    print("\nFitting 4-DOF (this may take ~30–60 s)...")
    (s, theta, tx, ty), dice = fit_similarity(samples, rgb_size, th_size, s0=s0)
    M = make_matrix(s, theta, tx, ty, rgb_size, th_size)
    theta_deg = np.degrees(theta)

    print(f"\nFitted:  s={s:.4f}  θ={theta_deg:+.3f}°  "
          f"tx={tx:+.2f}px  ty={ty:+.2f}px  |  mean Dice={dice:.3f}")
    print(f"Improvement over no-rotation baseline: {before:.3f} → {dice:.3f}  "
          f"({(dice - before) * 100:+.1f}%)")
    if abs(theta_deg) > 5:
        print("⚠️  |θ| > 5° — the two cameras are visibly rotated relative to each other.")
        print("    This is fine to model, but verify the alignment_check images carefully.")

    # ---- 5. physical sanity check ----
    if fov_hint:
        f_t_px = (th_w / 2) / np.tan(np.radians(b / 2))
        exp = f_t_px * baseline_mm / dist_mm
        print(f"\nPhysical prediction: baseline={baseline_mm:g} mm, distance={dist_mm:g} mm")
        print(f"  → expected parallax ≈ {exp:.1f} thermal px")
        print(f"  → fitted |tx| = {abs(tx):.1f} px, |ty| = {abs(ty):.1f} px, "
              f"|θ| = {abs(theta_deg):.2f}°")
        if abs(abs(tx) - exp) > max(4.0, 0.5 * exp) or abs(ty) > max(4.0, 0.25 * exp):
            print("⚠️  WARNING: fitted offset disagrees with camera geometry.")
            print("    Check FOVs, baseline/distance, and the alignment_check images.")

    if dice < 0.6:
        print("⚠️  WARNING: low overlap. Hand may not be clearly warmer than background,")
        print("    or orientation/FOV hint may be wrong. Inspect alignment_check.")

    # ---- 6. per-frame quality + overlays ----
    per = []
    for (p, m), _ in zip(samples, meta):
        sil, palm = silhouette(apply_matrix(M, p), m.shape)
        per.append(dice_score(sil, m, palm))
    per = np.array(per)
    print(f"\nPer-frame Dice: min={per.min():.2f}  median={np.median(per):.2f}  "
          f"max={per.max():.2f}")
    worst = np.argsort(per)[:5]
    print("Lowest 5 frames (frame, hand, dice):",
          [(meta[i][0], meta[i][1], round(float(per[i]), 2)) for i in worst])

    chk = patient / "alignment_check"; chk.mkdir(exist_ok=True)
    show = list(np.linspace(0, len(meta) - 1, min(6, len(meta))).astype(int)) + [int(worst[0])]
    for j in dict.fromkeys(show):
        key = meta[j]
        rad = dihedral(np.squeeze(np.load(rad_files[key[0]])).astype(np.float32), k)
        lo, hi = np.percentile(rad, [1, 99])
        u8 = np.clip((rad - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        cnts, _ = cv2.findContours(samples[j][1].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
        for x, y in apply_matrix(M, groups[key]):
            cv2.circle(img, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1)
        cv2.putText(img, f"frame {key[0]} dice {per[j]:.2f}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(chk / f"calib_{key[0]:06d}.png"), img)

    # ---- 7. save rig calibration ----
    rig_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rig_path.write_text(json.dumps({
            "model": "4-dof-similarity",
            "orientation": k,
            "matrix": M.tolist(),
            "scale": s,
            "rotation_rad": theta,
            "rotation_deg": theta_deg,
            "offset_px": [tx, ty],
            "rgb_size_wh": list(rgb_size),
            "thermal_size_wh": list(th_size),
            "mean_dice": dice,
            "frames_used": len(samples),
            "source_recording": str(patient),
            "source_csv": str(csv_path),
        }, indent=2))
    except (PermissionError, IsADirectoryError) as e:
        raise SystemExit(
            f"Could not write '{rig_path}': {e}\n"
            f"  → if that path is a FOLDER, add a filename, e.g. "
            f"'{rig_path}\\rig_calibration.json'"
        )
    print(f"\n✅ Rig calibration saved → {rig_path.resolve()}")
    print(f"   Check overlays     → {chk.resolve()}")
    print(f"   (white dots should sit on the hand; green contour = warm blob)")
    print(f"\nNow run 4_add_radiometric.py for EVERY patient — it uses this file.")


if __name__ == "__main__":
    main()