"""
COMPONENT 4: 4_add_radiometric.py   (run per patient, after 3_calibrate_alignment.py)

Maps every joint from RGB pixels into the radiometric frame and reads its
temperature.  Compared with the previous version:

  1. PER-FRAME LOCAL RE-ALIGNMENT  The rig calibration is global; the hand's
     depth changes frame to frame, so the true offset drifts by a few pixels.
     For every hand in every frame the joint set is shifted by a small bounded
     (dx, dy) (default +-6 px) that maximises Dice against that frame's warm
     blob.  The shift is only accepted if it improves Dice by >= MIN_GAIN.
  2. SNAP-TO-WARM-PIXEL  If a joint still lands on background, the reading is
     taken from the nearest warm pixel (<= SNAP_MAX_PX away, inside a slightly
     eroded blob so edge pixels are avoided) and flagged reading_method=snapped.
  3. NO FABRICATED VALUES  temperature_c contains MEASURED values only.
     Gaps are never filled with finger means.  temperature_filled_c is a
     separate column (short temporal interpolation, limit 3 frames) meant for
     plotting only; the report's statistics ignore it.

Input : joint_coordinates.csv, thermal_rad folder (.npy), rig_calibration.json
Output: joint_coordinates_radiometric.csv = input rows plus
   thermal_x_global, thermal_y_global  position from the rig calibration alone
   local_dx, local_dy                  per-frame correction applied (px)
   thermal_x, thermal_y                final joint position in thermal frame
   radiometric_value                   bilinear raw value at final position
   radiometric_hand_median             value actually used (raw units)
   reading_method                      direct | snapped | none
   snap_dist_px                        distance moved when snapped
   in_hand_blob                        final position lies on the warm blob
   align_dice_global, align_dice       alignment quality before / after local fix
   temperature_c                       measured temperature (NaN if none)
   temperature_filled_c, is_filled     plot-only temporal fill
Needs : pip install numpy pandas opencv-python
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FINGER_CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                 [13, 14, 15, 16], [17, 18, 19, 20]]
PALM = [0, 1, 5, 9, 13, 17]

SEARCH_PX = 6        # max local shift per axis
MIN_GAIN = 0.03      # Dice gain needed to accept a shift
SNAP_MAX_PX = 6      # max distance to snap a joint to a warm pixel
TEMPORAL_LIMIT = 3   # frames, plot-only fill


# ======================================================================
# Helpers (inline on purpose: this file is standalone)
# ======================================================================
def dihedral(a, k):
    if k >= 4:
        a = a.T
    return np.ascontiguousarray(np.rot90(a, k % 4))


def apply_matrix(M, pts):
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ M[:, :2].T + M[:, 2]


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
        raise TypeError("dice_score: `mask` must be 2-D (argument order sil, mask, palm)")
    k = int(max(3, round(0.35 * palm))) | 1
    near = cv2.dilate(sil.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    b = mask & near
    den = int(sil.sum()) + int(b.sum())
    return 2.0 * int((sil & b).sum()) / den if den else 0.0


def bilinear_many(arr, xs, ys):
    h, w = arr.shape
    out = np.full(len(xs), np.nan)
    ok = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    x, y = xs[ok], ys[ok]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    out[ok] = (arr[y0, x0] * (1 - fx) * (1 - fy) + arr[y0, x1] * fx * (1 - fy)
               + arr[y1, x0] * (1 - fx) * fy + arr[y1, x1] * fx * fy)
    return out


def patch_median_in_mask(arr, mask, x, y, r=2):
    h, w = arr.shape
    cx, cy = int(round(x)), int(round(y))
    if not (0 <= cx < w and 0 <= cy < h):
        return np.nan
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    vals = arr[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
    return float(np.median(vals)) if vals.size else np.nan


def to_celsius_params(values):
    med = np.nanmedian(values)
    if 5 <= med <= 80:        return 1.0, 0.0, "celsius"
    if 250 <= med <= 360:     return 1.0, -273.15, "kelvin"
    if 2500 <= med <= 3600:   return 0.1, -273.15, "decikelvin"
    if 25000 <= med <= 36000: return 0.01, -273.15, "centikelvin"
    return None


def load_groups(df):
    groups = {}
    for (fr, hand), g in df.groupby(["frame", "handedness"]):
        g = g.drop_duplicates("landmark_id").set_index("landmark_id")
        if all(i in g.index for i in range(21)):
            groups[(int(fr), str(hand))] = \
                g.loc[list(range(21)), ["x_px", "y_px"]].to_numpy(np.float32)
    return groups



def nearest_blob_pixel(mask, x, y, max_r):
    """Nearest True pixel of `mask` within max_r of (x, y) -> (px, py, dist) or None."""
    h, w = mask.shape
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - max_r), min(w, cx + max_r + 1)
    y0, y1 = max(0, cy - max_r), min(h, cy + max_r + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if xs.size == 0:
        return None
    xs, ys = xs + x0, ys + y0
    d = np.hypot(xs - x, ys - y)
    i = int(np.argmin(d))
    return (int(xs[i]), int(ys[i]), float(d[i])) if d[i] <= max_r else None


def shifted(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_NEAREST, borderValue=0)


def local_shift(pts_th, mask, search=SEARCH_PX, min_gain=MIN_GAIN):
    """Best integer (dx, dy) in +-search that raises Dice; (0,0) if gain too small."""
    sil, palm = silhouette(pts_th, mask.shape)
    sil_u8 = sil.astype(np.uint8)
    k = int(max(3, round(0.35 * palm))) | 1
    kern = np.ones((k, k), np.uint8)

    def score(dx, dy):
        s = shifted(sil_u8, dx, dy)
        near = cv2.dilate(s, kern) > 0
        b = mask & near
        den = int(s.sum()) + int(b.sum())
        return 2.0 * int(((s > 0) & b).sum()) / den if den else 0.0

    base = score(0, 0)
    best, bs = (0, 0), base
    for dy in range(-search, search + 1, 2):
        for dx in range(-search, search + 1, 2):
            sc = score(dx, dy)
            if sc > bs:
                best, bs = (dx, dy), sc
    cx, cy = best
    for dy in range(cy - 1, cy + 2):
        for dx in range(cx - 1, cx + 2):
            if abs(dx) <= search and abs(dy) <= search:
                sc = score(dx, dy)
                if sc > bs:
                    best, bs = (dx, dy), sc
    if bs - base < min_gain:
        return 0, 0, base, base
    return best[0], best[1], base, bs


def ask(prompt, default=None):
    shown = f" [{default}]" if default is not None else ""
    v = input(f"{prompt}{shown}: ").strip().strip('"').strip("'").strip()
    return v if v else (None if default is None else str(default))


def resolve_calibration_path(p):
    if p.is_file():
        return p
    if p.is_dir() and (p / "rig_calibration.json").is_file():
        return p / "rig_calibration.json"
    return None


UNITS = {"celsius": (1.0, 0.0), "kelvin": (1.0, -273.15),
         "decikelvin": (0.1, -273.15), "centikelvin": (0.01, -273.15)}


def main():
    csv_path = Path(ask("Path to joint_coordinates.csv"))
    rad_dir = Path(ask("Path to thermal_rad folder (.npy frames)"))
    rig_input = ask("Path to RIG calibration file or folder", "rig_calibration.json")
    rig_path = resolve_calibration_path(Path(rig_input))
    if rig_path is None:
        fb = rad_dir.parent / "alignment.json"
        if fb.is_file():
            print(f"[cal ] rig not found; using per-patient: {fb}")
            rig_path = fb
        else:
            raise SystemExit("No calibration found. Run 3_calibrate_alignment.py first.")
    print(f"[cal ] using calibration: {rig_path}")
    align = json.loads(rig_path.read_text())
    k = int(align["orientation"])
    M = np.array(align["matrix"])

    use_local = ask("Per-frame local re-alignment? [Y/n]", "Y").lower().startswith("y")
    use_snap = ask("Snap off-blob joints to nearest warm pixel? [Y/n]", "Y").lower().startswith("y")

    files = sorted(rad_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"No .npy files found in {rad_dir}")
    df = pd.read_csv(csv_path).reset_index(drop=True)
    n = len(df)
    n_csv_frames = int(df["frame"].max()) + 1 if n else 0
    if n_csv_frames > len(files):
        print(f"WARNING: CSV has {n_csv_frames} frames but only {len(files)} .npy files.")

    g_xy = apply_matrix(M, df[["x_px", "y_px"]].to_numpy())
    df["thermal_x_global"] = g_xy[:, 0].round(3)
    df["thermal_y_global"] = g_xy[:, 1].round(3)
    fin_xy = g_xy.copy()
    ldx = np.zeros(n); ldy = np.zeros(n)
    dice_g = np.full(n, np.nan); dice_l = np.full(n, np.nan)
    val = np.full(n, np.nan); med = np.full(n, np.nan)
    method = np.array(["none"] * n, dtype=object)
    snapd = np.full(n, np.nan)
    inblob = np.zeros(n, bool)
    groups = load_groups(df)
    kern3 = np.ones((3, 3), np.uint8)

    frames = df.groupby("frame").indices
    for c, (frame, idx) in enumerate(frames.items(), 1):
        fi = int(frame)
        if fi >= len(files):
            continue
        rad = dihedral(np.squeeze(np.load(files[fi])).astype(np.float32), k)
        mask = hand_mask(rad)
        core = cv2.erode(mask.astype(np.uint8), kern3) > 0
        if core.sum() < 50:
            core = mask
        hands = df["handedness"].to_numpy()[idx]
        for hand in np.unique(hands):
            rows = idx[hands == hand]
            key = (fi, str(hand))
            dx = dy = 0
            if key in groups:
                pts_g = apply_matrix(M, groups[key])
                if use_local:
                    dx, dy, d0, d1 = local_shift(pts_g, mask)
                else:
                    sil, palm = silhouette(pts_g, mask.shape)
                    d0 = d1 = dice_score(sil, mask, palm)
                dice_g[rows] = d0; dice_l[rows] = d1
            ldx[rows] = dx; ldy[rows] = dy
            fin_xy[rows, 0] = g_xy[rows, 0] + dx
            fin_xy[rows, 1] = g_xy[rows, 1] + dy

        xs, ys = fin_xy[idx, 0], fin_xy[idx, 1]
        val[idx] = bilinear_many(rad, xs, ys)
        for j, i in enumerate(idx):
            v = patch_median_in_mask(rad, mask, xs[j], ys[j])
            cx, cy = int(round(xs[j])), int(round(ys[j]))
            on = (0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0]
                  and bool(mask[cy, cx]))
            inblob[i] = on
            if np.isfinite(v):
                med[i] = v; method[i] = "direct"
            elif use_snap:
                hit = nearest_blob_pixel(core, xs[j], ys[j], SNAP_MAX_PX)
                if hit is not None:
                    v2 = patch_median_in_mask(rad, mask, hit[0], hit[1], r=1)
                    if np.isfinite(v2):
                        med[i] = v2; method[i] = "snapped"; snapd[i] = round(hit[2], 2)
        if c % 25 == 0 or c == len(frames):
            print(f"  processed {c}/{len(frames)} frames", end="\r")
    print()

    df["local_dx"] = ldx; df["local_dy"] = ldy
    df["thermal_x"] = fin_xy[:, 0].round(3)
    df["thermal_y"] = fin_xy[:, 1].round(3)
    df["radiometric_value"] = np.round(val, 4)
    df["radiometric_hand_median"] = np.round(med, 4)
    df["reading_method"] = method
    df["snap_dist_px"] = snapd
    df["in_hand_blob"] = inblob
    df["align_dice_global"] = np.round(dice_g, 3)
    df["align_dice"] = np.round(dice_l, 3)

    # ---- units ----
    mv = med[~np.isnan(med)]
    scale = None
    if len(mv):
        print(f"\nMedian joint value (raw): {np.nanmedian(mv):.3f}")
        choice = ask("Radiometric units [auto / celsius / kelvin / decikelvin / centikelvin]", "auto").lower()
        if choice == "auto":
            guess = to_celsius_params(mv)
            if guess:
                scale, offset, name = guess
                print(f"Auto-detected units: {name}")
            else:
                print("Could not guess units; temperature_c left blank.")
        elif choice in UNITS:
            scale, offset = UNITS[choice]
    df["temperature_c"] = np.round(med * scale + offset, 3) if scale is not None else np.nan

    # ---- plot-only temporal fill (never used by statistics) ----
    df["temperature_filled_c"] = df["temperature_c"]
    df["is_filled"] = False
    if scale is not None and TEMPORAL_LIMIT > 0:
        for _, grp in df.sort_values("frame").groupby(["handedness", "landmark_id"]):
            s = grp["temperature_c"]
            filled = s.interpolate(method="linear", limit=TEMPORAL_LIMIT, limit_area="inside")
            m = s.isna() & filled.notna()
            df.loc[grp.index[m.to_numpy()], "temperature_filled_c"] = filled[m].round(3)
            df.loc[grp.index[m.to_numpy()], "is_filled"] = True

    # ---- report ----
    print("\n=== Reading summary ===")
    for m_ in ["direct", "snapped", "none"]:
        c_ = int((df["reading_method"] == m_).sum())
        print(f"  {m_:8s}: {c_:6d} / {n} ({c_/n*100:5.1f}%)")
    dg, dl = np.nanmedian(dice_g), np.nanmedian(dice_l)
    print(f"  median Dice: global {dg:.3f} -> after local fix {dl:.3f}")
    if use_local:
        moved = (np.abs(ldx) + np.abs(ldy)) > 0
        print(f"  frames/hands shifted: {moved.mean()*100:.1f}% of rows; "
              f"mean |shift| = {np.hypot(ldx, ldy)[moved].mean() if moved.any() else 0:.2f} px")
    onp = float(inblob.mean())
    print(f"  joints on warm blob (final): {onp*100:.1f}%")
    if np.isfinite(dl) and dl < 0.6:
        print("  WARNING: median Dice < 0.6 even after local fix: the rig calibration "
              "or hand detection is poor for this recording. Re-run calibration "
              "or check the annotated frames.")
    if (df["reading_method"] == "none").mean() > 0.2:
        print("  WARNING: >20% of joints have no reading.")
        if ask("Proceed anyway? [y/N]", "N").lower() != "y":
            raise SystemExit("Aborted.")
    if scale is not None and df["temperature_c"].notna().any():
        t = df["temperature_c"].dropna()
        print(f"  temperature_c: min={t.min():.1f} median={t.median():.1f} max={t.max():.1f}")
    out = csv_path.with_name("joint_coordinates_radiometric.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out.resolve()}")


if __name__ == "__main__":
    main()