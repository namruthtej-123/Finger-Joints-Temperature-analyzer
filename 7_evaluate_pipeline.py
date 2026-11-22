"""
COMPONENT 7: 7_evaluate_pipeline.py   (standalone; run after 4_add_radiometric.py)

Evaluation + performance metrics for the hand-temperature pipeline.
Everything that needs ground truth is OPTIONAL and skipped when the file is absent.

  A. Coverage & data quality      reading rate (direct/snapped/none) per joint / frame,
                                  Dice before/after local re-alignment, hands per frame
  B. Alignment accuracy           needs ground_truth_thermal.csv (6_annotate_ground_truth.py)
                                  error in px (and mm if scale given): mean/median/RMSE/p90,
                                  PCK@2/3/5 px, per joint; for GLOBAL-only vs FINAL (local fix);
                                  ORACLE similarity fit to the clicks (best a single global
                                  transform could do) and 5-fold cross-validated version
  C. Temperature robustness       change in reading when the joint is moved +-1,2,3 px
  D. Temporal noise               robust frame-to-frame noise per joint (MAD of differences)
  E. Accuracy vs reference        optional reference_temperatures.csv (frame,hand,landmark_id,
                                  reference_c): bias, SD, 95% limits of agreement, MAE, r
  F. Performance                  ms/frame, frames/s, peak memory for the thermal stage

Outputs (folder evaluation_report/ next to the CSV): metrics.csv, per_joint_metrics.csv,
evaluation.json, and PNG plots.
Needs : pip install numpy pandas opencv-python matplotlib
"""
import json
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINGER_CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                 [13, 14, 15, 16], [17, 18, 19, 20]]
PALM = [0, 1, 5, 9, 13, 17]
SEARCH_PX = 6
MIN_GAIN = 0.03
SNAP_MAX_PX = 6

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


# ======================================================================
# Helpers copied from step 4 (inline on purpose: standalone file)
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


def style(ax, title, xl="", yl=""):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xl, color=INK2, fontsize=9)
    ax.set_ylabel(yl, color=INK2, fontsize=9)


def new_fig(w=7, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURF)
    return fig, ax


def err_stats(e, prefix):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {}
    return {f"{prefix}_n": int(e.size), f"{prefix}_mean_px": e.mean(),
            f"{prefix}_median_px": np.median(e), f"{prefix}_rmse_px": float(np.sqrt((e ** 2).mean())),
            f"{prefix}_p90_px": np.percentile(e, 90), f"{prefix}_max_px": e.max(),
            f"{prefix}_PCK@2px": (e <= 2).mean(), f"{prefix}_PCK@3px": (e <= 3).mean(),
            f"{prefix}_PCK@5px": (e <= 5).mean()}


def main():
    csv_path = Path(ask("Path to joint_coordinates_radiometric.csv"))
    patient = Path(ask("Press ## ENTER ## Path to PATIENT folder (contains thermal_rad)", str(csv_path.parent)))
    rig = Path(ask("Path to rig_calibration.json (orientation + matrix)", "rig_calibration.json"))
    gt_path = Path(ask("Ground-truth clicks CSV (Enter = skip)", str(patient / "ground_truth_thermal.csv")))
    ref_path = Path(ask("Reference temperatures CSV (Enter = skip)", str(patient / "reference_temperatures.csv")))
    mm_px = ask("Millimetres per thermal pixel at the hand (Enter = unknown)", "")
    mm_px = float(mm_px) if mm_px else None

    out = csv_path.parent / "evaluation_report"
    out.mkdir(exist_ok=True)
    df = pd.read_csv(csv_path)
    metrics = {}
    perjoint = {}
    df["joint_label"] = np.where(df["finger"].astype(str) == "wrist", "wrist",
                                 df["finger"].astype(str) + "_" + df["joint"].astype(str))
    order = df.drop_duplicates("landmark_id").sort_values("landmark_id")["joint_label"].tolist()

    # ------------------------------------------------ A. coverage
    n = len(df)
    rm = df["reading_method"] if "reading_method" in df else pd.Series("direct", index=df.index)
    for m in ("direct", "snapped", "none"):
        metrics[f"coverage_{m}_pct"] = 100 * (rm == m).mean()
    metrics["coverage_measured_pct"] = 100 * df["temperature_c"].notna().mean()
    metrics["rows"] = n
    metrics["frames"] = int(df["frame"].nunique())
    metrics["hands_per_frame"] = df.groupby("frame")["handedness"].nunique().mean()
    if "align_dice" in df:
        metrics["dice_median_global"] = df["align_dice_global"].median()
        metrics["dice_median_final"] = df["align_dice"].median()
        metrics["dice_below_0.6_pct"] = 100 * (df.groupby(["frame", "handedness"])["align_dice"]
                                               .first() < 0.6).mean()
    if "local_dx" in df:
        sh = np.hypot(df["local_dx"], df["local_dy"])
        metrics["local_shift_mean_px"] = sh.mean()
        metrics["local_shift_p90_px"] = np.percentile(sh, 90)
    cov = (df.assign(m=rm).pivot_table(index="joint_label", columns="m", values="frame",
                                        aggfunc="count", fill_value=0).reindex(order))
    for m in ("direct", "snapped", "none"):
        if m not in cov:
            cov[m] = 0
    cov_pct = cov[["direct", "snapped", "none"]].div(cov[["direct", "snapped", "none"]].sum(axis=1), axis=0) * 100
    for j in order:
        perjoint.setdefault(j, {}).update({"direct_pct": cov_pct.loc[j, "direct"],
                                           "snapped_pct": cov_pct.loc[j, "snapped"],
                                           "none_pct": cov_pct.loc[j, "none"]})
    fig, ax = new_fig(8, 4)
    x = np.arange(len(order))
    ax.bar(x, cov_pct["direct"], color=BLUE, label="direct", width=0.7)
    ax.bar(x, cov_pct["snapped"], bottom=cov_pct["direct"], color=YELLOW, label="snapped", width=0.7)
    ax.bar(x, cov_pct["none"], bottom=cov_pct["direct"] + cov_pct["snapped"], color=ORANGE, label="none", width=0.7)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=75, ha="right")
    style(ax, "Temperature reading source per joint", "", "% of rows")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "coverage_per_joint.png", dpi=150); plt.close(fig)

    # ------------------------------------------------ B. alignment accuracy
    if gt_path.is_file():
        gt = pd.read_csv(gt_path)
        gt = gt.rename(columns={"hand": "handedness"})
        m_ = df.merge(gt, on=["frame", "handedness", "landmark_id"], suffixes=("", "_gt"))
        if len(m_):
            eg = np.hypot(m_["thermal_x_global"] - m_["x_thermal"], m_["thermal_y_global"] - m_["y_thermal"]) \
                if "thermal_x_global" in m_ else None
            ef = np.hypot(m_["thermal_x"] - m_["x_thermal"], m_["thermal_y"] - m_["y_thermal"])
            if eg is not None:
                metrics.update(err_stats(eg, "align_global"))
            metrics.update(err_stats(ef, "align_final"))
            # oracle: best single similarity transform fitted to the clicks themselves
            src = m_[["x_px", "y_px"]].to_numpy(np.float32)
            dst = m_[["x_thermal", "y_thermal"]].to_numpy(np.float32)
            A, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
            if A is not None:
                eo = np.hypot(*(apply_matrix(A, src) - dst).T)
                metrics.update(err_stats(eo, "align_oracle_global_fit"))
                # 5-fold CV by frame
                frs = np.array(sorted(m_["frame"].unique()))
                rng = np.random.default_rng(0)
                rng.shuffle(frs)
                folds = np.array_split(frs, min(5, len(frs)))
                cv_err = []
                for f in folds:
                    te = m_["frame"].isin(f).to_numpy()
                    if te.all() or not te.any():
                        continue
                    Ak, _ = cv2.estimateAffinePartial2D(src[~te], dst[~te], method=cv2.LMEDS)
                    if Ak is not None:
                        cv_err.extend(np.hypot(*(apply_matrix(Ak, src[te]) - dst[te]).T))
                metrics.update(err_stats(cv_err, "align_oracle_cv"))
            if mm_px:
                metrics["align_final_mean_mm"] = metrics["align_final_mean_px"] * mm_px
                if "align_global_mean_px" in metrics:
                    metrics["align_global_mean_mm"] = metrics["align_global_mean_px"] * mm_px
            for lab, g in m_.assign(ef=ef).groupby("joint_label"):
                perjoint.setdefault(lab, {})["align_final_mean_px"] = g["ef"].mean()
            if eg is not None:
                for lab, g in m_.assign(eg=eg).groupby("joint_label"):
                    perjoint.setdefault(lab, {})["align_global_mean_px"] = g["eg"].mean()
            fig, ax = new_fig()
            bins = np.linspace(0, max(6, float(np.nanmax(ef))), 25)
            if eg is not None:
                ax.hist(eg, bins=bins, color=ORANGE, alpha=0.75, label="global calibration only")
            ax.hist(ef, bins=bins, color=BLUE, alpha=0.75, label="after local re-alignment")
            style(ax, "Joint position error vs hand-clicked ground truth", "error (thermal px)", "joints")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout(); fig.savefig(out / "alignment_error_hist.png", dpi=150); plt.close(fig)
        else:
            print("Ground truth found but no rows matched (frame/hand/landmark_id).")
    else:
        print("[B] no ground truth -> alignment accuracy skipped (run 6_annotate_ground_truth.py).")

    # ------------------------------------------------ C. sensitivity (needs rad frames)
    files = sorted((patient / "thermal_rad").glob("*.npy"))
    ra = json.loads(rig.read_text()) if rig.is_file() else None
    k = int(ra["orientation"]) if ra else None
    scale = offset = None
    if files and k is not None:
        frames = sorted(df["frame"].unique())
        rng = np.random.default_rng(1)
        pick = rng.choice(frames, size=min(10, len(frames)), replace=False)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        sens = {1: [], 2: [], 3: []}
        for fr in pick:
            if int(fr) >= len(files):
                continue
            rad = dihedral(np.squeeze(np.load(files[int(fr)])).astype(np.float32), k)
            if scale is None:
                guess = to_celsius_params(rad[rad > 0])
                if guess is None:
                    break
                scale, offset, _ = guess
            g = df[df["frame"] == fr]
            xs, ys = g["thermal_x"].to_numpy(), g["thermal_y"].to_numpy()
            base = bilinear_many(rad, xs, ys) * scale
            for r in sens:
                ch = []
                for dx, dy in dirs:
                    v = bilinear_many(rad, xs + r * dx, ys + r * dy) * scale
                    ch.append(np.abs(v - base))
                sens[r].append(np.nanmean(np.vstack(ch), axis=0))
        if scale is not None and sens[1]:
            rs = sorted(sens)
            vals = [np.nanmean(np.concatenate(sens[r])) for r in rs]
            for r, v in zip(rs, vals):
                metrics[f"temp_change_per_{r}px_shift_C"] = v
            fig, ax = new_fig(5.5, 3.6)
            ax.plot(rs, vals, color=BLUE, lw=2, marker="o", markersize=6, markeredgecolor=SURF)
            style(ax, "Reading change when joint is moved", "shift (thermal px)", "mean |dT| (°C)")
            ax.set_xticks(rs)
            fig.tight_layout(); fig.savefig(out / "temperature_sensitivity.png", dpi=150); plt.close(fig)
    else:
        print("[C] thermal_rad / rig file not found -> sensitivity skipped.")

    # ------------------------------------------------ D. temporal noise
    tc = "temperature_c"
    if df[tc].notna().any():
        noise = {}
        for (h, lab), g in df.sort_values("frame").groupby(["handedness", "joint_label"]):
            d = np.diff(g[tc].dropna().to_numpy())
            if len(d) >= 3:
                noise[(h, lab)] = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
        if noise:
            metrics["temporal_noise_median_C"] = float(np.median(list(noise.values())))
            for (h, lab), v in noise.items():
                perjoint.setdefault(lab, {})["temporal_noise_C"] = v

    # ------------------------------------------------ E. reference thermometer
    if ref_path.is_file():
        ref = pd.read_csv(ref_path).rename(columns={"hand": "handedness"})
        r_ = df.merge(ref, on=["frame", "handedness", "landmark_id"]).dropna(subset=[tc, "reference_c"])
        if len(r_) > 2:
            d = r_[tc] - r_["reference_c"]
            avg = (r_[tc] + r_["reference_c"]) / 2
            bias, sd = d.mean(), d.std()
            metrics.update({"ref_n": len(r_), "ref_bias_C": bias, "ref_sd_C": sd,
                            "ref_loa_low_C": bias - 1.96 * sd, "ref_loa_high_C": bias + 1.96 * sd,
                            "ref_MAE_C": d.abs().mean(), "ref_RMSE_C": float(np.sqrt((d ** 2).mean())),
                            "ref_pearson_r": float(np.corrcoef(r_[tc], r_["reference_c"])[0, 1])})
            fig, ax = new_fig(6, 3.8)
            ax.scatter(avg, d, s=14, color=BLUE, alpha=0.6, edgecolors="none")
            for y, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
                ax.axhline(y, color=INK2, lw=1, ls=ls)
            style(ax, "Bland-Altman: pipeline vs reference", "mean of the two (°C)", "pipeline - reference (°C)")
            fig.tight_layout(); fig.savefig(out / "bland_altman.png", dpi=150); plt.close(fig)
    else:
        print("[E] no reference temperatures -> accuracy vs reference skipped.")

    # ------------------------------------------------ F. performance
    if files and k is not None:
        n_t = min(20, len(files))
        tracemalloc.start()
        t_load = t_mask = t_local = t_samp = 0.0
        for fi in range(n_t):
            t0 = time.perf_counter()
            rad = dihedral(np.squeeze(np.load(files[fi])).astype(np.float32), k)
            t1 = time.perf_counter()
            mask = hand_mask(rad)
            t2 = time.perf_counter()
            g = df[df["frame"] == frames[min(fi, len(frames) - 1)]]
            if len(g) >= 21:
                pts = g.drop_duplicates("landmark_id").sort_values("landmark_id")[["thermal_x_global", "thermal_y_global"]] \
                    .to_numpy(np.float32)[:21] if "thermal_x_global" in g else \
                    g.sort_values("landmark_id")[["thermal_x", "thermal_y"]].to_numpy(np.float32)[:21]
                local_shift(pts, mask)
            t3 = time.perf_counter()
            xs, ys = g["thermal_x"].to_numpy(), g["thermal_y"].to_numpy()
            bilinear_many(rad, xs, ys)
            for x_, y_ in zip(xs, ys):
                patch_median_in_mask(rad, mask, x_, y_)
            t4 = time.perf_counter()
            t_load += t1 - t0; t_mask += t2 - t1; t_local += t3 - t2; t_samp += t4 - t3
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tot = (t_load + t_mask + t_local + t_samp) / n_t
        metrics.update({"perf_load_ms_per_frame": 1000 * t_load / n_t,
                        "perf_mask_ms_per_frame": 1000 * t_mask / n_t,
                        "perf_local_align_ms_per_frame": 1000 * t_local / n_t,
                        "perf_sampling_ms_per_frame": 1000 * t_samp / n_t,
                        "perf_total_ms_per_frame": 1000 * tot,
                        "perf_frames_per_second": 1 / tot if tot else float("nan"),
                        "perf_peak_python_memory_MB": peak / 1e6})

    # ------------------------------------------------ save
    mt = pd.DataFrame({"metric": list(metrics), "value": [float(v) for v in metrics.values()]})
    mt["value"] = mt["value"].round(4)
    mt.to_csv(out / "metrics.csv", index=False)
    pj = pd.DataFrame(perjoint).T.reindex(order).round(4)
    pj.index.name = "joint"
    pj.to_csv(out / "per_joint_metrics.csv")
    (out / "evaluation.json").write_text(json.dumps({k_: float(v) for k_, v in metrics.items()}, indent=2))
    print("\n=== Evaluation summary ===")
    print(mt.to_string(index=False))
    print(f"\nSaved -> {out.resolve()}")


if __name__ == "__main__":
    main()