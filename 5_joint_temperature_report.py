"""
COMPONENT 4: joint_temperature_report.py   (standalone, run per patient)

Reads joint_coordinates_radiometric.csv (from add_radiometric.py) and writes,
for EACH hand, a folder named lefthand_temperature_report / righthand_temperature_report
(next to the input CSV) containing exactly three files:

  <hand>hand_temperature_vs_frame.png      how the temperature at every joint varies with frame
  <hand>hand_heatmap.png                   all joints x all frames at a glance
  <hand>hand_joint_temperature_summary.csv one row per joint:
        joint name, finger, landmark id, frames used,
        max / min / mean / median / std dev / variance of temperature,
        mean joint location in the RGB image and in the thermal image,
        and where (frame + RGB x,y + thermal x,y) the max temperature occurred
Needs : pip install numpy pandas matplotlib
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- palette (validated default categorical order, light mode) ----
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # blue, orange, aqua, yellow
FINGERS = ["wrist", "thumb", "index", "middle", "ring", "little"]


def ask(prompt, default=None):
    shown = f" [{default}]" if default is not None else ""
    v = input(f"{prompt}{shown}: ").strip().strip('"').strip("'").strip()
    return v if v else (None if default is None else str(default))


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ------------------------------ charts ---------------------------------
def plot_vs_frame(mat, labels, hand, out, x_label):
    """mat: DataFrame frames x joint_label (temperature)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, facecolor=SURFACE)
    lo, hi = np.nanmin(mat.values), np.nanmax(mat.values)
    pad = max(0.3, (hi - lo) * 0.08)
    for ax, finger in zip(axes.ravel(), FINGERS):
        style_axes(ax)
        cols = [l for l in labels if l.split("_")[0] == finger]
        for i, c in enumerate(cols):
            name = c.split("_", 1)[1] if "_" in c else c
            ax.plot(mat.index, mat[c], color=SERIES[i % 4], linewidth=1.6,
                    label=name if len(cols) > 1 else None)
        ax.set_title(finger.capitalize(), color=INK, fontsize=11, loc="left")
        ax.set_ylim(lo - pad, hi + pad)
        if len(cols) > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=4,
                      loc="lower right", bbox_to_anchor=(1.0, 1.0))
    for ax in axes[1]:
        ax.set_xlabel(x_label, color=INK2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature (°C)", color=INK2, fontsize=9)
    fig.suptitle(f"{hand} hand - joint temperature vs frame", color=INK,
                 fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor=SURFACE); plt.close(fig)


def plot_heatmap(mat, labels, hand, out):
    cmap = LinearSegmentedColormap.from_list("seq", ["#e9f1fc", "#2a78d6", "#0b2f66"])
    cmap.set_bad("#d6d5d0")
    data = np.ma.masked_invalid(mat[labels].to_numpy().T)
    fig, ax = plt.subplots(figsize=(13, 7), facecolor=SURFACE)
    f0, f1 = mat.index.min(), mat.index.max()
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest",
                   extent=(f0 - 0.5, f1 + 0.5, len(labels) - 0.5, -0.5))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.replace("_", " ") for l in labels], fontsize=7.5, color=INK2)
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("Temperature (°C)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8); cb.outline.set_visible(False)
    ax.set_xlabel("Frame", color=INK2, fontsize=9)
    ax.set_title(f"{hand} hand - temperature of every joint in every frame (grey = no reading)",
                 color=INK, fontsize=12, loc="left")
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor=SURFACE); plt.close(fig)


# ------------------------------- main ----------------------------------
def main():
    csv_path = Path(ask("Path to joint_coordinates_radiometric.csv"))
    df = pd.read_csv(csv_path)

    temp_col = "temperature_c"
    if temp_col not in df or df[temp_col].notna().sum() == 0:
        temp_col = "radiometric_hand_median"
        print("No temperature_c values; using radiometric_hand_median (raw units).")
    unit = "\u00b0C" if temp_col == "temperature_c" else "raw units"

    df["joint_label"] = np.where(df["finger"].astype(str) == "wrist", "wrist",
                                 df["finger"].astype(str) + "_" + df["joint"].astype(str))
    order = df.drop_duplicates("landmark_id").sort_values("landmark_id")["joint_label"].tolist()
    # Statistics use MEASURED readings only (direct or snapped); filled/imputed values are excluded.
    if "is_imputed" in df:
        df.loc[df["is_imputed"].astype(bool), temp_col] = np.nan
    df["used"] = df[temp_col].notna()
    if "reading_method" not in df:
        df["reading_method"] = "direct"

    for hand, h in df.groupby("handedness"):
        tag = f"{str(hand).lower()}hand"
        out_dir = csv_path.parent / f"{tag}_temperature_report"
        out_dir.mkdir(exist_ok=True)
        used = h[h["used"]]

        rows = []
        for label in order:
            g_all = h[h["joint_label"] == label]
            g = used[used["joint_label"] == label]
            first = g_all.iloc[0]
            base = {"hand": hand, "joint": label, "finger": first["finger"],
                    "landmark_id": int(first["landmark_id"]),
                    "frames_total": g_all["frame"].nunique(),
                    "frames_with_reading": g["frame"].nunique(),
                    "frames_direct": int((g["reading_method"] == "direct").sum()),
                    "frames_snapped": int((g["reading_method"] == "snapped").sum())}
            if g.empty:
                rows.append(base); continue
            top = g.loc[g[temp_col].idxmax()]
            rows.append({**base,
                "max_temp_c": round(top[temp_col], 3),
                "min_temp_c": round(g[temp_col].min(), 3),
                "mean_temp_c": round(g[temp_col].mean(), 3),
                "median_temp_c": round(g[temp_col].median(), 3),
                "std_dev_temp_c": round(g[temp_col].std(), 4),
                "variance_temp_c": round(g[temp_col].var(), 4),
                "mean_x_rgb": round(g["x_px"].mean(), 2), "mean_y_rgb": round(g["y_px"].mean(), 2),
                "mean_x_thermal": round(g["thermal_x"].mean(), 2),
                "mean_y_thermal": round(g["thermal_y"].mean(), 2),
                "frame_at_max": int(top["frame"]),
                "x_rgb_at_max": int(top["x_px"]), "y_rgb_at_max": int(top["y_px"]),
                "x_thermal_at_max": round(top["thermal_x"], 2),
                "y_thermal_at_max": round(top["thermal_y"], 2)})
        summ = pd.DataFrame(rows)
        summ.to_csv(out_dir / f"{tag}_joint_temperature_summary.csv", index=False)

        mat = used.pivot_table(index="frame", columns="joint_label", values=temp_col, aggfunc="first")
        mat = mat.reindex(columns=order).reindex(sorted(h["frame"].unique()))
        if mat.notna().any().any():
            plot_vs_frame(mat, order, str(hand), out_dir / f"{tag}_temperature_vs_frame.png", "Frame")
            plot_heatmap(mat, order, str(hand), out_dir / f"{tag}_heatmap.png")

        print(f"\n=== {hand} hand: {h['frame'].nunique()} frames ({unit}) ===")
        ok = summ.dropna(subset=["max_temp_c"]) if "max_temp_c" in summ else summ.iloc[0:0]
        if not ok.empty:
            show = ok[["joint", "max_temp_c", "min_temp_c", "mean_temp_c", "median_temp_c",
                       "std_dev_temp_c", "variance_temp_c"]]
            show.columns = ["joint", "max", "min", "mean", "median", "std dev", "variance"]
            print(show.to_string(index=False))
        print("Saved:", out_dir.resolve())


if __name__ == "__main__":
    main()