#!/usr/bin/env python3
"""
Corrosion progression analysis for time-series component images.

Naming convention expected in --dir:
    <Component Name>-T<hours>[-<shot>].jpg
    e.g. "Nut Silver-T0.jpg", "Lock Nut Silver-T48-2.jpg"

T0  = healthy baseline  (used to calibrate component-specific pixel threshold)
T24 = 24 h of exposure
T48 = 48 h of exposure
T72 = 72 h of exposure

Outputs (in --output directory):
    report.json          — full structured data
    summary.txt          — human-readable table
    plots/               — progression line chart + per-component heatmap grids

Usage:
    python corrosion_analysis.py --model saved_models/patchcore_fastener.pkl
    python corrosion_analysis.py --model saved_models/patchcore_fastener.pkl \
        --dir data/components --output outputs/corrosion_report
"""

import argparse, re, sys, cv2, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import cfg
from preprocessing.pipeline import FastenerPreprocessor
from models.patchcore import PatchCore


TIMEPOINTS = [0, 24, 48, 72]
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# HSV range for rust / oxidation (orange-brown)
RUST_H_LO, RUST_H_HI = 5,  25
RUST_S_LO             = 45
RUST_V_LO, RUST_V_HI = 40, 210


# ─── filename parsing ────────────────────────────────────────────────────────

def parse_name(path: Path):
    """Return (component, timepoint_hours, shot_index) or (None,None,None)."""
    m = re.match(r"^(.+?)-T(\d+)(?:-(\d+))?$", path.stem, re.IGNORECASE)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3) or 0)


def group_images(directory: Path) -> Dict[str, Dict[int, List[Path]]]:
    """Group image paths by component name and timepoint."""
    groups: Dict[str, Dict[int, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for p in directory.iterdir():
        if p.suffix.lower() not in IMG_EXTS:
            continue
        comp, tp, _ = parse_name(p)
        if comp is None:
            print(f"  [SKIP] unrecognised filename: {p.name}")
            continue
        groups[comp][tp].append(p)
    # Sort shots within each group
    for comp in groups:
        for tp in groups[comp]:
            groups[comp][tp].sort()
    return groups


# ─── analysis helpers ─────────────────────────────────────────────────────────

def colour_rust_pct(orig_bgr: np.ndarray, fg_mask: np.ndarray) -> float:
    """
    Percentage of foreground pixels that fall in the rust HSV colour range.
    Works on the *original* (unmasked) BGR image + the foreground mask.
    """
    if fg_mask.sum() == 0:
        return 0.0
    hsv = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    rust_mask = (
        (h >= RUST_H_LO) & (h <= RUST_H_HI) &
        (s >= RUST_S_LO) &
        (v >= RUST_V_LO) & (v <= RUST_V_HI)
    )
    fg = fg_mask > 0
    return float((rust_mask & fg).sum() / fg.sum() * 100)


def heatmap_growth_pct(heatmap: np.ndarray, t0_mean_hm: np.ndarray,
                       t0_std_hm: np.ndarray, fg_mask: np.ndarray,
                       sigma: float = 1.5) -> float:
    """
    Percentage of foreground pixels where heatmap score has grown by more than
    `sigma` standard-deviations above the component's own T0 mean heatmap.

    This is a T0-relative metric: at T0 it is ~7 % by definition of the
    normal-distribution tail; at T24/T48/T72 it rises where rust has formed.
    It removes the absolute-scale problem of comparing to training-domain
    thresholds.
    """
    if fg_mask.sum() == 0:
        return 0.0
    pixel_thresh = t0_mean_hm + sigma * t0_std_hm
    fg = fg_mask > 0
    grown = (heatmap > pixel_thresh) & fg
    return float(grown.sum() / fg.sum() * 100)


def analyse_image(model: PatchCore, pp: FastenerPreprocessor,
                  path: Path) -> Optional[dict]:
    """Preprocess, run model, return per-image metrics dict."""
    raw = cv2.imread(str(path))
    if raw is None:
        print(f"  [WARN] cannot read {path.name}")
        return None

    result = pp.process_file(path)
    if not result.success:
        print(f"  [WARN] preprocess failed for {path.name}: {result.error}")
        return None

    score, heatmap = model.predict(result.image)
    is_anom = score > model.threshold if model.threshold > 0 else False

    # Resize fg mask to heatmap size for pixel calcs
    fg_orig = result.mask  # (H, W) uint8
    fg_heat = cv2.resize(fg_orig,
                         (heatmap.shape[1], heatmap.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    # Colour-based rust on original image (resized to match mask)
    raw_resized = cv2.resize(raw, (fg_orig.shape[1], fg_orig.shape[0]))
    c_rust = colour_rust_pct(raw_resized, fg_orig)

    return {
        "path":        str(path),
        "score":       float(score),
        "is_anomalous": bool(is_anom),
        "heatmap":     heatmap,          # numpy array, not serialised to JSON
        "fg_heat":     fg_heat,          # aligned foreground mask
        "colour_rust_pct": c_rust,
    }


# ─── per-component time-series analysis ──────────────────────────────────────

def analyse_component(name: str,
                      timepoints_data: Dict[int, List[dict]]) -> dict:
    """
    Compute time-series metrics for one component.

    Primary metric  — colour_rust_pct:
        Fraction of foreground pixels in the rust HSV colour range.
        Independent of the PatchCore training domain.

    Secondary metric — heatmap_growth_pct:
        Fraction of foreground pixels whose anomaly heatmap score exceeds the
        component's own T0 mean by >1.5 std-devs.  This measures RELATIVE
        deterioration, not absolute anomaly level, so it works even when the
        model flags all images as anomalous.

    is_rusting flag:
        Set True when colour_rust_pct is >2 pp above the T0 baseline, or when
        heatmap_growth_pct is >5 pp above T0, whichever fires first.
    """
    result = {"component": name, "timepoints": {}}

    # ── Build T0 reference heatmap (mean ± std per pixel) ──────────────────
    t0_items = [x for x in timepoints_data.get(0, []) if x]
    if t0_items:
        # Stack all T0 heatmaps (same spatial shape after preprocessing)
        t0_hm_stack = np.stack([x["heatmap"] for x in t0_items], axis=0)
        t0_mean_hm  = t0_hm_stack.mean(axis=0)
        t0_std_hm   = t0_hm_stack.std(axis=0).clip(min=0.1)  # avoid zero-std

        # T0 reference foreground mask (union of all T0 shots)
        t0_fgs = [x["fg_heat"] for x in t0_items]
        t0_fg  = (np.stack(t0_fgs, axis=0).mean(axis=0) > 0).astype(np.uint8) * 255

        t0_c_pct  = float(np.mean([x["colour_rust_pct"] for x in t0_items]))
        t0_sc     = float(np.mean([x["score"]           for x in t0_items]))
    else:
        t0_mean_hm = t0_std_hm = t0_fg = None
        t0_c_pct   = 0.0
        t0_sc      = 0.0

    result["t0_colour_rust_pct"] = round(t0_c_pct, 2)

    prev_c_pct  = t0_c_pct
    prev_hg_pct = 0.0  # T0 growth by definition starts at reference level ~7%

    for tp in TIMEPOINTS:
        items = [x for x in timepoints_data.get(tp, []) if x]
        if not items:
            result["timepoints"][tp] = None
            continue

        scores  = [x["score"]            for x in items]
        c_rusts = [x["colour_rust_pct"]  for x in items]

        avg_sc = float(np.mean(scores))
        c_pct  = float(np.mean(c_rusts))

        # T0-relative heatmap growth %
        if t0_mean_hm is not None:
            hg_pcts = [
                heatmap_growth_pct(x["heatmap"], t0_mean_hm, t0_std_hm,
                                   t0_fg if t0_fg is not None else x["fg_heat"])
                for x in items
            ]
            hg_pct = float(np.mean(hg_pcts))
        else:
            hg_pct = 0.0

        # Colour delta vs T0 baseline
        c_delta_t0   = round(c_pct  - t0_c_pct,  2)
        c_delta_prev = round(c_pct  - prev_c_pct, 2)

        # T0-relative heatmap growth reference: what was the growth % at T0?
        # (computed separately so we can compute delta_vs_t0 for hm growth)
        if tp == 0:
            hg_t0_ref = hg_pct  # record T0's own growth value as reference
            result["t0_hg_pct"] = round(hg_pct, 2)
        else:
            hg_t0_ref = result.get("t0_hg_pct", 0.0)

        hg_delta_t0 = round(hg_pct - hg_t0_ref, 2)

        # is_rusting: colour OR heatmap growth significantly above T0
        if tp == 0:
            is_rusting = False
            status = "BASELINE"
        else:
            is_rusting = (c_delta_t0 > 2.0) or (hg_delta_t0 > 5.0)
            status = "RUSTING" if is_rusting else "STABLE"

        tp_data = {
            "mean_score":          round(avg_sc, 3),
            "colour_rust_pct":     round(c_pct, 2),
            "colour_delta_vs_t0":  c_delta_t0,
            "colour_delta_vs_prev": c_delta_prev,
            "heatmap_growth_pct":  round(hg_pct, 2),
            "heatmap_growth_delta_vs_t0": hg_delta_t0,
            "is_rusting":          is_rusting,
            "status":              status,
            "shots":               len(items),
        }

        result["timepoints"][tp] = tp_data
        prev_c_pct  = c_pct
        prev_hg_pct = hg_pct

    return result


# ─── output helpers ───────────────────────────────────────────────────────────

SEVERITY_THRESHOLDS = [(0, "None"), (2, "Trace"), (10, "Mild"),
                       (25, "Moderate"), (50, "Severe"), (100, "Critical")]

def severity(delta_pct: float) -> str:
    """Rate corrosion severity based on colour rust % INCREASE above T0."""
    label = "None"
    for thr, lbl in SEVERITY_THRESHOLDS:
        if delta_pct >= thr:
            label = lbl
    return label


def _fmt_delta(v: float) -> str:
    return f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%"


def print_report(components: List[dict]):
    w = 100
    print()
    print("=" * w)
    print("  CORROSION PROGRESSION REPORT")
    print("  Primary:   Colour Rust %  — orange/brown HSV pixels in component foreground")
    print("  Secondary: HM Growth %    — foreground pixels exceeding T0 heatmap baseline by >1.5σ")
    print("=" * w)

    for comp in components:
        name    = comp["component"]
        t0_c    = comp.get("t0_colour_rust_pct", 0.0)
        print(f"\n  Component : {name}   (T0 baseline colour rust: {t0_c:.1f}%)")
        print(f"  {'Time':<6} {'Score':>8} {'Colour%':>9} {'ColΔT0':>8} {'ColΔPrev':>9} "
              f"{'HMGrow%':>9} {'HMΔvsT0':>8} {'Status':<10} {'Severity'}")
        print("  " + "-" * 88)

        for tp in TIMEPOINTS:
            tpd = comp["timepoints"].get(tp)
            if tpd is None:
                print(f"  T{tp:<5} {'—':>8}")
                continue

            if tpd["status"] == "BASELINE":
                status_str = "\033[94mBASELINE\033[0m"
            elif tpd["is_rusting"]:
                status_str = "\033[91mRUSTING \033[0m"
            else:
                status_str = "\033[92mSTABLE  \033[0m"

            sev = severity(tpd["colour_delta_vs_t0"])
            print(f"  T{tp:<5} {tpd['mean_score']:>8.3f} "
                  f"{tpd['colour_rust_pct']:>8.1f}% "
                  f"{_fmt_delta(tpd['colour_delta_vs_t0']):>8} "
                  f"{_fmt_delta(tpd['colour_delta_vs_prev']):>9} "
                  f"{tpd['heatmap_growth_pct']:>8.1f}% "
                  f"{_fmt_delta(tpd['heatmap_growth_delta_vs_t0']):>8}  "
                  f"{status_str}  {sev}")

    print()
    print("=" * w)


def write_summary_txt(components: List[dict], out_path: Path):
    lines = [
        "CORROSION PROGRESSION REPORT",
        "Primary:   Colour Rust %  — orange/brown HSV pixels in component foreground",
        "Secondary: HM Growth %    — foreground pixels exceeding T0 heatmap baseline by >1.5σ",
        "=" * 100, "",
    ]
    for comp in components:
        name = comp["component"]
        t0_c = comp.get("t0_colour_rust_pct", 0.0)
        lines.append(f"Component: {name}   (T0 baseline colour rust: {t0_c:.1f}%)")
        lines.append(f"  {'Time':<6} {'Score':>8} {'Colour%':>9} {'ColΔT0':>8} {'ColΔPrev':>9} "
                     f"{'HMGrow%':>9} {'HMΔvsT0':>8} {'Status':<10} {'Severity'}")
        lines.append("  " + "-" * 88)
        for tp in TIMEPOINTS:
            tpd = comp["timepoints"].get(tp)
            if tpd is None:
                lines.append(f"  T{tp:<5} —")
                continue
            sev = severity(tpd["colour_delta_vs_t0"])
            lines.append(f"  T{tp:<5} {tpd['mean_score']:>8.3f} "
                         f"{tpd['colour_rust_pct']:>8.1f}% "
                         f"{_fmt_delta(tpd['colour_delta_vs_t0']):>8} "
                         f"{_fmt_delta(tpd['colour_delta_vs_prev']):>9} "
                         f"{tpd['heatmap_growth_pct']:>8.1f}% "
                         f"{_fmt_delta(tpd['heatmap_growth_delta_vs_t0']):>8}  "
                         f"{tpd['status']:<10} {sev}")
        lines.append("")
    out_path.write_text("\n".join(lines))


# ─── plotting ────────────────────────────────────────────────────────────────

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2"]


def plot_progression(components: List[dict], out_path: Path):
    """2-panel line chart: colour rust % and T0-relative heatmap growth %."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_cl, ax_hg = axes

    for i, comp in enumerate(components):
        tps, c_pcts, hg_pcts = [], [], []
        for tp in TIMEPOINTS:
            tpd = comp["timepoints"].get(tp)
            if tpd:
                tps.append(tp)
                c_pcts.append(tpd["colour_rust_pct"])
                hg_pcts.append(tpd["heatmap_growth_pct"])
        col = COLORS[i % len(COLORS)]
        ax_cl.plot(tps, c_pcts,  marker="o", color=col,
                   label=comp["component"], linewidth=2)
        ax_hg.plot(tps, hg_pcts, marker="s", color=col,
                   label=comp["component"], linewidth=2, linestyle="--")

    for ax, title, ylabel in [
        (ax_cl, "Colour-Based Rust Area %\n(orange/brown HSV pixels)",
         "Rust-Coloured Area (%)"),
        (ax_hg, "Heatmap Growth vs T0 Baseline\n(pixels >1.5σ above T0 mean)",
         "Anomalous Growth Area (%)"),
    ]:
        ax.set_xlabel("Exposure (hours)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(TIMEPOINTS)
        ax.set_xticklabels([f"T{t}h" for t in TIMEPOINTS])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(bottom=0)

    fig.suptitle("Corrosion Progression — All Components",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_component_heatmaps(name: str,
                             tp_items: Dict[int, List[dict]],
                             out_path: Path):
    """
    2×4 grid: top row = preprocessed images, bottom row = anomaly heatmaps
    one column per timepoint (T0 T24 T48 T72).
    """
    n_tp = len(TIMEPOINTS)
    fig, axes = plt.subplots(2, n_tp, figsize=(4 * n_tp, 8))
    fig.suptitle(f"{name} — Corrosion Heatmaps Over Time",
                 fontsize=13, fontweight="bold")

    for col, tp in enumerate(TIMEPOINTS):
        items = [x for x in tp_items.get(tp, []) if x]
        ax_img = axes[0, col]
        ax_hm  = axes[1, col]
        ax_img.set_title(f"T{tp}h", fontsize=11)

        if not items:
            ax_img.axis("off")
            ax_hm.axis("off")
            ax_img.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax_img.transAxes)
            continue

        # Average heatmap across shots
        avg_hm = np.mean([x["heatmap"] for x in items], axis=0)
        # Show first shot's preprocessed image (RGB for matplotlib)
        result_img = cv2.imread(str(items[0]["path"]))
        if result_img is not None:
            # Reload and preprocess just for display
            ax_img.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        ax_img.axis("off")

        # Heatmap with foreground contour overlay
        im = ax_hm.imshow(avg_hm, cmap="hot", interpolation="bilinear")
        ax_hm.axis("off")
        plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

        # Annotate with score and colour rust %
        tpd_list = [x for x in items if x]
        if tpd_list:
            mean_score = np.mean([x["score"] for x in tpd_list])
            c_rust_avg = np.mean([x["colour_rust_pct"] for x in tpd_list])
            ax_hm.set_xlabel(f"score={mean_score:.2f}  colour_rust={c_rust_avg:.1f}%",
                             fontsize=7)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_delta_bars(components: List[dict], out_path: Path):
    """Grouped bar chart: colour rust % change vs T0 at each interval."""
    intervals = ["T24 vs T0", "T48 vs T0", "T72 vs T0"]
    tps_shown = [24, 48, 72]
    n_comp    = len(components)
    x         = np.arange(len(intervals))
    bar_w     = 0.8 / n_comp

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, comp in enumerate(components):
        deltas = []
        for tp in tps_shown:
            tpd = comp["timepoints"].get(tp)
            deltas.append(tpd["colour_delta_vs_t0"] if tpd else 0.0)
        offset = (i - n_comp / 2 + 0.5) * bar_w
        ax.bar(x + offset, deltas, bar_w, label=comp["component"],
               color=COLORS[i % len(COLORS)], alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(intervals, fontsize=11)
    ax.set_ylabel("Colour Rust % Change vs T0 Baseline", fontsize=11)
    ax.set_title("Corrosion Growth vs Healthy Baseline (Colour Method)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────────

model_thresh_fallback = 0.0   # updated once model is loaded


def main():
    global model_thresh_fallback

    pa = argparse.ArgumentParser()
    pa.add_argument("--model",  default="saved_models/patchcore_fastener.pkl")
    pa.add_argument("--dir",    default="data/components")
    pa.add_argument("--output", default=None)
    pa.add_argument("--device", default=None)
    pa.add_argument("--no-plots", action="store_true")
    args = pa.parse_args()

    out_dir = Path(args.output) if args.output else cfg.paths.output_dir / "corrosion_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load model
    device = args.device or cfg.training.device
    model  = PatchCore.load(args.model, device)
    model_thresh_fallback = model.threshold

    # Preprocessor
    pp = FastenerPreprocessor(
        target_size=cfg.preprocess.image_size,
        clahe_clip_limit=cfg.preprocess.clahe_clip_limit,
        clahe_tile_grid=cfg.preprocess.clahe_tile_grid,
        preserve_color=cfg.preprocess.preserve_color,
        align=cfg.preprocess.align_to_canonical,
    )

    # Group images
    comp_dir = Path(args.dir)
    groups   = group_images(comp_dir)
    if not groups:
        print(f"[ERROR] No component images found in {comp_dir}"); sys.exit(1)

    print(f"\n  Found {len(groups)} components in {comp_dir}")

    # ── analyse each image ────────────────────────────────────────────────────
    # raw_results[component][timepoint] = list of analyse_image() dicts
    raw_results: Dict[str, Dict[int, List[Optional[dict]]]] = {}

    for comp_name in sorted(groups):
        raw_results[comp_name] = {}
        print(f"\n  ── {comp_name} ──")
        for tp in sorted(groups[comp_name]):
            paths  = groups[comp_name][tp]
            items  = []
            for p in paths:
                r = analyse_image(model, pp, p)
                items.append(r)
                tag = "\033[91mANOMALY\033[0m" if (r and r["is_anomalous"]) else "\033[92mNORMAL \033[0m"
                sc  = f"{r['score']:.3f}" if r else "—"
                cr  = f"{r['colour_rust_pct']:.1f}%" if r else "—"
                print(f"    T{tp:>2}h  {tag}  score={sc}  colour_rust={cr}  {p.name}")
            raw_results[comp_name][tp] = items

    # ── per-component time-series metrics ─────────────────────────────────────
    component_reports = []
    for comp_name in sorted(raw_results):
        cr = analyse_component(comp_name, raw_results[comp_name])
        component_reports.append(cr)

    # ── console table ─────────────────────────────────────────────────────────
    print_report(component_reports)

    # ── text summary ──────────────────────────────────────────────────────────
    txt_path = out_dir / "summary.txt"
    write_summary_txt(component_reports, txt_path)
    print(f"  Summary → {txt_path}")

    # ── JSON report (strip numpy arrays) ──────────────────────────────────────
    def serialisable(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, dict):           return {k: serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):           return [serialisable(v) for v in obj]
        return obj

    json_out = {
        "model": args.model,
        "model_threshold": float(model.threshold),
        "components": [serialisable(c) for c in component_reports],
    }
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(json_out, indent=2))
    print(f"  JSON   → {json_path}")

    # ── plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n  Generating plots...")

        # 1. Progression line chart
        plot_progression(component_reports,
                         plots_dir / "progression_all.png")
        print(f"    progression_all.png")

        # 2. Delta bar chart
        plot_delta_bars(component_reports,
                        plots_dir / "delta_bars.png")
        print(f"    delta_bars.png")

        # 3. Per-component heatmap grids
        for comp_name in sorted(raw_results):
            safe = re.sub(r"[^\w\-]", "_", comp_name)
            plot_component_heatmaps(
                comp_name,
                raw_results[comp_name],
                plots_dir / f"{safe}_heatmaps.png",
            )
            print(f"    {safe}_heatmaps.png")

        print(f"\n  Plots → {plots_dir}")

    print(f"\n  DONE\n")


if __name__ == "__main__":
    main()
