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
from evaluation.visualize import heatmap_overlay, plot_score_distribution


TIMEPOINTS = [0, 24, 48, 72]
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ─── HSV colour bands for damage classification ───────────────────────────────
# OpenCV HSV scale: H=0-179, S=0-255, V=0-255.

# Iron rust / orange-brown oxidation (H ≈ 8-25 on 0-179 scale)
# S_LO=100 eliminates warm-light reflections on shiny surfaces.
RUST_H_LO, RUST_H_HI = 8,  25
RUST_S_LO             = 100
RUST_V_LO, RUST_V_HI = 40, 210

# Yellow-brown tarnish / sulfide discoloration (H ≈ 26-45).
# Distinct from orange rust; lower S cutoff to catch dull tarnish.
TARNISH_H_LO, TARNISH_H_HI = 26, 45
TARNISH_S_LO                = 40
TARNISH_V_LO, TARNISH_V_HI = 30, 210

# Green / blue-green oxidation (verdigris, copper-base alloys) H ≈ 55-100.
GREEN_OX_H_LO, GREEN_OX_H_HI = 55, 100
GREEN_OX_S_LO                 = 35
GREEN_OX_V_LO, GREEN_OX_V_HI = 30, 210

# Severity thresholds for the heatmap-growth channel (calibrated independently
# from colour-delta thresholds, which don't fire on dark/coated components).
HM_SEVERITY_THRESHOLDS = [
    (0,   "None"),
    (5,   "Trace"),
    (15,  "Mild"),
    (30,  "Moderate"),
    (60,  "Severe"),
    (100, "Critical"),
]


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

def _colour_pct(hsv: np.ndarray, fg_mask: np.ndarray,
                h_lo: int, h_hi: int, s_lo: int, v_lo: int, v_hi: int) -> float:
    """Fraction (%) of foreground pixels matching an HSV band."""
    fg = fg_mask > 0
    if not fg.any():
        return 0.0
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    band = (h >= h_lo) & (h <= h_hi) & (s >= s_lo) & (v >= v_lo) & (v <= v_hi)
    return float((band & fg).sum() / fg.sum() * 100)


def colour_rust_pct(orig_bgr: np.ndarray, fg_mask: np.ndarray) -> float:
    """% of fg pixels in the iron-rust (orange-brown) HSV range."""
    if fg_mask.sum() == 0:
        return 0.0
    hsv = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
    return _colour_pct(hsv, fg_mask,
                       RUST_H_LO, RUST_H_HI, RUST_S_LO, RUST_V_LO, RUST_V_HI)


def colour_tarnish_pct(orig_bgr: np.ndarray, fg_mask: np.ndarray) -> float:
    """% of fg pixels in the yellow-brown tarnish / sulfide-discoloration HSV range."""
    if fg_mask.sum() == 0:
        return 0.0
    hsv = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
    return _colour_pct(hsv, fg_mask,
                       TARNISH_H_LO, TARNISH_H_HI, TARNISH_S_LO,
                       TARNISH_V_LO, TARNISH_V_HI)


def colour_green_ox_pct(orig_bgr: np.ndarray, fg_mask: np.ndarray) -> float:
    """% of fg pixels in the green/blue-green oxidation (verdigris) HSV range."""
    if fg_mask.sum() == 0:
        return 0.0
    hsv = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
    return _colour_pct(hsv, fg_mask,
                       GREEN_OX_H_LO, GREEN_OX_H_HI, GREEN_OX_S_LO,
                       GREEN_OX_V_LO, GREEN_OX_V_HI)


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

    # Colour-based damage metrics on original image (resized to match mask)
    raw_resized = cv2.resize(raw, (fg_orig.shape[1], fg_orig.shape[0]))
    c_rust     = colour_rust_pct(raw_resized, fg_orig)
    c_tarnish  = colour_tarnish_pct(raw_resized, fg_orig)
    c_green_ox = colour_green_ox_pct(raw_resized, fg_orig)

    return {
        "path":             str(path),
        "score":            float(score),
        "is_anomalous":     bool(is_anom),
        "image":            result.image,   # preprocessed BGR, used for overlay panels
        "heatmap":          heatmap,        # numpy array, not serialised to JSON
        "fg_heat":          fg_heat,        # heatmap-resolution foreground mask
        "fg_image":         fg_orig,        # image-resolution foreground mask (for overlay)
        "colour_rust_pct":     c_rust,
        "colour_tarnish_pct":  c_tarnish,
        "colour_green_ox_pct": c_green_ox,
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

        t0_c_pct      = float(np.mean([x["colour_rust_pct"]     for x in t0_items]))
        t0_tarnish    = float(np.mean([x["colour_tarnish_pct"]  for x in t0_items]))
        t0_green_ox   = float(np.mean([x["colour_green_ox_pct"] for x in t0_items]))
        t0_sc         = float(np.mean([x["score"]               for x in t0_items]))
    else:
        t0_mean_hm = t0_std_hm = t0_fg = None
        t0_c_pct = t0_tarnish = t0_green_ox = t0_sc = 0.0

    result["t0_colour_rust_pct"]    = round(t0_c_pct,   2)
    result["t0_colour_tarnish_pct"] = round(t0_tarnish, 2)
    result["t0_colour_green_ox_pct"]= round(t0_green_ox,2)

    prev_c_pct  = t0_c_pct
    prev_hg_pct = 0.0  # T0 growth by definition starts at reference level ~7%

    for tp in TIMEPOINTS:
        items = [x for x in timepoints_data.get(tp, []) if x]
        if not items:
            result["timepoints"][tp] = None
            continue

        scores     = [x["score"]                for x in items]
        c_rusts    = [x["colour_rust_pct"]      for x in items]
        c_tarnishs = [x["colour_tarnish_pct"]   for x in items]
        c_green_oxs= [x["colour_green_ox_pct"]  for x in items]

        avg_sc    = float(np.mean(scores))
        c_pct     = float(np.mean(c_rusts))
        c_tan_pct = float(np.mean(c_tarnishs))
        c_gox_pct = float(np.mean(c_green_oxs))

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

        # Colour deltas vs T0 baseline
        c_delta_t0   = round(c_pct     - t0_c_pct,   2)
        c_delta_prev = round(c_pct     - prev_c_pct,  2)
        tan_delta_t0 = round(c_tan_pct - t0_tarnish,  2)
        gox_delta_t0 = round(c_gox_pct - t0_green_ox, 2)

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
            damage_type = "—"
        else:
            is_rusting = (c_delta_t0 > 2.0) or (hg_delta_t0 > 5.0)
            status = "DEGRADING" if is_rusting else "STABLE"
            damage_type = classify_damage_type(
                c_delta_t0, tan_delta_t0, gox_delta_t0, hg_delta_t0)

        tp_data = {
            "mean_score":               round(avg_sc, 3),
            "colour_rust_pct":          round(c_pct, 2),
            "colour_tarnish_pct":       round(c_tan_pct, 2),
            "colour_green_ox_pct":      round(c_gox_pct, 2),
            "colour_delta_vs_t0":       c_delta_t0,
            "colour_delta_vs_prev":     c_delta_prev,
            "tarnish_delta_vs_t0":      tan_delta_t0,
            "green_ox_delta_vs_t0":     gox_delta_t0,
            "heatmap_growth_pct":       round(hg_pct, 2),
            "heatmap_growth_delta_vs_t0": hg_delta_t0,
            "is_rusting":               is_rusting,
            "status":                   status,
            "damage_type":              damage_type,
            "shots":                    len(items),
        }

        result["timepoints"][tp] = tp_data
        prev_c_pct  = c_pct
        prev_hg_pct = hg_pct

    return result


# ─── output helpers ───────────────────────────────────────────────────────────

SEVERITY_THRESHOLDS = [(0, "None"), (2, "Trace"), (10, "Mild"),
                       (25, "Moderate"), (50, "Severe"), (100, "Critical")]


def classify_damage_type(c_rust_delta: float, c_tarnish_delta: float,
                         c_green_ox_delta: float, hm_delta: float) -> str:
    """
    Return a short label for the dominant surface damage type observed.

    Decision priority:
      - Colour signals (orange rust, tarnish, green oxidation) are checked first
        because they give direct material evidence.
      - Heatmap-growth is used when colour signals are weak — it catches wear,
        thread damage, pitting, cracking, blemishes, and loss of luster that
        don't shift the HSV distribution dramatically.
    """
    flags = []
    if c_rust_delta >= 1.0:
        flags.append("Oxidation/Rust")
    if c_tarnish_delta >= 0.5:
        flags.append("Tarnish")
    if c_green_ox_delta >= 0.3:
        flags.append("Green Ox.")
    if hm_delta >= 5.0:
        if not flags:
            # Heatmap fires but no colour signal → mechanical / surface damage
            if hm_delta >= 20:
                flags.append("Wear/Pitting")
            else:
                flags.append("Surface Dmg")
        else:
            # Colour + structural anomaly → combined
            flags.append("+Structural")
    return " / ".join(flags) if flags else "—"


def severity(colour_delta: float, hm_growth_delta: float = 0.0) -> str:
    """
    Rate overall degradation severity using both colour-change and heatmap-growth
    signals and return the worse of the two.

    Colour thresholds assume delta is % of fg pixels changing colour.
    HM thresholds are calibrated on the separate HM_SEVERITY_THRESHOLDS scale
    (empirically: 5 % growth ≈ trace, 15 % ≈ mild, 30 % ≈ moderate, etc.)
    """
    col_sev = "None"
    for thr, lbl in SEVERITY_THRESHOLDS:
        if colour_delta >= thr:
            col_sev = lbl

    hm_sev = "None"
    for thr, lbl in HM_SEVERITY_THRESHOLDS:
        if hm_growth_delta >= thr:
            hm_sev = lbl

    _order = ["None", "Trace", "Mild", "Moderate", "Severe", "Critical"]
    return _order[max(_order.index(col_sev), _order.index(hm_sev))]


def _fmt_delta(v: float) -> str:
    return f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%"


def print_report(components: List[dict]):
    w = 115
    print()
    print("=" * w)
    print("  SURFACE DEGRADATION PROGRESSION REPORT")
    print("  Primary:   Colour %  — damage-type colour pixels (rust/tarnish/green-ox) in fg")
    print("  Secondary: HM Growth %  — fg pixels exceeding T0 heatmap baseline by >1.5σ")
    print("  Damage Type: Oxidation/Rust | Tarnish | Green Ox. | Wear/Pitting | Surface Dmg")
    print("=" * w)

    for comp in components:
        name    = comp["component"]
        t0_c    = comp.get("t0_colour_rust_pct", 0.0)
        print(f"\n  Component : {name}   (T0 baseline rust colour: {t0_c:.1f}%)")
        print(f"  {'Time':<6} {'Score':>8} {'Rust%':>7} {'RustΔT0':>8} {'ColΔPrev':>9} "
              f"{'HMGrow%':>9} {'HMΔvsT0':>8} {'Status':<10} {'Severity':<10} {'Damage Type'}")
        print("  " + "-" * 103)

        for tp in TIMEPOINTS:
            tpd = comp["timepoints"].get(tp)
            if tpd is None:
                print(f"  T{tp:<5} {'—':>8}")
                continue

            st = tpd["status"]
            if st == "BASELINE":
                status_str = "\033[94mBASELINE \033[0m"
            elif tpd["is_rusting"]:
                status_str = "\033[91mDEGRADING\033[0m"
            else:
                status_str = "\033[92mSTABLE   \033[0m"

            sev = severity(tpd["colour_delta_vs_t0"],
                           tpd["heatmap_growth_delta_vs_t0"])
            dtype = tpd.get("damage_type", "—")
            print(f"  T{tp:<5} {tpd['mean_score']:>8.3f} "
                  f"{tpd['colour_rust_pct']:>6.1f}% "
                  f"{_fmt_delta(tpd['colour_delta_vs_t0']):>8} "
                  f"{_fmt_delta(tpd['colour_delta_vs_prev']):>9} "
                  f"{tpd['heatmap_growth_pct']:>8.1f}% "
                  f"{_fmt_delta(tpd['heatmap_growth_delta_vs_t0']):>8}  "
                  f"{status_str}  {sev:<10}  {dtype}")

    print()
    print("=" * w)


def write_summary_txt(components: List[dict], out_path: Path):
    lines = [
        "SURFACE DEGRADATION PROGRESSION REPORT",
        "Primary:   Colour %  — damage-type colour pixels (rust/tarnish/green-ox) in fg",
        "Secondary: HM Growth %  — fg pixels exceeding T0 heatmap baseline by >1.5sigma",
        "Damage Type: Oxidation/Rust | Tarnish | Green Ox. | Wear/Pitting | Surface Dmg",
        "=" * 115, "",
    ]
    for comp in components:
        name = comp["component"]
        t0_c = comp.get("t0_colour_rust_pct", 0.0)
        lines.append(f"Component: {name}   (T0 baseline rust colour: {t0_c:.1f}%)")
        lines.append(f"  {'Time':<6} {'Score':>8} {'Rust%':>7} {'RustDT0':>8} {'ColDPrev':>9} "
                     f"{'HMGrow%':>9} {'HMDvsT0':>8}  {'Status':<10} {'Severity':<10}  {'Damage Type'}")
        lines.append("  " + "-" * 103)
        for tp in TIMEPOINTS:
            tpd = comp["timepoints"].get(tp)
            if tpd is None:
                lines.append(f"  T{tp:<5} —")
                continue
            sev   = severity(tpd["colour_delta_vs_t0"],
                             tpd["heatmap_growth_delta_vs_t0"])
            dtype = tpd.get("damage_type", "—")
            lines.append(f"  T{tp:<5} {tpd['mean_score']:>8.3f} "
                         f"{tpd['colour_rust_pct']:>6.1f}% "
                         f"{_fmt_delta(tpd['colour_delta_vs_t0']):>8} "
                         f"{_fmt_delta(tpd['colour_delta_vs_prev']):>9} "
                         f"{tpd['heatmap_growth_pct']:>8.1f}% "
                         f"{_fmt_delta(tpd['heatmap_growth_delta_vs_t0']):>8}  "
                         f"{tpd['status']:<10} {sev:<10}  {dtype}")
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


def plot_score_timeline(components: List[dict],
                        threshold: float,
                        out_path: Path):
    """
    Line chart: per-component anomaly score across T0 → T24 → T48 → T72.

    Each component gets its own coloured line; markers distinguish timepoints.
    A horizontal dashed line marks the model threshold so it's immediately clear
    where the threshold sits relative to the actual scores.  A secondary panel
    shows the T0-relative heatmap-growth % so both signals are visible together.
    """
    fig, (ax_sc, ax_hg) = plt.subplots(1, 2, figsize=(14, 5))

    for i, comp in enumerate(components):
        tps, scores, hg_pcts = [], [], []
        for tp in TIMEPOINTS:
            tpd = comp["timepoints"].get(tp)
            if tpd:
                tps.append(tp)
                scores.append(tpd["mean_score"])
                hg_pcts.append(tpd["heatmap_growth_pct"])
        col = COLORS[i % len(COLORS)]
        ax_sc.plot(tps, scores, marker="o", color=col,
                   label=comp["component"], linewidth=2)
        ax_hg.plot(tps, hg_pcts, marker="s", color=col,
                   label=comp["component"], linewidth=2, linestyle="--")

    # Threshold reference
    if threshold > 0:
        ax_sc.axhline(threshold, color="black", ls="--", lw=1.5,
                      label=f"Threshold ({threshold:.2f})")

    for ax, title, ylabel in [
        (ax_sc,
         "Anomaly Score per Component (Current Model)",
         "PatchCore Anomaly Score"),
        (ax_hg,
         "Heatmap Growth vs T0 Baseline\n(pixels >1.5σ above T0 mean)",
         "Anomalous Growth Area (%)"),
    ]:
        ax.set_xlabel("Exposure (hours)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(TIMEPOINTS)
        ax.set_xticklabels([f"T{t}h" for t in TIMEPOINTS])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Component Score & Degradation Timeline",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
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


def plot_overlay_panel(name: str,
                       tp_items: Dict[int, List[dict]],
                       out_path: Path):
    """
    1×4 figure: anomaly heatmap blended onto the preprocessed image for each
    timepoint (T0 → T24 → T48 → T72).  Multiple shots per timepoint are averaged
    for the heatmap; the first shot's image is used as the background.

    Saved to overlays/<Component>_overlays.png
    """
    n_tp = len(TIMEPOINTS)
    fig, axes = plt.subplots(1, n_tp, figsize=(4.5 * n_tp, 4.5))
    fig.suptitle(f"{name} — Heatmap Overlay Progression",
                 fontsize=13, fontweight="bold")

    for col, tp in enumerate(TIMEPOINTS):
        items = [x for x in tp_items.get(tp, []) if x]
        ax = axes[col]

        if tp == 0:
            stage_label = "T0h  BASELINE"
            title_color = "#2196F3"
        else:
            stage_label = f"T{tp}h"
            title_color = "black"

        ax.set_title(stage_label, fontsize=11, color=title_color, fontweight="bold")
        ax.axis("off")

        if not items:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            continue

        # Average heatmap across shots; use first shot's preprocessed image.
        # Average the fg masks too so we blend only inside the component.
        avg_hm = np.mean([x["heatmap"] for x in items], axis=0)
        img_bgr = items[0]["image"]
        avg_fg = (np.mean([x["fg_image"] for x in items], axis=0) > 0
                  ).astype(np.uint8) * 255
        overlay_bgr = heatmap_overlay(img_bgr, avg_hm, alpha=0.45, fg_mask=avg_fg)
        ax.imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))

        mean_score = float(np.mean([x["score"] for x in items]))
        c_rust = float(np.mean([x["colour_rust_pct"] for x in items]))
        ax.set_xlabel(f"score={mean_score:.2f}  rust={c_rust:.1f}%",
                      fontsize=8, labelpad=4)

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
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

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
                if tp == 0:
                    tag = "\033[94mBASELINE\033[0m"
                elif r and r["is_anomalous"]:
                    tag = "\033[91mANOMALY \033[0m"
                else:
                    tag = "\033[92mNORMAL  \033[0m"
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

        # 1. Score timeline — per-component score + heatmap growth over time.
        #    Also regenerates outputs/visualizations/score_dist.png from actual
        #    component scores so it reflects the current model (not the stale
        #    training-time distribution from stock images).
        plot_score_timeline(component_reports, model.threshold,
                            plots_dir / "score_timeline.png")
        print(f"    score_timeline.png")

        # Collect T0 (baseline) and T24/T48/T72 (degraded) scores for the
        # canonical score_dist.png used by the broader pipeline.
        t0_scores, degraded_scores = [], []
        for comp_name, tp_dict in raw_results.items():
            for tp, items in tp_dict.items():
                for item in (items or []):
                    if item is None:
                        continue
                    if tp == 0:
                        t0_scores.append(item["score"])
                    else:
                        degraded_scores.append(item["score"])

        if t0_scores:
            viz_dir = cfg.paths.output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            plot_score_distribution(
                np.array(t0_scores),
                np.array(degraded_scores) if degraded_scores else None,
                model.threshold,
                str(viz_dir / "score_dist.png"),
                title="Anomaly Score Distribution — Component Images (Current Model)",
                normal_label=f"T0 Baseline  (n={len(t0_scores)})",
                anomaly_label=f"Degraded T24/T48/T72  (n={len(degraded_scores)})",
            )
            print(f"    score_dist.png  (→ {viz_dir})")

        # 2. Progression line chart
        plot_progression(component_reports,
                         plots_dir / "progression_all.png")
        print(f"    progression_all.png")

        # 3. Delta bar chart
        plot_delta_bars(component_reports,
                        plots_dir / "delta_bars.png")
        print(f"    delta_bars.png")

        # 4. Per-component heatmap grids
        for comp_name in sorted(raw_results):
            safe = re.sub(r"[^\w\-]", "_", comp_name)
            plot_component_heatmaps(
                comp_name,
                raw_results[comp_name],
                plots_dir / f"{safe}_heatmaps.png",
            )
            print(f"    {safe}_heatmaps.png")

        print(f"\n  Plots → {plots_dir}")

        # 5. Per-component heatmap-on-image overlay panels (T0→T24→T48→T72)
        print("\n  Generating overlay panels...")
        for comp_name in sorted(raw_results):
            safe = re.sub(r"[^\w\-]", "_", comp_name)
            plot_overlay_panel(
                comp_name,
                raw_results[comp_name],
                overlays_dir / f"{safe}_overlays.png",
            )
            print(f"    {safe}_overlays.png")

        print(f"\n  Overlays → {overlays_dir}")

    print(f"\n  DONE\n")


if __name__ == "__main__":
    main()
