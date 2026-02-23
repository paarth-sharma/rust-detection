# How to Interpret Corrosion Analysis Results

---

## What is HM Growth %?

**In plain English:** The model looks at every small patch of the component's surface and assigns it an "unusualness score" — how different does this patch look compared to what a healthy fastener looks like? At T0 (healthy baseline), we record those scores pixel-by-pixel. At T24/T48/T72, we ask: *what fraction of the surface now scores significantly higher than it did at T0?* That fraction is HM Growth %.

**A number for an outsider:** "25% HM Growth at T48" means that roughly 1 in 4 pixels of the component's surface has become noticeably more unusual-looking compared to when it was new. It does not say those pixels are orange — it says the AI sees something different there, which can include texture change, surface pitting, or oxidation that hasn't discoloured yet.

**The key formula** is in `corrosion_analysis.py:110`:
```python
pixel_thresh = t0_mean_hm + sigma * t0_std_hm   # "how much above T0 is significant"
grown = (heatmap > pixel_thresh) & fg            # pixels that crossed that line
```
It asks: which pixels have scores more than 1.5 standard deviations above the T0 average? That threshold (`sigma=1.5`) is why T0 always reads 0% — by definition, T0 is the reference.

---

## What handles rust detection through black coating?

The colour method (`colour_rust_pct`, lines 78–93) is **completely blind** to it. It only looks for orange/brown HSV pixels:
```python
RUST_H_LO, RUST_H_HI = 5, 25      # orange-red hue
RUST_S_LO             = 45         # saturated, not grey
RUST_V_LO, RUST_V_HI = 40, 210    # not too dark, not washed out
```
A black-coated component rusting underneath will not show these colours — the coating masks the discolouration — so the colour % stays near 0% for Washer Black and Nut Black.

**The HM Growth column handles this.** The PatchCore model (`models/patchcore.py`) was trained on the texture and appearance of healthy fasteners using a deep neural network (WideResNet50). When a surface oxidises under a coating, the texture changes — tiny bumps, pitting, surface swelling — even if the colour does not. The model picks up on those textural anomalies and raises the patch score. That elevated score is what HM Growth measures.

In short:
- **Colour % = "can I see orange rust?"** → works on silver/grey, fails on black
- **HM Growth % = "does this surface look different from T0?"** → works on both, catches subsurface/early-stage change

---

## How to read the heatmap plots (`*_heatmaps.png`)

Each plot is a 2-row × 4-column grid:

```
         T0h          T24h         T48h         T72h
Row 1: [Photo]      [Photo]      [Photo]      [Photo]     ← original image
Row 2: [Heatmap]    [Heatmap]    [Heatmap]    [Heatmap]   ← anomaly score map
```

**The heatmap colour scale (hot = yellow/white → black):**
- **Bright yellow/white patches** = the model finds this area most different from healthy
- **Dark/black areas** = looks similar to healthy
- **The colour bar on the right** shows the numeric score range

What to look for: if a region that was dark at T0 becomes bright at T48/T72, that area has degraded. On the Bolt Allen Black, you would see spreading bright patches from T24 onwards. On Washer Black, the entire surface brightens even though the photo looks the same colour — the texture is changing.

The `score=` and `colour_rust=%` labels under each heatmap give the averaged values for that timepoint.

---

## How the heatmap changes as a component ages

The heatmap is a map of surface "unusualness" compared to a healthy fastener. Every degradation mechanism produces a distinct visual signature in it, described below from earliest to latest stage.

### Stage 1 — Fresh / healthy (T0)
The heatmap is uniformly dark across the entire foreground. The component surface has consistent texture and colour, so every patch closely matches what the model has learned a healthy fastener looks like. The score colour bar sits at its lowest range. There are no bright spots. This is your reference; all later heatmaps are judged against it.

```
T0 heatmap:   ░░░░░░░░░░   (all dark — nothing unusual)
```

### Stage 2 — Early oxidation / surface film (typically T24)
A thin oxide layer starts forming on exposed metal. To the human eye the component may look nearly identical to T0, but the model begins to see it. A few scattered bright specks appear, usually at corners, thread roots, and edges first — these are stress-concentration points where the protective coating or oxide layer is thinnest and environmental attack starts earliest. The HM Growth % rises while the Colour % may still be near zero because the surface has not yet discoloured to orange.

```
T24 heatmap:  ░░░▒░▒░░░░   (faint scattered bright spots at edges/threads)
```

### Stage 3 — Active rust / discolouration (typically T48)
Iron oxide (rust) is now forming visibly. On silver and grey components this appears as orange-brown patches — the Colour % rises sharply at this point. On the heatmap, those same patches become strongly bright because the texture has changed: rust is granular and rough, completely unlike the smooth machined surface in training. The bright areas grow in size and intensity and are now spatially coherent — they cluster where you would physically expect rust to start (flat faces, thread valleys, any scratched or damaged area).

```
T48 heatmap:  ░░▓▓░░▓░░░   (bright patches, often on faces and thread roots)
```

### Stage 4 — Pitting and surface loss (typically T72 and beyond)
Extended exposure causes the metal beneath the oxide to be eaten away, forming pits — small craters visible under magnification. On the heatmap, pitted regions produce the highest scores because:
- the surface texture is radically different from healthy (deep, irregular cavities vs. smooth metal)
- the local colour has changed (dark rust-brown inside pits, orange flaking at edges)
- the geometry of the light reflection changes (pits create shadows the model has never seen on a healthy component)

The heatmap at this stage shows **large, solid bright regions** rather than scattered specks. The bright area covers a significant fraction of the foreground. If pitting is severe the entire face of the component can be bright white in the heatmap even if the overall photo still looks mostly intact.

```
T72 heatmap:  ░▓▓▓▓▓▓░░░   (large solid bright regions — structural degradation)
```

### What each degradation type looks like specifically

| Degradation | What happens physically | Heatmap signature |
|---|---|---|
| **Surface oxidation** | Thin, uniform oxide film, no colour change yet | Slightly elevated score across the whole face, no specific bright spots |
| **Discolouration / rust staining** | Iron oxide turns orange-brown | Bright patches that spatially match the orange areas visible in the photo above |
| **Rust under black coating** | Coating intact but oxidation swells the surface microscopically | Diffuse brightening across the face with no corresponding colour change — the coating hides the rust from the eye but not from the texture model |
| **Flaking / delamination** | Rust or coating lifts off in flakes | Sharp, high-contrast bright edges where the flake boundary is; the flake itself may be very bright if the exposed metal underneath is fresh |
| **Pitting** | Small craters eat into the metal | Intensely bright small spots, often with a slightly less-bright halo around each pit |
| **Thread corrosion** | Thread valleys fill with rust, profile changes | Bright bands running along the thread pitch lines, more visible when viewing from the side |
| **General surface roughening** | Smooth machined surface becomes granular | Broad, moderate brightening across the whole foreground rather than localised spots |

### Reading the four-column progression side by side

When you look at the T0 → T24 → T48 → T72 heatmaps left to right, you are watching the corrosion front advance in real time. Ask these three questions:

1. **Where did the first bright spot appear?** That is the weakest point of the component — the site most susceptible to corrosion. It tells you something about the manufacturing quality or coating uniformity at that location.

2. **Did the bright area grow outward from that spot, or did new spots appear elsewhere?** Growing outward from one point suggests a single initiation site (e.g. a scratch). New independent spots suggest generalised attack across the surface.

3. **How much did the score colour bar's maximum value increase?** If the bar's upper end is much higher at T72 than T0, the worst-affected patches have degraded severely. If the bar's range stayed similar but more pixels are bright, the degradation is widespread but still shallow.

---

## How to read `summary.txt`

Each column explained with an example row:

```
T48   13.363   13.7%   +13.5%   +10.3%   24.4%   +24.4%   RUSTING   Mild
```

| Column | Value | Meaning |
|---|---|---|
| `Time` | T48 | 48-hour exposure image |
| `Score` | 13.363 | Raw PatchCore anomaly score (higher = more unusual). Not directly comparable across components |
| `Colour%` | 13.7% | 13.7% of the component's surface area is orange/brown coloured |
| `ColΔT0` | +13.5% | 13.5 percentage points more rust-coloured than the T0 baseline |
| `ColΔPrev` | +10.3% | 10.3 pp increase since the previous interval (T24) — the rate of new discolouration |
| `HMGrow%` | 24.4% | 24.4% of the surface exceeds the T0 heatmap baseline by >1.5σ |
| `HMΔvsT0` | +24.4% | All of that growth is new relative to T0 |
| `Status` | RUSTING | Triggered because ColΔT0 > 2% or HMΔvsT0 > 5% |
| `Severity` | Mild | Based on ColΔT0: None <2%, Trace 2–10%, Mild 10–25%, Moderate 25–50%, Severe 50%+ |

**T0 row always reads 0.0% for all delta columns** — it is the reference everything else is measured against, not a claim that the component has zero imperfections.
