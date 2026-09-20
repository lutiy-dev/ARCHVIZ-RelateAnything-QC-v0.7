# ARCHVIZ RelateAnything QC v0.7

**Deterministic BEFORE vs AFTER geometry-quality control for architectural visualization workflows in ComfyUI.**

ARCHVIZ RelateAnything QC v0.7 compares a geometry-truth render against an AI-processed result and reports structural drift in repeated facade elements such as windows.

The workflow combines:

- **SAM3** for object detection / segmentation;
- **RelateAnything ONNX** for advisory spatial relations;
- a **deterministic bbox-based QC layer** for geometry decisions.

The goal is simple: preserve camera, facade structure, repetition, alignment, spacing, and object count while still allowing AI-assisted finishing.

## What it checks

- matched regions;
- missing regions;
- newly added regions;
- normalized center drift;
- bounding-box size drift;
- left/right ordering flips;
- above/below ordering flips;
- same-row / same-column changes;
- pairwise spacing drift;
- optional RelateAnything semantic relation delta.

## Geometry truth

**PASS / WARN / FAIL is determined from normalized bounding-box geometry, not from RelateAnything confidence scores.**

RelateAnything is used as a secondary semantic layer for relations such as:

- `to the left of`
- `to the right of`
- `above`
- `below`
- `next to`

This keeps geometry QC deterministic and repeatable.

## Pipeline

```text
BEFORE / geometry truth
→ SAM3
→ RA Regions
→ RelateAnything ONNX
        \
         → BEFORE vs AFTER QC
        /
AFTER / AI result
→ SAM3
→ RA Regions
→ RelateAnything ONNX
```

## Installation

This package extends the existing RelateAnything ONNX v0.6 setup.

Copy:

```text
ComfyUI-RelateAnything-QC-v07
```

into:

```text
Q:\AI_ArchViz\ComfyUI_windows_portable\ComfyUI\custom_nodes\
```

Keep the existing:

```text
ComfyUI-RelateAnything-ONNX-v06
```

Restart ComfyUI and import:

```text
workflows/ARCHVIZ_RELATEANYTHING_BEFORE_AFTER_QC_v007.json
```

## Recommended baseline

For the current facade-window test:

- same camera / same composition;
- SAM3 precision: `auto`;
- text prompt: `window`;
- confidence threshold: `0.35`;
- max detections: `16`;
- RelateAnything vocabulary: left / right / above / below / next to;
- RelateAnything score threshold: `0.10`.

## QC defaults

- max center match distance: `0.08`;
- minimum IoU: `0.05`;
- center drift warning: `1.0%` of image diagonal;
- size drift warning: `8%`;
- row tolerance: `0.02`;
- column tolerance: `0.02`;
- order tolerance: `0.01`.

## Result states

- **FAIL** — missing / added regions or spatial order flips;
- **WARN** — matched geometry exists, but center / size / grid drift exceeds tolerance;
- **PASS** — no hard geometry changes and drift remains inside configured tolerances.

## Status

**LAB / TEST**

Dense repetitive facades can produce ambiguous matches. Review `qc_json` when validating production-critical results.

---

Built for AI-assisted architectural visualization workflows where **3D / render geometry remains the source of truth**.
