# ARCHVIZ RelateAnything QC v0.7

BEFORE vs AFTER geometry QC for architectural visualization.

This package does not replace `ComfyUI-RelateAnything-v0.6`; it adds `RA · BEFORE vs AFTER QC · v0.7`.

## Checks
- matched / missing / added regions
- normalized center drift
- bbox size drift
- left/right and above/below order flips
- same-row / same-column changes
- pairwise spacing drift
- optional RelateAnything semantic delta (advisory only)

Geometry PASS/FAIL is deterministic bbox math. RelateAnything confidence never decides geometry truth.

## Install
Copy `ComfyUI-RelateAnything-QC-v07` into:
`Q:\AI_ArchViz\ComfyUI_windows_portable\ComfyUI\custom_nodes\`

Keep your existing `ComfyUI-RelateAnything-ONNX-v06`. Restart ComfyUI.

Import `workflows/ARCHVIZ_RELATEANYTHING_BEFORE_AFTER_QC_v007.json`.

## Baseline
- same camera / same composition
- SAM3 precision: auto
- prompt: window
- confidence: 0.35
- max detections: 16
- RA predicates: left/right/above/below/next to
- RA score threshold: 0.10

## QC defaults
- max center match distance: 0.08
- min IoU: 0.05
- center drift WARN: 1.0% image diagonal
- size drift WARN: 8%
- row tolerance: 0.02
- column tolerance: 0.02
- order tolerance: 0.01

Status:
- FAIL: missing/added regions or order flips
- WARN: center/size/grid drift above thresholds
- PASS: no hard geometry changes and drift inside thresholds

LAB / TEST: review matches in qc_json on dense repetitive facades.
