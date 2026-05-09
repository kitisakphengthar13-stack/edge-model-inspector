# Real PatchCore Checkpoint Test

This document is for testing a real Anomalib PatchCore checkpoint on the PC
side.

The current toolkit can inspect checkpoints, validate specs, check checkpoint
contents, and plan loading. The current generic exporter is expected to refuse
direct export for this spec because it does not provide `model.module` and
`model.class_name`. That refusal is correct behavior.

PatchCore checkpoints may contain a `memory_bank`, feature extractor weights,
post-processing thresholds, and training metadata. Direct full-model ONNX or
TensorRT export may not be practical without a framework-specific loader or
wrapper.

Likely deployment paths:

1. Use Anomalib's own export path to produce ONNX, then use this toolkit for
   `validate-onnx` and `plan-tensorrt`.
2. Implement a future Anomalib/PatchCore loader plugin in this toolkit.
3. Export only the neural feature extractor and keep PatchCore scoring and
   post-processing outside TensorRT.

## Manual PC Commands

```bash
python -m converter.cli validate-spec specs/patchcore_cable_coreset_0_1.yaml

python -m converter.cli check-checkpoint specs/patchcore_cable_coreset_0_1.yaml --max-items 120

python -m converter.cli plan-load specs/patchcore_cable_coreset_0_1.yaml

python -m converter.cli export-onnx specs/patchcore_cable_coreset_0_1.yaml --allow-imports
```

The export command should refuse or fail clearly because `model.module` and
`model.class_name` are missing, or because a framework-specific loader is not
implemented. This is expected and should not be treated as a project failure.

## If Anomalib Exports ONNX Separately

```bash
python -m converter.cli validate-onnx path/to/patchcore.onnx --spec specs/patchcore_cable_coreset_0_1.yaml

python -m converter.cli plan-tensorrt path/to/patchcore.onnx --spec specs/patchcore_cable_coreset_0_1.yaml --target orin_nano --precision fp16
```
