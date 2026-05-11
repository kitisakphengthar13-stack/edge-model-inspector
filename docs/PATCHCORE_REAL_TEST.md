# Real PatchCore Checkpoint Test

This document records the PC-side workflow for a real Anomalib PatchCore
checkpoint.

Observed result:

- The PatchCore checkpoint was inspected successfully.
- Anomalib official export produced ONNX successfully.
- This toolkit validated the exported ONNX successfully.
- This toolkit planned a TensorRT/`trtexec` command successfully.

This confirms the source-first export strategy: Anomalib exported the model,
and this toolkit validated and prepared the ONNX artifact for downstream
deployment planning.

For PatchCore, using Anomalib's official export path is preferred over forcing
generic PyTorch export from this toolkit.

PatchCore checkpoints may contain a `memory_bank`, feature extractor weights,
post-processing thresholds, and training metadata. Direct full-model export may
not be practical without a framework-specific loader or wrapper.

The generic exporter correctly refuses direct export for this spec because
`model.module` and `model.class_name` are missing, and no framework-specific
loader is implemented. This refusal is intentional and not a project failure.

## Manual PC Commands

```bash
python -m converter.cli validate-spec specs/patchcore_cable_coreset_0_1.yaml

python -m converter.cli check-checkpoint specs/patchcore_cable_coreset_0_1.yaml --max-items 120

python -m converter.cli plan-load specs/patchcore_cable_coreset_0_1.yaml

python -m converter.cli export-onnx specs/patchcore_cable_coreset_0_1.yaml --allow-imports
```

The export command should refuse or fail clearly because `model.module` and
`model.class_name` are missing, or because a framework-specific loader is not
implemented. This is expected.

## If Anomalib Exports ONNX Separately

```bash
python -m converter.cli validate-onnx path/to/patchcore.onnx --spec specs/patchcore_cable_coreset_0_1.yaml

python -m converter.cli plan-tensorrt path/to/patchcore.onnx --spec specs/patchcore_cable_coreset_0_1.yaml --target orin_nano --precision fp16
```
