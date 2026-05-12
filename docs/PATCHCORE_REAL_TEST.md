# Real PatchCore Checkpoint Test

## Purpose

This case demonstrates the source-first ONNX export strategy for an Anomalib
PatchCore anomaly detection checkpoint. The toolkit does not replace Anomalib
export. Anomalib produced the ONNX artifact, then this toolkit validated it and
assessed the route.

## Source Model

- checkpoint path: `<PATH_TO_PATCHCORE_CHECKPOINT>`
- task: anomaly detection
- source framework: Anomalib
- model family: PatchCore

## Checkpoint Inspection

A real PatchCore checkpoint was inspected successfully. PatchCore checkpoints
may contain feature extractor weights, a memory bank, post-processing
thresholds, and training metadata.

Trusted local loading may be required for PyTorch Lightning or framework-owned
checkpoints. Use `--unsafe-load` only for files from trusted sources.

```bash
python -m converter.cli inspect "<PATH_TO_PATCHCORE_CHECKPOINT>" --max-items 120 --unsafe-load
```

## Official Anomalib ONNX Export

The ONNX artifact was produced outside this toolkit with Anomalib's official
export path. The exact command depends on the Anomalib version and project
configuration; conceptually it uses the trained PatchCore checkpoint as input
and writes an ONNX model.

Example shape of the workflow:

```bash
anomalib export --model Patchcore --export_type onnx --ckpt_path "<PATH_TO_PATCHCORE_CHECKPOINT>" --input_size "[256,256]"
```

Observed result:

- Anomalib official export produced ONNX successfully.
- The toolkit validated the exported ONNX successfully.
- The generic toolkit exporter correctly refused direct export for this spec.

## ONNX Validation With This Toolkit

```bash
python -m converter.cli validate-onnx "<PATH_TO_PATCHCORE_ONNX>" --input-shape 1,3,256,256 --input-name input --input-dtype float32
```

Observed ONNX outputs included:

- `pred_score`
- `pred_label`
- `anomaly_map`
- `pred_mask`

## Export Capability Assessment

```bash
python -m converter.cli assess-export specs/patchcore_cable_coreset_0_1.yaml
```

Expected reasoning:

- detected framework: Anomalib
- detected model family: PatchCore
- official ONNX exporter route should be preferred first
- toolkit generic exporter is blocked in the current spec because
  `model.module` and `model.class_name` are not provided
- recommended route: official source exporter

This matches the real experiment: Anomalib exported the model, and this toolkit
validated the ONNX artifact.

## Repository Spec

The repository spec uses placeholders for local machine paths:

```bash
python -m converter.cli validate-spec specs/patchcore_cable_coreset_0_1.yaml
python -m converter.cli plan-load specs/patchcore_cable_coreset_0_1.yaml
```

The spec is intended for validation and route assessment. Direct toolkit export
is expected to refuse until a framework-specific loader/wrapper or explicit
module/class construction is provided.
