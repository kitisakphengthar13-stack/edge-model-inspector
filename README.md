# PyTorch Model Inspection and ONNX Preparation Toolkit

A local CLI toolkit for inspecting PyTorch checkpoints, assessing the right ONNX
export route, using official source-library exporters when available, falling
back to guarded generic PyTorch-to-ONNX export when appropriate, and validating
ONNX artifacts before downstream deployment.

This project does **not** claim to automatically convert every `.pt`, `.pth`, or
`.ckpt` file into ONNX. PyTorch checkpoint files can store very different
things, and some models are best exported through their original framework.

## What This Solves

PyTorch model files are often ambiguous: a file may be a raw `state_dict`, a
training checkpoint, a full model object, a PyTorch Lightning checkpoint, or a
framework-specific artifact. This toolkit makes that uncertainty explicit and
turns it into a repeatable ONNX preparation workflow:

- inspect checkpoint structure and deployment signals
- validate a model spec before running trusted code
- assess whether an official source-library ONNX exporter should be used
- attempt generic PyTorch-to-ONNX export only when the spec is sufficient
- validate ONNX files with ONNX checker and ONNX Runtime inference

## What This Project Is / Is Not

This project is:

- a PyTorch checkpoint inspection toolkit
- a spec-driven ONNX preparation workflow
- an export-route assessment tool
- a guarded generic PyTorch-to-ONNX fallback exporter
- an ONNX validation tool

This project is not:

- a universal `.pt` / `.pth` / `.ckpt` converter
- a replacement for Anomalib, Ultralytics, or other source frameworks
- an automatic exporter for every model architecture
- a TensorRT engine builder
- a collection of runtime-specific deployment backends

## Core Workflow

1. Inspect a PyTorch checkpoint.
2. Describe model construction, checkpoint loading, input, output, and export
   intent in a spec.
3. Assess the recommended ONNX export route.
4. Use the official source-library ONNX exporter first when available.
5. Use this toolkit's generic PyTorch-to-ONNX exporter only when the spec has
   enough information and a dry-run forward pass succeeds.
6. Validate the resulting ONNX artifact with ONNX checker and ONNX Runtime.

## Export Strategy

The project uses a source-first ONNX export strategy.

Prefer the official exporter from the source framework/library:

- Anomalib models should use Anomalib export when available.
- Ultralytics YOLO models should use Ultralytics export when available.
- Other framework-specific models should use their official ONNX path when one
  exists.

Use this toolkit's generic PyTorch-to-ONNX exporter only as a fallback when:

- `model.module` and `model.class_name` are available
- checkpoint loading is defined
- input and output specs are provided
- a guarded PyTorch dry-run forward pass succeeds

Official source-library exporters and the toolkit generic exporter are separate
routes. The goal is not to force every model through one converter; it is to
produce a reliable, validated ONNX artifact through the most appropriate path.

## Export Capability Assessment

`assess-export` is a non-executing analysis command. It does not import model
code, instantiate models, load checkpoints from specs, run external exporters,
or create artifacts.

It reports:

- detected or declared source framework and model family
- whether an official source-library ONNX exporter route is known or likely
- whether this toolkit's generic PyTorch-to-ONNX exporter can be attempted
- recommended route, evidence, blockers, and unknowns

Examples:

```bash
python -m converter.cli assess-export specs/patchcore_cable_coreset_0_1.yaml
python -m converter.cli assess-export specs/yolo26n_task_detect.yaml
python -m converter.cli assess-export specs/example_simple_classifier_dryrun.yaml
```

Checkpoint-path assessment is preliminary and lower confidence because a
checkpoint alone usually does not contain the full export contract:

```bash
python -m converter.cli assess-export path/to/model.pt --unsafe-load
```

## Main CLI Examples

```bash
python -m converter.cli inspect models/model.pt
python -m converter.cli inspect models/model.ckpt --max-items 120
python -m converter.cli inspect models/model.pt --unsafe-load

python -m converter.cli validate-spec specs/example_simple_classifier_dryrun.yaml
python -m converter.cli assess-export specs/example_simple_classifier_dryrun.yaml
python -m converter.cli plan-load specs/patchcore_cable_coreset_0_1.yaml

python -m converter.cli dry-run-model specs/example_simple_classifier_dryrun.yaml --allow-imports
python -m converter.cli export-onnx specs/example_simple_classifier_dryrun.yaml --allow-imports
python -m converter.cli validate-onnx artifacts/simple_classifier_dryrun/model.onnx --spec specs/example_simple_classifier_dryrun.yaml
```

`dry-run-model` and `export-onnx` may import local user code, so both are
guarded with explicit flags. Use them only for trusted local modules.

## ONNX Validation

`validate-onnx` checks exported ONNX files before downstream deployment. It:

- loads the ONNX file
- runs `onnx.checker.check_model`
- creates an ONNX Runtime CPU session
- builds dummy input from the spec or CLI arguments
- runs inference
- prints readable input/output summaries

It does not import user modules, load checkpoints, execute source-framework
exporters, or build deployment engines.

## Real Case Studies

### PatchCore / Anomalib

See [docs/PATCHCORE_REAL_TEST.md](docs/PATCHCORE_REAL_TEST.md) and
`specs/patchcore_cable_coreset_0_1.yaml`.

A real Anomalib PatchCore checkpoint was inspected, exported to ONNX with the
official Anomalib exporter, and validated with this toolkit. `assess-export`
recommends the official Anomalib ONNX route first, while the toolkit generic
exporter is intentionally blocked for the current spec because module/class
construction is not provided.

### YOLO / Ultralytics

See [docs/YOLO_REAL_TEST.md](docs/YOLO_REAL_TEST.md) and
`specs/yolo26n_task_detect.yaml`.

A real Ultralytics YOLO26n detection checkpoint was inspected, exported to ONNX
with the official Ultralytics exporter, and validated with this toolkit.
`assess-export` recommends the official Ultralytics ONNX route first, while the
toolkit generic exporter is not the appropriate path for that spec.

## Optional Downstream TensorRT Planning

`plan-tensorrt` is a small downstream helper that generates a `trtexec` planning
command from an existing ONNX artifact. It does not run `trtexec`, does not
require TensorRT, and does not build engine files.

```bash
python -m converter.cli plan-tensorrt artifacts/simple_classifier_dryrun/model.onnx --spec specs/example_simple_classifier_dryrun.yaml --target orin_nano --precision fp16
```

TensorRT engine creation is outside the core ONNX workflow and should happen on
the target device or a matching runtime environment.

## Project Boundaries

- ONNX is the core artifact format for this project.
- Source-framework exporters are preferred when they are the authoritative route
  for a model family.
- Generic export is intentionally guarded and spec-driven.
- Real exported ONNX artifacts may be kept local-only when they are large or
  machine-specific.
