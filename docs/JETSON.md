# Jetson TensorRT Notes

Jetson/TensorRT is a downstream deployment step. The toolkit's core output is a
validated ONNX artifact. TensorRT engine building is manual today, or future
optional automation on the target device.

This toolkit does not replace `trtexec`. `trtexec` is the authoritative
TensorRT build tool.

Do not treat TensorRT engine files as portable artifacts. Build them on the
actual target device or a matching TensorRT, CUDA, and JetPack runtime.

## PC Responsibilities

- Inspect checkpoint files.
- Validate `spec.yaml`.
- Dry-run the PyTorch model when trusted imports are allowed.
- Export ONNX.
- Validate ONNX with ONNX Runtime CPU inference.
- Optionally generate a TensorRT/`trtexec` build plan.

## Jetson Responsibilities

- Check TensorRT and `trtexec` availability.
- Build the TensorRT engine from ONNX.
- Benchmark the engine.
- Save build logs, benchmark logs, and deployment metadata.

## PC-Side Commands

```bash
python -m converter.cli export-onnx specs/example_simple_classifier_dryrun.yaml --allow-imports
python -m converter.cli validate-onnx artifacts/simple_classifier_dryrun/model.onnx --spec specs/example_simple_classifier_dryrun.yaml
python -m converter.cli plan-tensorrt artifacts/simple_classifier_dryrun/model.onnx --spec specs/example_simple_classifier_dryrun.yaml --target orin_nano --precision fp16
```

## Files To Copy To Jetson

- `converter/`
- `configs/`
- `specs/`
- `model_zoo/`
- `artifacts/simple_classifier_dryrun/model.onnx`
- `requirements.txt`
- `README.md`
- `docs/JETSON.md`

Large checkpoint files such as `.pt`, `.pth`, and `.ckpt` are not required for
the first TensorRT demo if the ONNX file already exists.

## Manual TensorRT Examples

These commands are examples. Actual TensorRT build behavior should be verified
on Jetson.

```bash
mkdir -p artifacts/simple_classifier_dryrun/targets/orin_nano

trtexec \
  --onnx=artifacts/simple_classifier_dryrun/model.onnx \
  --saveEngine=artifacts/simple_classifier_dryrun/targets/orin_nano/model_fp16.engine \
  --fp16
```

Dynamic shape example:

```bash
trtexec \
  --onnx=artifacts/simple_classifier_dryrun/model.onnx \
  --saveEngine=artifacts/simple_classifier_dryrun/targets/orin_nano/model_fp16.engine \
  --fp16 \
  --minShapes=input:1x3x2x2 \
  --optShapes=input:1x3x2x2 \
  --maxShapes=input:4x3x2x2
```

Future optional automation can add a target-side command that checks `trtexec`,
runs the build, and saves logs and metadata.
