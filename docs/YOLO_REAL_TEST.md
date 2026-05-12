# Real YOLO Detection Export Test

## Purpose

This case demonstrates the source-first export strategy for an Ultralytics YOLO
object detection checkpoint. The toolkit does not replace Ultralytics export:
Ultralytics exports the `.pt` model to ONNX, then this toolkit inspects the
checkpoint, validates the exported ONNX artifact, and assesses the recommended
export route.

## Source Model

- checkpoint path: `<PATH_TO_YOLO_CHECKPOINT>`
- task: object detection
- source framework: Ultralytics

## Checkpoint Inspection

Safe inspection failed because the checkpoint references
`ultralytics.nn.tasks.DetectionModel`. Trusted local inspection required
`--unsafe-load`.

The toolkit identified the file as a checkpoint dictionary and reported object
detection as a high-confidence task hint.

```bash
python -m converter.cli inspect "<PATH_TO_YOLO_CHECKPOINT>" --max-items 120 --unsafe-load
```

## Official Ultralytics ONNX Export

The ONNX artifact was produced manually outside this toolkit with the official
Ultralytics exporter:

```bash
yolo export model="<PATH_TO_YOLO_CHECKPOINT>" format=onnx imgsz=640 opset=18
```

Observed official export result:

- ONNX export succeeded
- ONNX input shape: `images: [1, 3, 640, 640]`
- ONNX output shape: `output0: [1, 300, 6]`
- ONNX file saved by Ultralytics before local copy:
  `<PATH_TO_YOLO_ONNX>`

## ONNX Validation With This Toolkit

```bash
python -m converter.cli validate-onnx "<PATH_TO_YOLO_ONNX>" --input-shape 1,3,640,640
```

Observed result:

- ONNX checker passed
- ONNX Runtime session created
- ONNX Runtime inference passed
- input name was `images`
- output name was `output0`

This differs from the PatchCore case. YOLO detection returns a detection tensor
such as `[1, 300, 6]`. PatchCore anomaly detection returns multiple outputs such
as `pred_score`, `pred_label`, `anomaly_map`, and `pred_mask`.

## Export Capability Assessment

```bash
python -m converter.cli assess-export specs/yolo26n_task_detect.yaml
```

Expected reasoning:

- detected framework: Ultralytics
- official ONNX exporter route should be preferred
- generic toolkit ONNX exporter is not the recommended path for this spec
- recommended route: official source exporter

## Optional Downstream Deployment Planning

```bash
python -m converter.cli plan-tensorrt artifacts/yolo26n_task_detect/yolo26n.onnx --spec specs/yolo26n_task_detect.yaml --target orin_nano --precision fp16 --input-name images --min-shape 1x3x640x640 --opt-shape 1x3x640x640 --max-shape 1x3x640x640
```

This is optional deployment planning only. The toolkit does not build TensorRT
engines. TensorRT engine creation belongs on the target device or a matching
TensorRT, CUDA, and runtime environment.
