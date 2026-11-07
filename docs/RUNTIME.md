# Runtime and optional experiments

## Default path

`evaluate.py` loads local JPEGs, normalizes orientation and scales once to a maximum 2048-pixel edge, then calls `worker.process_page`. The worker validates JPEG bytes/dimensions/SHA-256, runs RapidOCR on the page and narrow column strips, groups question anchors, assembles warning-bearing text and returns pixel geometry.

`worker.py` is a stdin/stdout JSON worker. In single-request mode, malformed input returns exit 64; unrecoverable geometry exceptions can return 65; runtime/dependency errors return 70. Some uncertain layouts are handled conservatively as a successful response with empty blocks and warnings, so do not use exit status alone to decide success.

`worker.py --serve` is bounded, serial JSON-lines IPC. It caches model sessions, not student result data. The caller must implement its own process timeout, access control, concurrency and storage policies. No web server, authentication or production deployment is included.

The default ONNXRuntime path has no network client. Setup downloads and optional external packages are a separate boundary. `cloud_geometry.py` can consume already-provided Tencent OCR polygons; it does not perform a Tencent API call or contain cloud credentials. Never send student data to a cloud provider merely because an output warning says to verify it there.

## Configuration

- `EXAM_OCR_BACKEND`: `onnxruntime` (default) or optional `openvino`.
- `EXAM_OCR_THREADS`: 1–16, default 2, capped to available CPU count.
- `EXAM_OV_CACHE_DIR`: optional absolute OpenVINO compiled-model cache directory. No effect in the ONNXRuntime path.
- `EXAM_LAYOUT_MODEL_DIR`: optional absolute directory with explicitly acquired PP-DocLayout-L Paddle files. Windows requires an ASCII model path.
- `EXAM_RAPID_LAYOUT_MODEL`: optional absolute `.onnx` path for the PP-DocLayoutV3 RapidLayout adapter. Do not set together with the Paddle layout directory.
- `EXAM_PREPARE`: `off` (default), `perspective`, `shadow`, or `auto`. Geometric preprocessing maps recognized rows back to original upload pixels. It is heuristic, not universal document restoration.
- `EXAM_ENHANCE_FALLBACK=1`: optional shadow-flattening OCR recovery when initial grouping abstains. Default off.

Direct `process_page(...)` calls take explicit optional arguments; environment options are consumed by `configured_process(...)` and the worker CLI. `evaluate.py` calls the default direct API, so setting layout environment variables does not enable them in that script.

## Optional dependencies and model acquisition

Use separate virtual environments. `requirements-openvino.txt` supplements baseline dependencies. `requirements-layout-lock.txt` records the historical Windows Paddle experiment environment; it is not a cross-platform lockfile and was not clean-installed for this release. `requirements-candidate.txt` adds RapidLayout, RapidTable and modern RapidOCR; it was not live-validated here.

For the Paddle research branch, the explicit setup script downloads **both** PP-DocLayout-L and UVDoc from these official Paddle BOS endpoints:

- https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar
- https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar

```sh
python fetch_layout_models.py --directory /absolute/path/to/models
```

This downloads archives, extracts with Python's data filter and writes a local manifest of hashes. It does not compare against a pre-approved digest; verify expected files, license and hashes before execution/deployment. Inference does not invoke the downloader. Do not commit the model folder or its machine-specific manifest.

For RapidLayout/SLANet-plus/newer OCR, consult the official project and model links in `THIRD_PARTY_NOTICES.md`, acquire files explicitly and record their revision/hash/license. Exact converted artifacts are intentionally not recommended as verified until that review is done.

`model_benchmark.py`, `compare_reference.py`, and `replay_benchmark.py` are local research utilities accepting caller-selected paths. No real reference outputs are included. They are not evidence of accuracy on an independent corpus. `table_worker.py` requires `EXAM_RAPID_LAYOUT_MODEL`, `EXAM_TABLE_MODEL`, and optionally `EXAM_TABLE_OCR_DIR`; the modern OCR path has an additional recognition-model hash check.

## Safety and data

Keep real school/exam images outside the repository, with appropriate permission and retention controls. Review bounding boxes and uncertain text before using them. Use a new output directory for evaluation; never point it at the source-image directory. Default inference does not erase handwriting, reconstruct occluded text, solve problems, or grade students.
