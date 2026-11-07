# Validation record

Date: 2026-11-07. Platform: Windows CPU. Python: 3.12.14.

## Passed

- `python -m unittest discover -s . -p "test_*.py"`: 170 tests passed. Existing tests include synthetic geometry, mocked inference boundaries, grouping, crop protection, transcription warnings, runtime validation, and bounded IPC. Three `OCR_MODEL_CACHE_BYPASSED` diagnostics are expected from cache-failure tests, not live inference failures.
- `python -m pip check`: no broken requirements in the tested environment.
- Live RapidOCR / ONNXRuntime inference on two newly generated synthetic pages: four numbered blocks in column reading order; a continuation page with only sub-question markers abstained.
- Visual inspection of the generated two-column overlay: four regions retain their figures and tables. OCR symbol/option errors remain and are disclosed in README.
- SHA-256 of all three installed legacy OCR models matched the documented hashes.

## Scope and limitations

The source was copied into a new standalone directory before testing. The original application's existing isolated Python environment supplied dependencies; a clean dependency installation on another machine was not run. No original application files or configuration were changed.

Real-exam OCR fixtures, their dependent replay test, two private-path benchmark scripts, model weights, downloaded wheels, parent application integrations, credentials, configuration files, and prior Git history are excluded. The source project's historical test counts and private-paper accuracy claims are not carried into this release.

This run does not establish accuracy on unseen exams, latency guarantees, real student handwriting recognition, Linux/macOS support, GPU support, or production readiness. The reported per-image times in JSON are one local run only.

Optional Paddle layout, RapidLayout, RapidTable, newer OCR models, UVDoc and OpenVINO were not live-tested for this standalone release. Unit tests that mock their boundaries do not establish model accuracy or runtime compatibility.
