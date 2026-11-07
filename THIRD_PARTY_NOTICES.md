# Third-party components and model provenance

This standalone source distribution does not include dependency wheels, OCR/layout/table weights, original exam pages, commercial SDKs, or original application history. Installing dependencies or separately downloading models is distinct from this repository's code license. Retain the actual packages' licenses and notices when redistributing a bundled application.

Source/license references checked on 2026-11-07. A license label in a model card is the publisher's declaration, not a legal warranty. Check the pinned artifact you actually use and its license if upgrading.

## Default OCR framework

- **RapidOCR / `rapidocr-onnxruntime==1.4.4`: Apache License 2.0.** [Exact PyPI release](https://pypi.org/project/rapidocr-onnxruntime/1.4.4/), [source at the recorded commit](https://github.com/RapidAI/RapidOCR/tree/86ae3f5079df3422c1829cd84baf19bc8a7a9453), [LICENSE](https://raw.githubusercontent.com/RapidAI/RapidOCR/86ae3f5079df3422c1829cd84baf19bc8a7a9453/LICENSE). Copyright 2021 RapidOCR Authors.
- Install with `python -m pip install -r requirements.txt`. The 1.4.4 wheel contains the three legacy ONNX models below; default inference reads the installed package, not a download API.
- Published 1.4.4 wheel SHA-256: `971d7d5f223a7a808662229df1ef69893809d8457d834e6373d3854bc1782cbf`. This is an artifact reference, not a full dependency lock with verified hashes.

## Models bundled in the upstream dependency wheel

Upstream model weights originate from Baidu/PaddleOCR; RapidOCR provides converted ONNX artifacts. Retain applicable upstream copyright and attribution notices.

The three hashes below were recomputed from the installed package during this release's test run and match the exact artifacts in the RapidOCR maintainer's `SWHL/RapidOCR` distribution, whose model repository declares Apache-2.0. They are converted ONNX artifacts, not a claim of byte identity with original Paddle checkpoints.

| Role | Bundled file | SHA-256 | Exact converted artifact |
| --- | --- | --- | --- |
| Text detection | `ch_PP-OCRv4_det_infer.onnx` | `d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9` | [source and license label](https://huggingface.co/SWHL/RapidOCR/blob/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx) |
| Text recognition | `ch_PP-OCRv4_rec_infer.onnx` | `48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b` | [source and license label](https://huggingface.co/SWHL/RapidOCR/blob/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx) |
| Text orientation classifier | `ch_ppocr_mobile_v2.0_cls_infer.onnx` | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` | [source and license label](https://huggingface.co/SWHL/RapidOCR/blob/main/PP-OCRv1/ch_ppocr_mobile_v2.0_cls_infer.onnx) |

Official upstream cards separately declare Apache-2.0 for [PP-OCRv4_mobile_det](https://huggingface.co/PaddlePaddle/PP-OCRv4_mobile_det) and [PP-OCRv4_mobile_rec](https://huggingface.co/PaddlePaddle/PP-OCRv4_mobile_rec). RapidOCR's [model-license statement](https://rapidai.github.io/RapidOCRDocs/latest/model_licenses/) explicitly traces the same classifier hash, under a newer filename, to PaddleOCR's legacy `ch_ppocr_mobile_v2.0_cls`. Its current list does not automatically apply to every historical weight.

If manual model acquisition is needed, the exact download links are:

- https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx
- https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx
- https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv1/ch_ppocr_mobile_v2.0_cls_infer.onnx

Those `main` URLs can change. Verify the documented SHA-256 and review current license metadata; do not silently replace the installed files. Ordinary installation through the pinned wheel is the supported default.

## Runtime dependencies

| Component | License | Reference / caveat |
| --- | --- | --- |
| ONNX Runtime | MIT | [Microsoft LICENSE](https://github.com/Microsoft/onnxruntime/blob/main/LICENSE); preserve packaged third-party notices |
| OpenCV | Apache-2.0 for OpenCV 4.5.0 onward | [OpenCV licensing](https://opencv.org/license/) |
| opencv-python packaging | MIT for packaging scripts | [Packaging project](https://github.com/opencv/opencv-python); the complete wheel also contains OpenCV/native components with separate notices |
| NumPy | BSD-3-Clause | [License documentation](https://github.com/numpy/numpy/blob/main/doc/source/license.rst); native wheel dependencies have additional notices |
| Pillow | MIT-CMU | [Pillow LICENSE](https://github.com/python-pillow/Pillow/blob/main/LICENSE), not generic MIT; preserve bundled image-library notices |
| PyYAML | MIT | [Project](https://github.com/yaml/pyyaml) |
| pyclipper | MIT for Python wrapper; Clipper uses Boost Software License 1.0 | [Project](https://github.com/fonttools/pyclipper); inspect installed package notices |
| Shapely | BSD-3-Clause for Shapely; GEOS is separately LGPL-2.1 | [Project](https://github.com/shapely/shapely); preserve bundled GEOS notices and review the applicable LGPL source/relinking obligations when redistributing binary bundles |

The pinned requirements also include support packages (`six`, `flatbuffers`, `packaging`, `protobuf`, `tqdm`, `colorama`). Their distribution metadata and accompanying license files govern them. This source-only repository is not a complete SBOM or a substitute for notices from the wheels used in a deployment.

## Optional adapters and research tools

No optional models are distributed. None of the following optional live-inference paths was exercised for this standalone release. Code-license declarations do not settle provenance of an arbitrary downloaded weight.

- **OpenVINO `2026.4.1`: Apache-2.0**, [release](https://pypi.org/project/openvino/2026.4.1/). `openvino_v4.py` changes the runtime for the same hash-checked legacy PP-OCRv4 weights. The package page discloses telemetry; review the installed runtime's settings before claiming completely offline/private deployment.
- **PaddleOCR / PaddleX / PaddlePaddle: Apache-2.0 code**, [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE), [PaddleX](https://github.com/PaddlePaddle/PaddleX/blob/develop/LICENSE), [PaddlePaddle](https://github.com/PaddlePaddle/Paddle/blob/develop/LICENSE).
- **PP-DocLayout-L:** [official model card](https://huggingface.co/PaddlePaddle/PP-DocLayout-L) declares Apache-2.0. Download setup is explicit in `fetch_layout_models.py`; the tool records artifact hashes but does not enforce a previously pinned expected digest. Original research recorded `inference.pdiparams` SHA-256 `4df69115349e1e6215629d058e3bc7f087cad7a0d2776ef40c463daf73c38982`; it was not downloaded/revalidated in this release.
- **UVDoc:** the [Paddle-distributed model card](https://huggingface.co/PaddlePaddle/UVDoc) declares Apache-2.0. The [original UVDoc research code](https://github.com/tanguymagne/UVDoc) declares MIT; these are distinct distributions. Original research recorded Paddle `inference.pdiparams` SHA-256 `810488899520e0da843b9bd9769ba4949f1c81e357f0eceb12d4a7da459c3eca`, not revalidated here. UVDoc is only an ablation tool, not the default splitter; corrected coordinates cannot be overlaid on the original without a valid inverse mapping.
- **RapidLayout `1.2.1`: Apache-2.0 code**, [release](https://pypi.org/project/rapid-layout/1.2.1/). [PP-DocLayoutV3 upstream card](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) declares Apache-2.0. The exact converted ONNX used with `rapid_layout_adapter.py` is caller-supplied and not inventoried here. Verify URL, revision, conversion provenance, license and hash before using or redistributing it. Do not extend this statement to other RapidLayout model families.
- **RapidTable `3.0.2`: Apache-2.0 code**, [release](https://pypi.org/project/rapid-table/3.0.2/), [source](https://github.com/RapidAI/RapidTable). [SLANet_plus upstream card](https://huggingface.co/PaddlePaddle/SLANet_plus) declares Apache-2.0; the exact converted model is not verified or distributed here. `local_table.py` is experimental selected-question table evidence, not required for splitting.
- **RapidOCR `3.10.0`: Apache-2.0 code**, [release](https://pypi.org/project/rapidocr/3.10.0/). The [model-license statement](https://rapidai.github.io/RapidOCRDocs/latest/model_licenses/) covers only the listed v6 detection/recognition/classifier hashes. This newer stack is optional for table evidence, not the default OCR stack. A pinned recognition hash in adapter code is not a complete provenance record for the other files.

## Synthetic assets

The examples contain generated exercise text, figures and empty tables, not reproduced exam material. Noto Sans SC was used only to render the example JPEGs; no font binary is included. [Noto CJK source](https://github.com/notofonts/noto-cjk), SIL Open Font License 1.1; [license copy](third_party/SIL-OFL-1.1.txt). Font licensing does not change the repository code license.

## Code provenance boundary

The splitter, grouping rules and adapters were extracted from an existing private application. Existing source notes describe those modules as project-written code and distinguish them from proprietary scanner binaries. The publisher has confirmed the right to publish this module and selected the MIT license for the repository code. This does not change third-party dependency or model licenses.


