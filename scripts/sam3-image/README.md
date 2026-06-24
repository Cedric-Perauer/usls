# SAM3 Image ONNX Export & Inference

## Updates

| Version | ONNX Files | Resolution | Notes |
|---------|------------|------------|-------|
| **v1(4 onnx)** | Vision-encoder<br>Text-encoder<br>Geometry-encoder<br>Decoder | 1008×1008 | Original architecture |
| **v2(3 onnx)** | Vision-encoder<br>Text-encoder<br>Geo-Encoder-Mask-Decoder | 1008×1008 | Geometry integrated into Decoder |
| **compiled(1 onnx)** | Single fused file (Vision + Decoder, text prompt baked in) | 1008×1008 | One fixed text prompt; image-only input, no tokenizer/text-encoder at runtime. See [Prompt-Compiled Single-File Model](#prompt-compiled-single-file-model-merged) |


## Reference
- **Exported ONNX models:** https://github.com/jamjamjon/assets/releases/tag/sam3
- **Sam3Image Demo (text & bboxes prompts):** [sam3-image](../../examples/open-set-segmentation/README.md#sam3-image)
- **Sam3Tracker Demo (points & bboxes prompts):** [sam3-tracker](../../examples/image-segmentation/README.md#sam3-tracker)
- **Sam3ImageCompiled Demo (prompt-compiled, single file):** [sam3-image-compiled](../../examples/sam3-image-compiled/main.rs)
- **Sam3Image Implementation:** [sam3_image](../../src/models/vlm/sam3_image)
- **Sam3ImageCompiled Implementation:** [sam3_image_compiled](../../src/models/vlm/sam3_image_compiled)
- **Sam3Tracker Implementation:** [sam3_tracker](../../src/models/vision/sam3_tracker)


## Export ONNX Models

### v2 (3 ONNX, 1008×1008) - Recommended
```bash
uv run export_v2.py --all \
  --model-path /path/to/sam3-models \
  --output-dir onnx-models-v2 \
  --device cuda \
  --image-height 1008 --image-width 1008
```


### v1 (4 ONNX, 1008×1008)

<details>
<summary>Click to expand v1 export commands</summary>

```bash
uv run export.py --all --model-path /path/to/sam3-models --output-dir onnx-models
```

</details>



## Inference Code

### Python (v2, 1008×1008) - Recommended

```bash
# Text prompt
uv run inference_v2.py \
  --image ../../assets/kids.jpg \
  --text "shoe" \
  --model-dir ./onnx-models-v2 \
  --tokenizer /path/to/tokenizer.json \
  --output output-text-v2.png \
  --device cuda \
  --image-height 1008 --image-width 1008

# Box prompt (xywh format: x,y,w,h)
uv run inference_v2.py \
  --image ../../assets/kids.jpg \
  --boxes "pos:480,290,110,360" \
  --model-dir ./onnx-models-v2 \
  --tokenizer /path/to/tokenizer.json \
  --output output-box-v2.png \
  --device cuda \
  --image-height 1008 --image-width 1008

# Box prompt with positive + negative
uv run inference_v2.py \
  --image ../../assets/kids.jpg \
  --boxes "pos:480,290,110,360;neg:370,280,115,375" \
  --model-dir ./onnx-models-v2 \
  --tokenizer /path/to/tokenizer.json \
  --output output-box-posneg-v2.png \
  --device cuda \
  --image-height 1008 --image-width 1008


# Text + Negative box (mixed prompt)
uv run inference_v2.py \
    --image ../../assets/oven.jpg \
    --text "handle" \
    --boxes "neg:40,183,278,21" \
    --model-dir ./onnx-models-v2 \
    --tokenizer /path/to/tokenizer.json \
    --output output-text-box-v2.png \
    --device cuda \
    --image-height 1008 --image-width 1008

```

### Python (v1, 1008×1008)

<details>
<summary>Click to expand v1 inference commands</summary>

```bash
# Text prompt
uv run inference.py \
    --image ../../assets/kids.jpg \
    --text "shoe" \
    --model-dir ./onnx-models \
    --tokenizer /path/to/tokenizer.json \
    --output output-text-v1.png

# Box prompt (xywh format: x,y,w,h)
uv run inference.py \
    --image ../../assets/kids.jpg \
    --boxes "pos:480,290,110,360" \
    --model-dir ./onnx-models \
    --tokenizer /path/to/tokenizer.json \
    --output output-box-v1.png

# Positive + Negative box
uv run inference.py \
    --image ../../assets/kids.jpg \
    --boxes "pos:480,290,110,360;neg:370,280,115,375" \
    --model-dir ./onnx-models \
    --tokenizer /path/to/tokenizer.json \
    --output output-box-posneg-v1.png

# Text + Negative box (mixed prompt)
uv run inference.py \
    --image ../../assets/oven.jpg \
    --text "handle" \
    --boxes "neg:40,183,278,21" \
    --model-dir ./onnx-models \
    --tokenizer /path/to/tokenizer.json \
    --output output-text-box-v1.png
```

</details>



## Prompt-Compiled Single-File Model (merged)

A variant where **one fixed text prompt is baked into the model** and the vision
encoder + decoder are **fused into a single ONNX file**. The only runtime input is
the image — no tokenizer and no text encoder are run at inference time.

```
images [1, 3, 1008, 1008]  ->  pred_masks, pred_boxes, pred_logits, presence_logits
```

How it works: the text encoder is run **once at export time** and its outputs
(`text_features` / `text_mask`) are frozen into the graph as constant initializers
feeding the decoder. The prompt is baked as a batch-1 constant, so the compiled
model runs **one image at a time** (batch = 1). Box/geometry prompts are not
supported in the compiled file (their features depend on the per-image vision
features); use the multi-file pipeline above for those.

Rust counterpart: [`Sam3ImageCompiled`](../../src/models/vlm/sam3_image_compiled)
([`examples/sam3-image-compiled`](../../examples/sam3-image-compiled/main.rs)).

### Export the compiled model

Requires the **v2** multi-file ONNX (`vision-encoder.onnx`, `text-encoder.onnx`,
`geo-encoder-mask-decoder.onnx`) and the `tokenizer.json`.

```bash
# Bake the prompt "person" into a single file (v2, recommended)
./run.sh export_compiled.py --arch v2 \
    --text "person" \
    --model-dir ./onnx-models \
    --tokenizer ./onnx-models/tokenizer.json \
    --output ./onnx-models/sam3-person-v2.onnx
```

The fused graph is ~1.9 GB, so weights are written next to the `.onnx` as an
external-data sidecar (`sam3-person-v2.onnx.data`); ONNX Runtime loads it
automatically. Keep the two files together.

<details>
<summary>v1 (4-file) variant</summary>

```bash
./run.sh export_compiled.py --arch v1 \
    --text "person" \
    --model-dir ./onnx-models \
    --tokenizer ./onnx-models/tokenizer.json \
    --output ./onnx-models/sam3-person-v1.onnx
```

</details>

### Inference (Python)

The prompt is already baked in, so only the image is needed:

```bash
./run.sh inference_compiled.py \
    --image ../../assets/kids.jpg \
    --model ./onnx-models/sam3-person-v2.onnx \
    --output output-person-compiled.png \
    --device cuda           # or cpu
```

### Validate against the multi-file pipeline

Confirms the compiled model reproduces the full pipeline (per-detection mask /
box IoU, score and raw-output agreement):

```bash
./run.sh validate_compiled.py --arch v2 \
    --image ../../assets/kids.jpg \
    --compiled-model ./onnx-models/sam3-person-v2.onnx \
    --model-dir ./onnx-models --tokenizer ./onnx-models/tokenizer.json \
    --text "person" --device cuda
```

### Benchmark: compiled vs. pipeline

Compares per-image latency of the fused single-file model against the multi-file
pipeline. `--provider` selects the execution provider (`cuda`, `trt`, or `cpu`);
`trt` uses the TensorRT EP at fp16 with engine caching.

```bash
./run.sh benchmark.py \
    --image ../../assets/kids.jpg \
    --compiled-model ./onnx-models/sam3-person-v2.onnx \
    --model-dir ./onnx-models --tokenizer ./onnx-models/tokenizer.json \
    --text "person" --arch v2 --provider trt --cache-dir ./trt-cache \
    --runs 30 --warmup 5
```

It reports three paths:
- **COMPILED** — one `session.run` on the fused, prompt-baked model.
- **PIPELINE** — text-encode + vision-encode + decode every call (what `inference.py` does).
- **PIPELINE+cache** — text features computed once, then vision-encode + decode per call.

**Measured** — RTX 4090, 1008×1008, prompt `"person"` on `assets/kids.jpg`,
fp16, 30 runs / 5 warmup. Detections match the full pipeline (6 = 6). Pure ONNX
compute (pre/post-processing excluded):

| Provider | COMPILED | PIPELINE+cache | PIPELINE | Compiled speedup |
|----------|---------:|---------------:|---------:|:----------------:|
| **TensorRT (fp16)** | **39.1 ms** (25.6 img/s) | 53.6 ms (18.7 img/s) | 55.4 ms (18.1 img/s) | **1.42×** vs PIPELINE |
| **CUDA (fp16)**     | 202.1 ms (4.9 img/s) | 219.4 ms (4.6 img/s) | 222.9 ms (4.5 img/s) | 1.10× vs PIPELINE |

Takeaways:
- **TensorRT is ~5× faster** than the CUDA EP here (39 ms vs 202 ms for the
  compiled model) — the TensorRT runtime is where this model wants to run.
- **Fusing into one engine helps most under TensorRT** (1.42× vs the 3-engine
  pipeline) because the per-op cost is small, so avoiding the cross-engine
  feature hand-offs (vision → text → decoder) is a larger share of the time.
  Under the CUDA EP the kernels dominate, so fusion buys less (1.10×).
- Caching the (fixed) text features barely moves the pipeline — the text encoder
  is cheap; the vision encoder and decoder dominate.

> **Provider / driver note.** The TensorRT EP and the `CUDAExecutionProvider`
> must match the machine's NVIDIA driver. `onnxruntime-gpu` ≥ 1.23 on PyPI is a
> **CUDA-13** build (needs driver ≥ 580); on a CUDA-12 driver install a CUDA-12
> build instead, e.g.
> `uv pip install "onnxruntime-gpu==1.22.0" --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`,
> together with a matching CUDA-12 `tensorrt` (`tensorrt-cu12==10.x`, providing
> `libnvinfer.so.10`) and `nvidia-cudnn-cu12==9.x`.


## TensorRT Conversion

Choose `optShapes` and `maxShapes` according to your GPU memory.

### Compiled single-file model (image-only, batch 1)

The fused model has a single `images` input, so the engine build is one command:

```bash
trtexec --fp16 --onnx=onnx-models/sam3-person-v2.onnx \
    --shapes=images:1x3x1008x1008 \
    --saveEngine=onnx-models/sam3-person-v2.engine
```

Or let the ONNX Runtime **TensorRT EP** build/cache the engine on first run (what
`benchmark.py --provider trt` and the Rust `--device tensorrt:0` path do) — no
separate `trtexec` step needed.


### v2 (1008×1008) - Recommended

```bash
# Vision Encoder
/home/qweasd/Documents/TensorRT-10.11.0.33/bin/trtexec --fp16 --onnx=onnx-models-v2/vision-encoder.onnx \
    --minShapes=images:1x3x1008x1008 \
    --optShapes=images:4x3x1008x1008 \
    --maxShapes=images:8x3x1008x1008 \
    --saveEngine=onnx-models-v2/vision-encoder.engine

# Text Encoder
/home/qweasd/Documents/TensorRT-10.11.0.33/bin/trtexec --fp16 --onnx=onnx-models-v2/text-encoder.onnx \
    --minShapes=input_ids:1x32,attention_mask:1x32 \
    --optShapes=input_ids:4x32,attention_mask:4x32 \
    --maxShapes=input_ids:8x32,attention_mask:8x32 \
    --saveEngine=onnx-models-v2/text-encoder.engine

# Decoder (with integrated Geometry Encoder)
/home/qweasd/Documents/TensorRT-10.11.0.33/bin/trtexec --fp16 --onnx=onnx-models-v2/decoder.onnx \
    --minShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,text_features:1x32x256,text_mask:1x32,input_boxes:1x1x4,input_boxes_labels:1x1 \
    --optShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,text_features:1x32x256,text_mask:1x32,input_boxes:1x8x4,input_boxes_labels:1x8 \
    --maxShapes=fpn_feat_0:8x256x288x288,fpn_feat_1:8x256x144x144,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72,text_features:8x32x256,text_mask:8x32,input_boxes:8x20x4,input_boxes_labels:8x20 \
    --saveEngine=onnx-models-v2/decoder.engine
```

### v1 (1008×1008, 4 ONNX)

<details>
<summary>Click to expand v1 TensorRT conversion commands</summary>

```bash
# Vision Encoder
trtexec --fp16 --onnx=onnx-models/vision-encoder.onnx \
    --minShapes=images:1x3x1008x1008 \
    --optShapes=images:4x3x1008x1008 \
    --maxShapes=images:8x3x1008x1008 \
    --saveEngine=onnx-models/vision-encoder.engine

# Text Encoder
trtexec --fp16 --onnx=onnx-models/text-encoder.onnx \
    --minShapes=input_ids:1x32,attention_mask:1x32 \
    --optShapes=input_ids:4x32,attention_mask:4x32 \
    --maxShapes=input_ids:8x32,attention_mask:8x32 \
    --saveEngine=onnx-models/text-encoder.engine

# Geometry Encoder
trtexec --fp16 --onnx=onnx-models/geometry-encoder.onnx \
    --minShapes=input_boxes:1x1x4,input_boxes_labels:1x1,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72 \
    --optShapes=input_boxes:1x8x4,input_boxes_labels:1x8,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72 \
    --maxShapes=input_boxes:8x20x4,input_boxes_labels:8x20,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72 \
    --saveEngine=onnx-models/geometry-encoder.engine

# Decoder
trtexec --fp16 --onnx=onnx-models/decoder.onnx \
    --minShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x1x256,prompt_mask:1x1 \
    --optShapes=fpn_feat_0:1x256x288x288,fpn_feat_1:1x256x144x144,fpn_feat_2:1x256x72x72,fpn_pos_2:1x256x72x72,prompt_features:1x33x256,prompt_mask:1x33 \
    --maxShapes=fpn_feat_0:8x256x288x288,fpn_feat_1:8x256x144x144,fpn_feat_2:8x256x72x72,fpn_pos_2:8x256x72x72,prompt_features:8x60x256,prompt_mask:8x60 \
    --saveEngine=onnx-models/decoder.engine
```

</details>

## Rust Inference & TensorRT (usls)

The compiled single-file model is also served from Rust by
[`Sam3ImageCompiled`](../../src/models/vlm/sam3_image_compiled), via the
[`sam3-image-compiled`](../../examples/sam3-image-compiled/main.rs) example. The
device is chosen at runtime with `--device` (`cpu`, `cuda:0`, `tensorrt:0`, …),
parsed by [`Device`](../../src/utils/device.rs) (`"trt" | "tensorrt"`).

### CUDA

```bash
cargo run -r -F vlm -F annotator -F cuda-12040 --example sam3-image-compiled -- \
    --model scripts/sam3-image/onnx-models/sam3-person-v2.onnx \
    --name person --source assets/kids.jpg --device cuda:0
```

### TensorRT

Add a `tensorrt-*` build feature and pass `--device tensorrt:0`. Pick the feature
that matches your CUDA toolkit (here CUDA 12.4 → `tensorrt-cuda-12040`); the bare
`tensorrt` feature also works if you don't need CUDA-accelerated image processing.

```bash
cargo run -r -F vlm -F annotator -F tensorrt-cuda-12040 --example sam3-image-compiled -- \
    --model scripts/sam3-image/onnx-models/sam3-person-v2.onnx \
    --name person --source assets/kids.jpg --device tensorrt:0
```

The **first run builds and serializes the TensorRT engine (a few minutes)** —
`usls` prints `Initial model serialization with TensorRT may require a wait...`.
The keep-it-together `.onnx` + `.onnx.data` sidecar is required.

> **Verified on this machine** (RTX 4090, driver 570.195 / CUDA 12.8, glibc 2.35):
> engine build ~2m47s on the first run, then 6 `person` detections (conf
> ≈0.70–0.74), matching the Python pipeline. A benign `SIGSEGV` can occur at
> process teardown (CUDA/TensorRT context cleanup ordering) **after** inference
> finishes and the result is saved — it does not affect the output.

#### Older glibc (< 2.38): use a dynamically-loaded onnxruntime

`usls`'s default `ort-download-binaries` links a prebuilt onnxruntime that needs
**glibc ≥ 2.38**. On glibc 2.35 (Ubuntu 22.04) the static link fails with
`undefined reference to __isoc23_strtoll`. Build against a CUDA-12 onnxruntime you
provide instead, using `ort-load-dynamic` (this is how the run above was
verified):

```bash
# Build: drop download-binaries, load onnxruntime dynamically, match its API level
cargo build -r --no-default-features \
    -F vision,vlm,annotator,ort-api-22,ort-load-dynamic,tensorrt-cuda-12040 \
    --example sam3-image-compiled

# Run: point ORT at a CUDA-12 onnxruntime (e.g. the one in scripts/sam3-image/.venv)
SP=scripts/sam3-image/.venv/lib/python3.12/site-packages
export ORT_DYLIB_PATH=$SP/onnxruntime/capi/libonnxruntime.so.1.22.0
export LD_LIBRARY_PATH=$SP/onnxruntime/capi:$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/tensorrt_libs:/usr/local/cuda/targets/x86_64-linux/lib
./target/release/examples/sam3-image-compiled \
    --model scripts/sam3-image/onnx-models/sam3-person-v2.onnx \
    --name person --source assets/kids.jpg --device tensorrt:0
```

`ort-api-22` matches `onnxruntime` 1.22; use the API level that matches whichever
onnxruntime you point `ORT_DYLIB_PATH` at.


## ONNX Model Specifications

All models support dynamic batch processing.

### Compiled single-file model (prompt baked in)

```
Inputs:
  images               [1, 3, 1008, 1008]        FLOAT   (batch fixed to 1)

Outputs:
  pred_masks           [1, 200, 288, 288]        FLOAT
  pred_boxes           [1, 200, 4]               FLOAT
  pred_logits          [1, 200]                  FLOAT
  presence_logits      [1, 1]                    FLOAT
```

Post-processing (identical to the v2 text-only path):
`score = sigmoid(pred_logits) * sigmoid(presence_logits)`, keep `score > conf`,
resize each kept mask to the source resolution, and scale the normalized boxes to
pixels.

### v2 (3 ONNX, 1008×1008)

**Vision Encoder**
```
Inputs:
  images                [batch, 3, 1008, 1008]    FLOAT

Outputs:
  fpn_feat_0            [batch, 256, 288, 288]    FLOAT
  fpn_feat_1            [batch, 256, 144, 144]    FLOAT
  fpn_feat_2            [batch, 256, 72, 72]      FLOAT
  fpn_pos_2             [batch, 256, 72, 72]      FLOAT
```

**Text Encoder**
```
Inputs:
  input_ids             [batch, 32]               INT64
  attention_mask        [batch, 32]               INT64

Outputs:
  text_features         [batch, 32, 256]          FLOAT
  text_mask             [batch, 32]               BOOL
```

**Decoder (with integrated Geometry Encoder)**
```
Inputs:
  fpn_feat_0            [batch, 256, 288, 288]    FLOAT
  fpn_feat_1            [batch, 256, 144, 144]    FLOAT
  fpn_feat_2            [batch, 256, 72, 72]      FLOAT
  fpn_pos_2             [batch, 256, 72, 72]      FLOAT
  text_features         [batch, 32, 256]          FLOAT
  text_mask             [batch, 32]               BOOL
  input_boxes           [batch, num_boxes, 4]     FLOAT
  input_boxes_labels    [batch, num_boxes]        INT64  (1=pos, 0=neg, -10=ignore)

Outputs:
  pred_masks            [batch, 200, 288, 288]    FLOAT
  pred_boxes            [batch, 200, 4]           FLOAT
  pred_logits           [batch, 200]              FLOAT
  presence_logits       [batch, 1]                FLOAT
```


### v1 (4 ONNX, 1008×1008)

<details>
<summary>Click to expand v1 TensorRT conversion commands</summary>


**Vision Encoder** - Same as v2 1008×1008

**Text Encoder** - Same as v2

**Geometry Encoder**
```
Inputs:
  input_boxes           [batch, num_boxes, 4]     FLOAT
  input_boxes_labels    [batch, num_boxes]        INT64
  fpn_feat_2            [batch, 256, 72, 72]      FLOAT
  fpn_pos_2             [batch, 256, 72, 72]      FLOAT

Outputs:
  geometry_features     [batch, num_boxes+1, 256] FLOAT
  geometry_mask         [batch, num_boxes+1]      BOOL
```

**Decoder**
```
Inputs:
  fpn_feat_0            [batch, 256, 288, 288]    FLOAT
  fpn_feat_1            [batch, 256, 144, 144]    FLOAT
  fpn_feat_2            [batch, 256, 72, 72]      FLOAT
  fpn_pos_2             [batch, 256, 72, 72]      FLOAT
  prompt_features       [batch, prompt_len, 256]  FLOAT  (text + geometry concatenated)
  prompt_mask           [batch, prompt_len]       BOOL

Outputs:
  pred_masks            [batch, 200, 288, 288]    FLOAT
  pred_boxes            [batch, 200, 4]           FLOAT
  pred_logits           [batch, 200]              FLOAT
  presence_logits       [batch, 1]                FLOAT
```

</details>
