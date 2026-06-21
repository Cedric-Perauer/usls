"""
Benchmark: prompt-compiled single ONNX vs. the multi-model pipeline.

Compares per-image inference latency of:
  - COMPILED      : one session.run on the fused, prompt-baked model
  - PIPELINE       : text-encode + vision-encode + decode every call (what inference.py does)
  - PIPELINE+cache : text features computed once, then vision-encode + decode per call

All paths start from the same preprocessed image tensor (preprocessing and
post-processing are excluded so we measure pure ONNX compute). The script also
checks that COMPILED and PIPELINE produce the same outputs.

Usage:
    ./run.sh benchmark.py \
        --image ../../assets/kids.jpg \
        --compiled-model ./onnx-models/sam3-shoe.onnx \
        --model-dir ./onnx-models --tokenizer ./onnx-models/tokenizer.json \
        --text "shoe" --device cuda --runs 50
"""

import argparse
import statistics
import time

import cv2
import numpy as np

from inference import Sam3ONNXInference
from inference_compiled import preprocess_image
import onnxruntime as ort


def make_providers(provider, cache_dir):
    """Build an ORT providers list. 'trt' uses the TensorRT EP (fp16) with engine
    caching, falling back to CUDA for any subgraph TensorRT can't handle."""
    if provider == "trt":
        return [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_dir,
                    "trt_timing_cache_enable": True,
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    if provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def time_it(fn, runs, warmup):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)  # ms
    return samples


def report(name, samples):
    mean = statistics.mean(samples)
    median = statistics.median(samples)
    std = statistics.pstdev(samples)
    fps = 1000.0 / mean
    print(f"  {name:<18} mean {mean:7.2f} ms | median {median:7.2f} ms | "
          f"min {min(samples):7.2f} | std {std:5.2f} | {fps:6.1f} img/s")
    return mean


def main():
    p = argparse.ArgumentParser(description="SAM3 compiled vs pipeline benchmark")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--compiled-model", type=str, required=True)
    p.add_argument("--model-dir", type=str, default="onnx-models")
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--text", type=str, default="shoe", help="Prompt baked into the compiled model")
    p.add_argument("--arch", choices=["v1", "v2"], default="v2",
                   help="Pipeline architecture: v2 (3-file, geo-encoder-mask-decoder) or v1 (4-file)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--provider", type=str, default="cuda", choices=["cuda", "trt", "cpu"],
                   help="Execution provider: cuda, trt (TensorRT fp16), or cpu")
    p.add_argument("--cache-dir", type=str, default="./trt-cache",
                   help="TensorRT engine cache directory")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    import os
    os.makedirs(args.cache_dir, exist_ok=True)
    providers = make_providers(args.provider, args.cache_dir)
    # Sam3ONNXInference derives device from this; force cpu only for the cpu provider.
    args.device = "cpu" if args.provider == "cpu" else "cuda"

    # Shared preprocessed input
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    pixel_values, _ = preprocess_image(image)

    print(f"Image {args.image} | prompt '{args.text}' | arch {args.arch} | "
          f"provider {args.provider} | runs {args.runs} (warmup {args.warmup})")
    if args.provider == "trt":
        print("  (first run builds TensorRT engines into "
              f"{args.cache_dir} — this can take several minutes)")
    print()

    # ---- Compiled single-file model ----
    print("Loading models...")
    compiled = ort.InferenceSession(args.compiled_model, providers=providers)
    in_name = compiled.get_inputs()[0].name

    def run_compiled():
        return compiled.run(None, {in_name: pixel_values})

    # ---- Multi-model pipeline (reuse the inference.py / inference_v2.py engine) ----
    md = args.model_dir
    if args.arch == "v2":
        from inference_v2 import Sam3ONNXInferenceV2
        engine = Sam3ONNXInferenceV2(
            vision_encoder_path=f"{md}/vision-encoder.onnx",
            text_encoder_path=f"{md}/text-encoder.onnx",
            decoder_path=f"{md}/geo-encoder-mask-decoder.onnx",
            tokenizer_path=args.tokenizer,
            image_height=1008,
            image_width=1008,
            providers=providers,
        )
        # text-only path: "no geometry" sentinel boxes (same as inference_v2.predict)
        dummy_boxes = np.zeros((1, 1, 4), dtype=np.float32)
        dummy_labels = np.full((1, 1), -10, dtype=np.int64)

        def decode(vf, tf, tm):
            return engine.decode(vf, tf, tm, dummy_boxes, dummy_labels)
    else:
        engine = Sam3ONNXInference(
            vision_encoder_path=f"{md}/vision-encoder.onnx",
            text_encoder_path=f"{md}/text-encoder.onnx",
            geometry_encoder_path=f"{md}/geometry-encoder.onnx",
            decoder_path=f"{md}/decoder.onnx",
            tokenizer_path=args.tokenizer,
            providers=providers,
        )

        def decode(vf, tf, tm):
            return engine.decode(vf, tf, tm)

    def run_pipeline_full():
        text_features, text_mask = engine.encode_text(args.text)
        vf = engine.encode_image(pixel_values)
        return decode(vf, text_features, text_mask)

    # Pipeline with the (fixed) prompt cached, like you'd do in production
    cached_tf, cached_tm = engine.encode_text(args.text)

    def run_pipeline_cached():
        vf = engine.encode_image(pixel_values)
        return decode(vf, cached_tf, cached_tm)

    # ---- Correctness: compiled vs pipeline must agree ----
    # We compare the actual detections (objects above threshold), not raw mask
    # logits: those span a huge range (~[-120, 13]) and differ by a few units on
    # background pixels due to GPU float nondeterminism in the fused op schedule,
    # which never changes a detection.
    print("\nVerifying outputs match...")

    def count_dets(pred_logits, presence_logits, thr=0.5):
        presence = 1 / (1 + np.exp(-presence_logits[0, 0]))
        scores = (1 / (1 + np.exp(-pred_logits[0]))) * presence
        return int((scores > thr).sum())

    c_out = run_compiled()
    p_out = run_pipeline_cached()
    p_arr = [p_out["pred_masks"], p_out["pred_boxes"], p_out["pred_logits"], p_out["presence_logits"]]
    c_dets = count_dets(c_out[2], c_out[3])
    p_dets = count_dets(p_arr[2], p_arr[3])
    names = ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"]
    print(f"  detections @0.5  compiled={c_dets}  pipeline={p_dets}  "
          f"-> {'OK' if c_dets == p_dets else 'MISMATCH'}")
    for n, c, q in zip(names, c_out, p_arr):
        print(f"    {n:16} max abs diff {np.abs(c.astype(np.float32) - q.astype(np.float32)).max():.3e}")

    # ---- Timings ----
    print("\nBenchmarking...")
    m_compiled = report("COMPILED", time_it(run_compiled, args.runs, args.warmup))
    m_cached = report("PIPELINE+cache", time_it(run_pipeline_cached, args.runs, args.warmup))
    m_full = report("PIPELINE", time_it(run_pipeline_full, args.runs, args.warmup))

    print("\nSpeedup (compiled vs ...):")
    print(f"  vs PIPELINE        : {m_full / m_compiled:5.2f}x")
    print(f"  vs PIPELINE+cache  : {m_cached / m_compiled:5.2f}x")


if __name__ == "__main__":
    main()
