"""
Validate that the prompt-compiled single-file model matches the multi-model
pipeline, not just in detection count but in the actual outputs.

For every query slot kept above threshold we compare:
  - raw logits / presence / box agreement (max abs diff)
  - per-detection mask IoU (after resize+threshold to source resolution)
  - per-detection box IoU and score difference

Usage:
    ./run.sh validate_compiled.py --arch v2 \
        --image ../../assets/kids.jpg \
        --compiled-model ./onnx-models/sam3-person-v2.onnx \
        --model-dir ./onnx-models --tokenizer ./onnx-models/tokenizer.json \
        --text "person" --device cuda
"""

import argparse

import cv2
import numpy as np
import onnxruntime as ort

from inference_compiled import preprocess_image


def providers_for(device):
    return (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )


def scores_of(pred_logits, presence_logits):
    presence = 1 / (1 + np.exp(-presence_logits[0, 0]))
    return (1 / (1 + np.exp(-pred_logits[0]))) * presence


def masks_to_bool(pred_masks, keep, w, h):
    """Resize each kept mask logit map to (h, w) and threshold > 0 (as inference does)."""
    out = []
    for m in pred_masks[0][keep]:
        r = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(r > 0)
    return out


def boxes_to_xyxy(pred_boxes, keep, w, h):
    b = pred_boxes[0][keep].copy()
    b[:, [0, 2]] *= w
    b[:, [1, 3]] *= h
    return np.clip(b, 0, [[w, h, w, h]])


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 1.0 if union <= 0 else inter / union


def main():
    p = argparse.ArgumentParser(description="Validate compiled vs pipeline outputs")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--compiled-model", type=str, required=True)
    p.add_argument("--model-dir", type=str, default="onnx-models")
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--text", type=str, default="person")
    p.add_argument("--arch", choices=["v1", "v2"], default="v2")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    providers = providers_for(args.device)
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    px, _ = preprocess_image(image)

    # ---- compiled single-file model ----
    compiled = ort.InferenceSession(args.compiled_model, providers=providers)
    c_masks, c_boxes, c_logits, c_presence = compiled.run(
        None, {compiled.get_inputs()[0].name: px}
    )

    # ---- multi-model pipeline ----
    md = args.model_dir
    if args.arch == "v2":
        from inference_v2 import Sam3ONNXInferenceV2
        eng = Sam3ONNXInferenceV2(
            f"{md}/vision-encoder.onnx", f"{md}/text-encoder.onnx",
            f"{md}/geo-encoder-mask-decoder.onnx", args.tokenizer,
            image_height=1008, image_width=1008, providers=providers,
        )
        vf = eng.encode_image(px)
        tf, tm = eng.encode_text(args.text)
        out = eng.decode(vf, tf, tm,
                         np.zeros((1, 1, 4), np.float32), np.full((1, 1), -10, np.int64))
    else:
        from inference import Sam3ONNXInference
        eng = Sam3ONNXInference(
            f"{md}/vision-encoder.onnx", f"{md}/text-encoder.onnx",
            f"{md}/geometry-encoder.onnx", f"{md}/decoder.onnx", args.tokenizer,
            providers=providers,
        )
        vf = eng.encode_image(px)
        tf, tm = eng.encode_text(args.text)
        out = eng.decode(vf, tf, tm)
    p_masks, p_boxes = out["pred_masks"], out["pred_boxes"]
    p_logits, p_presence = out["pred_logits"], out["presence_logits"]

    print(f"Image {args.image} ({w}x{h}) | prompt '{args.text}' | arch {args.arch} | {args.device}\n")

    # ---- raw tensor agreement ----
    print("Raw output agreement (compiled vs pipeline):")
    for n, a, b in [("pred_logits", c_logits, p_logits),
                    ("presence_logits", c_presence, p_presence),
                    ("pred_boxes", c_boxes, p_boxes),
                    ("pred_masks", c_masks, p_masks)]:
        a = a.astype(np.float32); b = b.astype(np.float32)
        print(f"  {n:16} max abs diff {np.abs(a - b).max():.3e}")

    # ---- detection-level agreement ----
    c_scores = scores_of(c_logits, c_presence)
    p_scores = scores_of(p_logits, p_presence)
    c_keep = c_scores > args.conf
    p_keep = p_scores > args.conf
    same_idx = np.array_equal(c_keep, p_keep)
    print(f"\nDetections @{args.conf}: compiled={int(c_keep.sum())}  "
          f"pipeline={int(p_keep.sum())}  same kept query set: {same_idx}")

    keep = c_keep  # identical to p_keep when same_idx
    cm = masks_to_bool(c_masks, keep, w, h)
    pm = masks_to_bool(p_masks, keep, w, h)
    cb = boxes_to_xyxy(c_boxes, keep, w, h)
    pb = boxes_to_xyxy(p_boxes, keep, w, h)

    mask_ious = [mask_iou(a, b) for a, b in zip(cm, pm)]
    box_ious = [box_iou(a, b) for a, b in zip(cb, pb)]
    score_diffs = np.abs(c_scores[keep] - p_scores[keep])

    if mask_ious:
        print("\nPer-detection similarity (matched by query slot):")
        print(f"  mask IoU   mean {np.mean(mask_ious):.5f}  min {np.min(mask_ious):.5f}")
        print(f"  box  IoU   mean {np.mean(box_ious):.5f}  min {np.min(box_ious):.5f}")
        print(f"  score diff max  {score_diffs.max():.3e}")

    ok = same_idx and (not mask_ious or min(mask_ious) > 0.99)
    print(f"\nVERDICT: {'IDENTICAL ✓' if ok else 'DIVERGENT — inspect above'}")


if __name__ == "__main__":
    main()
