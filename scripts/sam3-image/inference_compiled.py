"""
Run a prompt-compiled single-file SAM3 model (see export_compiled.py).

The prompt is already baked into the model, so the only input is the image:

    ./run.sh inference_compiled.py \
        --image ../../assets/kids.jpg \
        --model ./onnx-models/sam3-shoe.onnx \
        --output output-shoe-compiled.png
"""

import argparse

import cv2
import numpy as np
import onnxruntime as ort

# Reuse the original post-processing / drawing so output matches inference.py.
from inference import TARGET_SIZE, visualize_results


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize to target size and normalize to [-1, 1] (matches inference.py)."""
    from PIL import Image as PILImage

    orig_size = image.shape[:2]  # (h, w)
    pil_image = PILImage.fromarray(image)
    resized = np.array(pil_image.resize((TARGET_SIZE, TARGET_SIZE), PILImage.BILINEAR))
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    tensor = normalized.transpose(2, 0, 1)[np.newaxis]  # NCHW
    return tensor, orig_size


def postprocess(outputs, orig_size, conf_threshold):
    """Same logic as Sam3ONNXInference._postprocess (text-only path)."""
    pred_masks, pred_boxes, pred_logits, presence_logits = outputs
    pred_masks = pred_masks[0]
    pred_boxes = pred_boxes[0]
    pred_logits = pred_logits[0]
    presence_logits = presence_logits[0, 0]

    presence_score = 1 / (1 + np.exp(-presence_logits))
    scores = (1 / (1 + np.exp(-pred_logits))) * presence_score
    keep = scores > conf_threshold

    h, w = orig_size
    masks = []
    for m in pred_masks[keep]:
        mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        masks.append(mask_resized > 0)
    boxes = pred_boxes[keep].copy()
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    boxes = np.clip(boxes, 0, [[w, h, w, h]])

    return {"masks": masks, "boxes": boxes, "scores": scores[keep], "orig_size": orig_size}


def main():
    parser = argparse.ArgumentParser(description="SAM3 single-file (prompt-compiled) inference")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--model", type=str, required=True, help="Compiled single .onnx")
    parser.add_argument("--output", type=str, default="output-compiled.png")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.device == "cuda"
        else ["CPUExecutionProvider"]
    )
    print("Loading compiled model...")
    session = ort.InferenceSession(args.model, providers=providers)
    print("  ✓ Loaded (single input:", session.get_inputs()[0].name, ")")

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"Cannot load image: {args.image}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"\nProcessing: {args.image} ({image.shape[1]}x{image.shape[0]})")

    pixel_values, orig_size = preprocess_image(image)
    outputs = session.run(None, {session.get_inputs()[0].name: pixel_values})
    results = postprocess(outputs, orig_size, args.conf)

    print(f"  Found {len(results['masks'])} objects")
    visualize_results(image_bgr, results, args.output)


if __name__ == "__main__":
    main()
