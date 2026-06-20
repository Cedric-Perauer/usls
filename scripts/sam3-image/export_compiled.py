"""
Compile a fixed text prompt into a single SAM3 ONNX file.

Given the SAM3 ONNX models and a text prompt, this produces ONE self-contained
ONNX model whose only input is the preprocessed image and whose outputs are the
raw decoder predictions:

    images [1, 3, 1008, 1008]  ->  pred_masks, pred_boxes, pred_logits, presence_logits

The prompt is "pre-compiled": the text encoder is run once at export time and its
outputs (text_features / text_mask) are frozen into the graph as constant
initializers feeding the decoder. The text encoder and tokenizer are therefore
NOT needed at inference time.

Two architectures:
  v2 (default, 3-file, matches the Rust `Sam3Image`): fuses
      vision-encoder + geo-encoder-mask-decoder. The decoder takes text_features,
      text_mask, input_boxes, input_boxes_labels directly; for a text-only prompt
      we freeze text_features/text_mask and the "no geometry" sentinel
      (input_boxes = zeros[1,1,4], input_boxes_labels = [[-10]]).
  v1 (4-file): fuses vision-encoder + decoder, where the decoder takes a single
      prompt_features/prompt_mask (text only).

Note: the prompt is baked as a batch-1 constant, so the exported model runs one
image at a time (batch = 1). Box/geometry prompts are not supported here because
their features depend on the per-image vision features.

Usage (v2, recommended):
    ./run.sh export_compiled.py --arch v2 \
        --text "person" \
        --model-dir ./onnx-models \
        --tokenizer ./onnx-models/tokenizer.json \
        --output ./onnx-models/sam3-person-v2.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import compose, helper, numpy_helper
from tokenizers import Tokenizer

PROMPT_LEN = 32
PAD_ID = 49407
FPN_TENSORS = ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]
OUTPUT_NAMES = ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"]

ARCH = {
    "v1": {"decoder_file": "decoder.onnx"},
    "v2": {"decoder_file": "geo-encoder-mask-decoder.onnx"},
}


def encode_text(text_encoder_path, tokenizer_path, text, device):
    """Run the text encoder once to get the constant prompt features/mask."""
    tok = Tokenizer.from_file(tokenizer_path)
    tok.enable_padding(pad_id=PAD_ID, length=PROMPT_LEN)
    tok.enable_truncation(max_length=PROMPT_LEN)
    enc = tok.encode(text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(text_encoder_path, providers=providers)
    text_features, text_mask = sess.run(
        None, {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    return text_features.astype(np.float32), text_mask


def frozen_inputs(arch, text_features, text_mask):
    """The (prefixed) decoder inputs to bake as constants, per architecture."""
    if arch == "v1":
        return {
            "decoder/prompt_features": text_features,
            "decoder/prompt_mask": text_mask,
        }
    # v2: text + the "no geometry" sentinel boxes
    return {
        "decoder/text_features": text_features,
        "decoder/text_mask": text_mask,
        "decoder/input_boxes": np.zeros((1, 1, 4), dtype=np.float32),
        "decoder/input_boxes_labels": np.full((1, 1), -10, dtype=np.int64),
    }


def build_single_model(vision_path, decoder_path, frozen):
    """Merge vision-encoder + decoder and freeze the prompt as constants."""
    vision = onnx.load(vision_path)
    decoder = onnx.load(decoder_path)

    # Namespace the decoder so its internal names cannot collide with the
    # vision encoder's during merge.
    decoder = compose.add_prefix(decoder, prefix="decoder/")

    # Wire vision-encoder outputs into the (prefixed) decoder inputs.
    io_map = [(name, f"decoder/{name}") for name in FPN_TENSORS]
    merged = compose.merge_models(vision, decoder, io_map=io_map)
    graph = merged.graph

    # Drop the frozen inputs from graph inputs, add them back as constants.
    kept_inputs = [i for i in graph.input if i.name not in frozen]
    del graph.input[:]
    graph.input.extend(kept_inputs)
    for name, arr in frozen.items():
        graph.initializer.append(numpy_helper.from_array(arr, name=name))

    # Friendly output names (decoder/pred_masks -> pred_masks) via Identity nodes.
    new_outputs = []
    for out in graph.output:
        nice = out.name.split("decoder/")[-1]
        nice = nice if nice in OUTPUT_NAMES else out.name
        graph.node.append(helper.make_node("Identity", [out.name], [nice]))
        vi = onnx.ValueInfoProto()
        vi.CopyFrom(out)
        vi.name = nice
        new_outputs.append(vi)
    del graph.output[:]
    graph.output.extend(new_outputs)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Compile a fixed text prompt into a single SAM3 ONNX model"
    )
    parser.add_argument("--arch", choices=["v1", "v2"], default="v2")
    parser.add_argument("--text", type=str, required=True, help="Text prompt to bake in")
    parser.add_argument("--model-dir", type=str, default="onnx-models")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.json")
    parser.add_argument("--output", type=str, required=True, help="Output single .onnx path")
    parser.add_argument("--decoder-file", type=str, default=None,
                        help="Override the decoder onnx filename (default per --arch)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for the one-off text encode")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_file = args.decoder_file or ARCH[args.arch]["decoder_file"]

    print(f"Compiling prompt '{args.text}' into a single ONNX model ({args.arch}, "
          f"decoder={decoder_file})...")
    print("  Running text encoder once...")
    text_features, text_mask = encode_text(
        str(model_dir / "text-encoder.onnx"), args.tokenizer, args.text, args.device
    )
    print(f"    text_features {text_features.shape} {text_features.dtype}, "
          f"text_mask {text_mask.shape} {text_mask.dtype}")

    print("  Merging vision-encoder + decoder and freezing the prompt...")
    frozen = frozen_inputs(args.arch, text_features, text_mask)
    merged = build_single_model(
        str(model_dir / "vision-encoder.onnx"),
        str(model_dir / decoder_file),
        frozen,
    )

    # The fused graph is ~1.9 GB; store weights as external data to stay clear of
    # the 2 GB protobuf limit. ONNX Runtime loads the sidecar automatically.
    data_file = out_path.name + ".data"
    # ONNX's external-data writer appends to the sidecar, so a stale/partial file
    # from a previous run would bloat (or corrupt) the output. Remove both first.
    out_path.unlink(missing_ok=True)
    (out_path.parent / data_file).unlink(missing_ok=True)
    print(f"  Saving -> {out_path} (+ {data_file})")
    onnx.save(
        merged,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=1024,
    )
    print("  ✓ Done. Single-input model: images -> "
          "pred_masks, pred_boxes, pred_logits, presence_logits")


if __name__ == "__main__":
    main()
