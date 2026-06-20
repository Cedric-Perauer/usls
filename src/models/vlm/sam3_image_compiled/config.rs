use crate::{Config, Task};

///
/// > # SAM3 Image (prompt-compiled, single file)
/// >
/// > A variant of `sam3-image` where a fixed text prompt is baked into the model
/// > weights and the vision encoder + decoder are fused into one ONNX file. The
/// > only runtime input is the image; no tokenizer or text encoder is needed.
/// >
/// > Produce the single file with `scripts/sam3-image/export_compiled.py --arch v2`.
/// > Set the concept label (used for detections) via `.with_class_names(&["person"])`.
/// >
/// > # Notes
/// >
/// > - One ONNX engine ([`crate::Module::Model`]): `images [B,3,1008,1008]` ->
/// >   `pred_masks, pred_boxes, pred_logits, presence_logits`.
/// > - The baked prompt is a batch-1 constant, so images are decoded one at a time.
///
impl Config {
    /// SAM3 Image with a prompt pre-compiled into a single ONNX model.
    pub fn sam3_image_compiled() -> Self {
        Self::sam3()
            .with_task(Task::Sam3Image)
            // single fused engine: images [B, 3, 1008, 1008] -> 4 decoder outputs
            .with_model_batch_size_min_opt_max(1, 1, 1)
            .with_model_file("sam3-image-compiled.onnx")
    }
}
