use aksr::Builder;
use anyhow::{anyhow, Result};
use ndarray::{s, Array1};
use rayon::prelude::*;

use crate::{
    Config, DynConf, Engine, Engines, FromConfig, Hbb, Image, ImageProcessor, Mask, Model, Module,
    Ops, Xs, Y,
};

/// SAM3 Image with a fixed text prompt **pre-compiled** into a single ONNX model.
///
/// Unlike [`crate::Sam3Image`] (which loads a vision encoder, a text encoder and a
/// decoder and accepts a [`crate::Sam3Prompt`] at runtime), the prompt here is
/// frozen into the model weights: the vision encoder and decoder are fused and the
/// text encoder output is baked in as a constant. The only runtime input is the
/// image — no tokenizer, no text encoder.
///
/// See `scripts/sam3-image/export_compiled.py` (`--arch v2`) for how the single
/// file is produced.
///
/// The baked prompt is a batch-1 constant, so images are run one at a time.
#[derive(Debug, Builder)]
pub struct Sam3ImageCompiled {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    /// The concept the model was compiled for (used as the detection label).
    pub name: String,
    pub conf: DynConf,
    pub processor: ImageProcessor,
    pub spec: String,
}

impl Model for Sam3ImageCompiled {
    type Input<'a> = &'a [Image];

    fn batch(&self) -> usize {
        self.batch
    }

    fn spec(&self) -> &str {
        &self.spec
    }

    fn build(mut config: Config) -> Result<(Self, Engines)> {
        let engine = Engine::from_config(config.take_module(&Module::Model)?)?;
        let (batch, height, width) = (
            engine.batch().opt(),
            engine.try_height().unwrap_or(&1008.into()).opt(),
            engine.try_width().unwrap_or(&1008.into()).opt(),
        );
        let spec = engine.spec().to_string();
        let conf = DynConf::new_or_default(config.class_confs(), 1);
        let name = config
            .inference
            .class_names
            .first()
            .cloned()
            .unwrap_or_else(|| "object".to_string());
        let processor = ImageProcessor::from_config(config.image_processor)?
            .with_image_width(width as _)
            .with_image_height(height as _);

        let model = Self {
            batch,
            height,
            width,
            name,
            conf,
            processor,
            spec: if spec.is_empty() {
                "sam3-image-compiled".to_string()
            } else {
                spec
            },
        };

        Ok((model, Engines::from(engine)))
    }

    fn run(&mut self, engines: &mut Engines, images: Self::Input<'_>) -> Result<Vec<Y>> {
        if images.is_empty() {
            return Ok(vec![]);
        }

        // The baked prompt is a batch-1 constant, so decode one image at a time.
        let mut ys = Vec::with_capacity(images.len());
        for image in images {
            let slice = std::slice::from_ref(image);
            let x = crate::perf!(
                "Sam3ImageCompiled::preprocess",
                self.processor.process(slice)?
            );
            let outputs = crate::perf!(
                "Sam3ImageCompiled::inference",
                engines.run(&Module::Model, &x)?
            );
            let (src_h, src_w) = {
                let info = &self.processor.images_transform_info()[0];
                (info.height_src as usize, info.width_src as usize)
            };
            let y = crate::perf!(
                "Sam3ImageCompiled::postprocess",
                self.postprocess(&outputs, src_h, src_w)?
            );
            ys.push(y);
        }

        Ok(ys)
    }
}

impl Sam3ImageCompiled {
    /// Decode the fused model's outputs for a single (batch-1) image.
    ///
    /// Mirrors `Sam3Image::postprocess`: score = sigmoid(logit) * sigmoid(presence),
    /// keep above threshold, then resize each kept mask to the source resolution and
    /// scale the normalized boxes to pixels.
    fn postprocess(&self, outputs: &Xs, image_height: usize, image_width: usize) -> Result<Y> {
        let masks = outputs
            .get::<f32>(0)
            .ok_or_else(|| anyhow!("Failed to get masks"))?;
        let boxes = outputs
            .get::<f32>(1)
            .ok_or_else(|| anyhow!("Failed to get boxes"))?;
        let logits = outputs
            .get::<f32>(2)
            .ok_or_else(|| anyhow!("Failed to get logits"))?;
        let presence = outputs
            .get::<f32>(3)
            .ok_or_else(|| anyhow!("Failed to get presence"))?;

        let presence_score = 1.0 / (1.0 + (-presence.0[[0, 0]]).exp());
        let scores: Array1<f32> = logits
            .0
            .slice(s![0, ..])
            .mapv(|x| 1.0 / (1.0 + (-x).exp()) * presence_score);
        let valid: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s >= self.conf[0])
            .map(|(i, _)| i)
            .collect();
        if valid.is_empty() {
            return Ok(Y::default());
        }

        let res: Vec<_> = valid
            .into_par_iter()
            .filter_map(|idx| {
                let mask_view = masks.0.slice(s![0, idx, .., ..]);
                let (mh, mw) = mask_view.dim();

                let src = match mask_view.as_slice_memory_order() {
                    Some(s) => std::borrow::Cow::Borrowed(s),
                    None => {
                        std::borrow::Cow::Owned(mask_view.to_owned().into_raw_vec_and_offset().0)
                    }
                };

                let luma = Ops::interpolate_1d_u8(
                    src.as_ref(),
                    mw as _,
                    mh as _,
                    image_width as _,
                    image_height as _,
                    false,
                )
                .ok()?;

                let mask = Mask::new(&luma, image_width as u32, image_height as u32)
                    .ok()?
                    .with_id(0)
                    .with_name(&self.name)
                    .with_confidence(scores[idx]);

                let hbb = Hbb::default()
                    .with_xyxy(
                        boxes.0[[0, idx, 0]] * image_width as f32,
                        boxes.0[[0, idx, 1]] * image_height as f32,
                        boxes.0[[0, idx, 2]] * image_width as f32,
                        boxes.0[[0, idx, 3]] * image_height as f32,
                    )
                    .with_confidence(scores[idx])
                    .with_id(0)
                    .with_name(&self.name);

                Some((mask, hbb))
            })
            .collect();

        let (y_masks, y_hbbs): (Vec<_>, Vec<_>) = res.into_iter().unzip();
        Ok(Y::default().with_masks(&y_masks).with_hbbs(&y_hbbs))
    }
}
