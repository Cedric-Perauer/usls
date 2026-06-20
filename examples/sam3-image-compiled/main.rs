use anyhow::Result;
use clap::Parser;
use usls::{
    models::Sam3ImageCompiled, Annotator, Config, DType, DataLoader, Device, Model, Source,
};

#[path = "../utils/mod.rs"]
mod utils;

/// SAM3 Image with a fixed text prompt pre-compiled into a single ONNX file.
///
/// The prompt is baked into the model weights, so the only input is the image —
/// no tokenizer or text encoder. Produce the single file with
/// `scripts/sam3-image/export_compiled.py --arch v2`.
#[derive(Parser)]
#[command(author, version, about = "SAM3 Image (prompt-compiled, single file)")]
struct Cli {
    /// Path to (or hub name of) the compiled single-file ONNX model.
    #[arg(long, default_value = "sam3-image-compiled.onnx")]
    model: String,

    /// The concept baked into the model (used as the detection label).
    #[arg(long, default_value = "object")]
    name: String,

    /// Source: image path, folder, or video.
    #[arg(long, default_value = "./assets/kids.jpg")]
    source: Source,

    /// Confidence threshold.
    #[arg(long, default_value_t = 0.5)]
    conf: f32,

    /// Device: cpu, cuda:0, mps, coreml, etc.
    #[arg(long, default_value = "cpu")]
    device: Device,

    /// Model dtype: fp32, fp16, etc.
    #[arg(long, default_value = "fp32")]
    dtype: DType,
}

fn main() -> Result<()> {
    utils::init_logging();
    let cli = Cli::parse();

    let config = Config::sam3_image_compiled()
        .with_model_file(&cli.model)
        .with_class_names_owned(vec![cli.name.clone()])
        .with_class_confs(&[cli.conf])
        .with_model_device(cli.device)
        .with_model_dtype(cli.dtype)
        .commit()?;

    let mut model = Sam3ImageCompiled::new(config)?;

    let annotator = Annotator::default()
        .with_mask_style(
            usls::MaskStyle::default()
                .with_visible(true)
                .with_draw_polygon_largest(true),
        )
        .with_polygon_style(usls::PolygonStyle::default().with_thickness(2));

    let dl = DataLoader::new(cli.source)?
        .with_batch(1)
        .with_progress_bar(true)
        .stream()?;

    for batch in dl {
        let ys = model.forward(&batch)?;
        tracing::info!("ys: {:?}", ys);
        for (img, y) in batch.iter().zip(ys.iter()) {
            let annotated = annotator.annotate(img, y)?;
            annotated.save(
                usls::Dir::Current
                    .base_dir_with_subs(&["runs/sam3-image-compiled"])?
                    .join(format!("{}.jpg", usls::timestamp(None))),
            )?;
        }
    }

    Ok(())
}
