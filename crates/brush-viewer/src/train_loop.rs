use std::io::Read;

use anyhow::anyhow;
use async_std::{
    channel::{Sender, TrySendError},
    stream::StreamExt,
};
use brush_dataset::{scene_batch::SceneLoader, Dataset, LoadDatasetArgs, ZipData};
use brush_render::{gaussian_splats::RandomSplatsConfig, PrimaryBackend};
use brush_train::train::{SplatTrainer, TrainConfig};
use burn::{
    backend::Autodiff, lr_scheduler::exponential::ExponentialLrSchedulerConfig,
    module::AutodiffModule,
};
use burn_wgpu::WgpuDevice;
use web_time::Instant;
use zip::ZipArchive;

use crate::{splat_import, viewer::ViewerMessage};

pub(crate) struct SharedTrainState {
    pub paused: bool,
}

pub(crate) async fn train_loop(
    zip_data: ZipData,
    device: WgpuDevice,
    sender: Sender<ViewerMessage>,
    load_data_args: LoadDatasetArgs,
) -> anyhow::Result<()> {
    let total_steps = 30000;
    let mut archive = ZipArchive::new(zip_data.open_for_read())?;

    // Load initial splats if included
    let mut initial_splats = None;

    let data = archive
        .by_name("init.ply")
        .map(|f| f.bytes().collect::<std::io::Result<Vec<u8>>>());

    if let Ok(Ok(data)) = data {
        let splat_stream =
            splat_import::load_splat_from_ply::<Autodiff<PrimaryBackend>>(&data, device.clone());

        let mut splat_stream = std::pin::pin!(splat_stream);
        while let Some(splats) = splat_stream.next().await {
            let splats = splats?;
            let msg = ViewerMessage::Splats {
                splats: splats.valid(),
            };
            sender
                .send(msg)
                .await
                .map_err(|e| anyhow!("Failed to send message: {}", e))?;
            initial_splats = Some(splats);
        }
    }

    let data_stream =
        brush_dataset::read_dataset::<Autodiff<PrimaryBackend>>(archive.clone(), &load_data_args)?;

    let mut dataset = Dataset::empty();
    let mut data_stream = std::pin::pin!(data_stream);
    while let Some(d) = data_stream.next().await {
        dataset = d?;

        sender
            .send(ViewerMessage::Dataset {
                data: dataset.clone(),
            })
            .await
            .map_err(|e| anyhow!("Failed to send message: {}", e))?;
    }
    sender
        .send(ViewerMessage::DoneLoading { training: true })
        .await
        .map_err(|e| anyhow!("Failed to send message: {}", e))?;

    let mut splats = if let Some(splats) = initial_splats {
        splats
    } else {
        // By default, spawn the splats in bounds.
        let bounds = dataset.train.bounds(0.0);
        let bounds_extent = bounds.extent.length();
        let adjusted_bounds = dataset.train.bounds(bounds_extent);
        let config = RandomSplatsConfig::new(adjusted_bounds);
        config.init(&device)
    };

    let train_scene = dataset.train.clone();

    let scene_extent = train_scene.bounds(0.0).extent.length() as f64;
    let lr_max = 1.6e-4 * scene_extent;
    let decay = 1e-2f64.powf(1.0 / total_steps as f64);
    let config = TrainConfig::new(ExponentialLrSchedulerConfig::new(lr_max, decay))
        .with_total_steps(total_steps)
        .with_densify_size_thresh(0.01 * scene_extent as f32);

    let mut dataloader = SceneLoader::new(&train_scene, 1, &device);
    let mut trainer = SplatTrainer::new(splats.num_splats(), &config, &splats);

    loop {
        // TODO: Restore the pause button but better.
        // if shared_state.read().paused {
        //     #[cfg(not(target_arch = "wasm32"))]
        //     std::thread::yield_now();
        //     #[cfg(target_arch = "wasm32")]
        //     gloo_timers::future::TimeoutFuture::new(0).await;
        //     continue;
        // }
        let batch = {
            // let _span = trace_span!("Get batch").entered();
            dataloader.next_batch().await
        };
        let (new_splats, stats) = trainer.step(batch, train_scene.background, splats).await?;
        splats = new_splats;

        if trainer.iter % 5 == 0 {
            if let Err(TrySendError::Closed(_)) = sender.try_send(ViewerMessage::Splats {
                splats: splats.valid(),
            }) {
                break; // channel closed, bail.
            }

            if let Err(TrySendError::Closed(_)) = sender.try_send(ViewerMessage::TrainStep {
                splats: splats.clone(),
                stats,
                iter: trainer.iter,
                timestamp: Instant::now(),
            }) {
                break; // channel closed, bail.
            }
        }
    }

    Ok(())
}
