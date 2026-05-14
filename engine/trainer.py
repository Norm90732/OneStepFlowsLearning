import os
os.environ["JAXTYPING_DISABLE"] = "1"
import torch 
from torch_ema import ExponentialMovingAverage

from models.dit import ImfDIT
from models.flows import ImprovedMeanFlowTrainer
from engine.trainerUtils import createOptimizerAndScheduler,mapBatches, resumeCheckpoint,saveCheckpoint,ValidationMonitoring

#ray 
import ray 
import ray.train 
from ray import train 
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, RunConfig, Checkpoint, CheckpointConfig
from ray.data import DataContext

import wandb
from ray.air.integrations.wandb import setup_wandb
import hydra 
from omegaconf import DictConfig, OmegaConf


def trainingFunction(config):
    cfg = config["cfg"]
    
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    
    wandbConfig = OmegaConf.to_container(cfg, resolve=True)
    
    context = ray.train.get_context()
    localRank = context.get_local_rank()
    worldRank = context.get_world_rank()

    device = torch.device(f"cuda:{localRank}")
    
    wandbRun = setup_wandb(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity", None),
        name=cfg.wandb.get("name", None),
        group=cfg.wandb.get("group", None),
        config=wandbConfig,  #pyrefly:ignore 
        rank_zero_only=True,
    )
    
    model = ImfDIT(cfg=cfg).to(device)
    
    model = train.torch.prepare_model(
        model,parallel_strategy_kwargs={"find_unused_parameters": False}
    )
    
    ema = ExponentialMovingAverage(
    model.parameters(),
    decay=cfg.models.training.ema.emaDecay,
    )
    ema.to(device)

    
    optimizer, scheduler = createOptimizerAndScheduler(model,cfg=cfg)
    
    startEpoch = resumeCheckpoint(model=model,optimizer=optimizer,scheduler=scheduler,ema=ema)
    
    
    trainDataShard = train.get_dataset_shard("train")
    totalEpochs = cfg.models.training.scheduler.warmupEpochs +cfg.models.training.scheduler.trainingEpochs
    batchSize = cfg.models.training.batchSize
    prefetchBatches = cfg.models.training.prefetchBatches
    trainer = ImprovedMeanFlowTrainer(cfg=cfg,device=device)
    
    validator = ValidationMonitoring(device=device)
    validateEvery = cfg.models.training.validate.validateEvery
    
    
    for epoch in range(startEpoch,totalEpochs):
        model.train()
        dataloader = trainDataShard.iter_torch_batches(
            batch_size=batchSize,
            local_shuffle_buffer_size=512,
            prefetch_batches=prefetchBatches,
            device=device,
            pin_memory=True,
            dtypes={
                "latent": torch.bfloat16, 
                "label": torch.long
            }
        )
        runningLoss = 0.0
        numSteps = 0
        metrics = {}
        
        for batch in dataloader:
            inputLatents = batch["latent"].to(device,non_blocking=True)
            labels = batch["label"].to(device,non_blocking=True)
            
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                forwardModel = model.module if hasattr(model, "module") else model
                loss,metrics = trainer.trainer(
                    model=forwardModel, #pyrefly:ignore 
                    latent=inputLatents,
                    classIdx=labels
                )
            
            loss.backward()
            optimizer.step()
            ema.update()
            runningLoss += loss.detach().item()
            numSteps += 1
            
        
        scheduler.step()
        metrics["lossEpoch"] = runningLoss / max(numSteps, 1)
        metrics["lr"] = scheduler.get_last_lr()[0]

        if worldRank == 0:
            if epoch % validateEvery == 0:
                validator.validate(
                    model=model,
                    ema=ema,
                    wandbRun=wandbRun,
                    epoch=epoch,
                )
        
        cleanMetrics = saveCheckpoint(
            model=model,
            metrics=metrics,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            worldRank=worldRank
        )
        wandbRun.log(cleanMetrics, step=epoch) #pyrefly:ignore 
        
    wandbRun.finish() #pyrefly:ignore 
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg:DictConfig):    
    ray.init(ignore_reinit_error=True)
    
    data_ctx = DataContext.get_current()
    data_ctx.enable_rich_progress_bars = True
    data_ctx.use_ray_tqdm = False
    data_ctx.enable_operator_progress_bars = False
    
    resolvedDict = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(resolvedDict) #pyrefly:ignore 
    #dataloaderSet up 
    datasetPath = cfg.dataset.trainTars
    trainDataset = ray.data.read_webdataset(datasetPath,override_num_blocks=512,)
    trainDataset = trainDataset.map_batches(mapBatches)
        
    trainConfig = {
        "cfg": resolvedDict,
    }
    
    trainer = TorchTrainer(
        train_loop_per_worker=trainingFunction,
        train_loop_config=trainConfig,
        datasets={"train": trainDataset},
        torch_config=TorchConfig(backend="nccl"),
        scaling_config=ScalingConfig( #pyrefly:ignore 
            num_workers=cfg.ray.totalWorkers,  #totalNum Workers 
            use_gpu=True,
            resources_per_worker={ 
                "CPU": cfg.ray.perWorkerCPU, #per worker cpu 
                "GPU": cfg.ray.perWorkerGPU, #perworkergpu 
            },
        ),
        run_config=RunConfig( #pyrefly:ignore 
            name=cfg.run.name,
            storage_path=cfg.run.storagePath, 
            checkpoint_config=CheckpointConfig( #pyrefly:ignore 
                num_to_keep=cfg.checkpoint.numToKeep,
                checkpoint_score_attribute="lossEpoch",
                checkpoint_score_order="min",
            ),
        ),
    )

    result = trainer.fit()
    
if __name__ == "__main__":
    main()