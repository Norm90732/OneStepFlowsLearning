import tempfile
import torch 
from omegaconf import DictConfig 
import io 
import numpy as np 
import tempfile
from ray.train import get_checkpoint, report, Checkpoint
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import os 
from models.pretrainedComponents import FluxVAEDecoder
from einops import rearrange
import wandb 

def createOptimizerAndScheduler(model,cfg:DictConfig):
    cfgTraining = cfg.models.training
    cfgOptimizer = cfgTraining.optimizer
    
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr= cfgOptimizer.lr,
        weight_decay= cfgOptimizer.weightDecay,
        betas = tuple(cfgOptimizer.betas)
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=cfgTraining.scheduler.warmupEpochs
    )
    
    return optimizer, scheduler

def mapBatches(batch):
    latentCol = batch["latent.npy"]

    if isinstance(latentCol, np.ndarray) and latentCol.dtype != object:
        latents = latentCol.astype(np.float32)
    else:
        latents = np.stack(
            [np.asarray(x, dtype=np.float32) for x in latentCol],
            axis=0,
        )

    labels = np.asarray(batch["classidx.cls"], dtype=np.int64) + 1 #@null embedding table is classidx =0, so predoing this 

    return {
        "latent": latents,
        "label": labels,
    }

def resumeCheckpoint(model, optimizer, scheduler, ema):
    checkpoint = get_checkpoint()
    startEpoch = 0

    if checkpoint:
        with checkpoint.as_directory() as checkpointDir:
            checkpointDict = torch.load(
                os.path.join(checkpointDir, "checkpoint.pt"),
                map_location="cpu",
            )
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpointDict["modelState"])
            else:
                model.load_state_dict(checkpointDict["modelState"])
            optimizer.load_state_dict(checkpointDict["optimizerState"])
            ema.load_state_dict(checkpointDict["emaState"])
            scheduler.load_state_dict(checkpointDict["schedulerState"])

            startEpoch = checkpointDict["epoch"] + 1

    return startEpoch


def saveCheckpoint(model, metrics, epoch, optimizer, scheduler, ema, worldRank):
    cleanMetrics = {}

    for key, value in metrics.items():
        if torch.is_tensor(value):
            cleanMetrics[key] = value.detach().cpu().item()
        else:
            cleanMetrics[key] = value

    cleanMetrics["epoch"] = epoch

    checkpoint = None

    if worldRank == 0:
        with tempfile.TemporaryDirectory() as tempDir:
            with ema.average_parameters():
                modelState = model.state_dict()
                consume_prefix_in_state_dict_if_present(modelState, "module.")

            emaState = ema.state_dict()
            consume_prefix_in_state_dict_if_present(emaState, "module.")

            stateSave = {
                "modelState": modelState,
                "optimizerState": optimizer.state_dict(),
                "emaState": emaState,
                "schedulerState": scheduler.state_dict(),
                "epoch": epoch,
            }

            torch.save(stateSave, os.path.join(tempDir, "checkpoint.pt"))

            checkpoint = Checkpoint.from_directory(tempDir)

            report(
                metrics=cleanMetrics,
                checkpoint=checkpoint,
            )
    else:
        report(
            metrics=cleanMetrics,
            checkpoint=None,
        )

    return cleanMetrics

#quick validation set up for wandb 
class ValidationMonitoring():
    def __init__(self,device):
        self.device = device
        self.decoder = FluxVAEDecoder().to(device)
        
        labels = [240,251,971,980,976,973,986,987,739,672]
        self.batchSize = len(labels)   
        self.labels = torch.tensor(labels, dtype=torch.long,device=self.device)
        
        self.z = torch.randn(self.batchSize,32,32,32,device=self.device)
    def _inferenceLatents(self,model,ema): # -> Float[torch.Tensor, "b c h w"]
        
        model.eval()
        forwardModel = model.module if hasattr(model, "module") else model
        with torch.inference_mode():
            with ema.average_parameters():
                z= self.z.clone()
                h = torch.ones_like(self.labels,device=self.device)
                omegaHat = torch.ones_like(self.labels,device=self.device) * 0.5 # 2cfg 
                tStart = torch.zeros_like(self.labels,device=self.device)
                tEnd = torch.ones_like(self.labels,device=self.device)
                
                u = forwardModel.forwardU(
                    latent=z,
                    h=h,
                    omega=omegaHat,
                    cfgTStart=tStart,
                    cfgTEnd=tEnd,
                    classIdx=self.labels
                )
                
                hReshape = rearrange(h, "b -> b 1 1 1")
                z0 = z - hReshape*u
                
                return z0 
                

    def _toPixelSpace(self,latents):
        with torch.inference_mode():
            images = self.decoder(latents)
            
        images = images.detach().float().cpu()
        images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        return images  
    
    def _imagesToWandb(self, images):
        wandbImages = [
            wandb.Image(image.permute(1, 2, 0).numpy())
            for image in images
        ]

        return wandbImages
    
    
    def validate(self, model, ema, wandbRun, epoch):
        latents = self._inferenceLatents(
            model=model,
            ema=ema,
        )

        images = self._toPixelSpace(
            latents=latents,
        )

        wandbImages = self._imagesToWandb(
            images=images,
        )

        wandbRun.log(
            {
                "validation/images": wandbImages,
            },
            step=epoch,
        )
    