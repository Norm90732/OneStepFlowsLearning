from omegaconf import DictConfig
import ray
import torch
from models.pretrainedComponents import FluxVAEEncoder
import numpy as np
import torchvision.transforms.v2 as v2
import torchvision.io as visionIO
from pathlib import Path
import json
from dataset.imagenetUtils import getImagePaths
from ray.data import DataContext, SaveMode


def makeWdsKey(path: str, classidx: int) -> str:
    p = Path(path)
    return f"{classidx:05d}_{p.parent.name}_{p.stem}"


class VaeCompressor:
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = FluxVAEEncoder().to(self.device, dtype=torch.bfloat16).eval()

        self.transform = v2.Compose(
            [
                v2.Resize((256, 256)),
                v2.ToImage(),
                v2.ToDtype(torch.bfloat16, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ).to(self.device)

    def _decodeImages(self, paths, labels, encoded):
        try:
            imgs = visionIO.decode_jpeg(
                encoded,
                mode=visionIO.ImageReadMode.RGB,
                device=self.device,
            )
            return imgs, paths, labels

        except RuntimeError as batchError:
            goodImgs = []
            goodPaths = []
            goodLabels = []
            badPaths = []

            for path, label, encodedImage in zip(paths, labels, encoded):
                try:
                    img = visionIO.decode_jpeg(
                        encodedImage,
                        mode=visionIO.ImageReadMode.RGB,
                        device=self.device,
                    )
                    goodImgs.append(img)
                    goodPaths.append(path)
                    goodLabels.append(int(label))

                except RuntimeError:
                    badPaths.append(path)
            return goodImgs, goodPaths, np.asarray(goodLabels, dtype=np.int64)

    def __call__(self, batch: dict) -> dict:
        paths = [str(p) for p in batch["imagepath"]]
        labels = np.asarray(batch["classidx"], dtype=np.int64)

        encoded = [visionIO.read_file(p) for p in paths]

        imgs, paths, labels = self._decodeImages(
            paths=paths,
            labels=labels,
            encoded=encoded,
        )

        transformedImages = torch.stack(
            [self.transform(img) for img in imgs],
            dim=0,
        )

        latents = self.model(transformedImages)
        latents = latents.detach().to(torch.float16).cpu().numpy()

        keys = [
            makeWdsKey(path, int(classidx)) for path, classidx in zip(paths, labels)
        ]

        return {
            "__key__": np.asarray(keys, dtype=object),
            "latent.npy": latents,
            "classidx.cls": labels,
            "imagepath.txt": np.asarray(paths, dtype=object),
        }


if __name__ == "__main__":
    from omegaconf import OmegaConf

    ctx = DataContext.get_current()
    ctx.enable_progress_bars = True
    ctx.use_ray_tqdm = True
    ctx.enable_operator_progress_bars = True
    ray.init()
    resources = ray.available_resources()
    numGpus = resources.get("GPU", 0)
    cfg = OmegaConf.load("configs/config.yaml")
    cfg.dataset = OmegaConf.load("configs/dataset/imagenet21k.yaml")

    allData = getImagePaths(cfg, "train")  # pyrefly:ignore
    print(f"{allData[100000]}")
    print(f"Total Images {len(allData)}")

    ds = ray.data.from_items(allData)
    latentDs = ds.map_batches(
        VaeCompressor,
        batch_size=1536,
        batch_format="numpy",
        num_gpus=1,
        compute=ray.data.ActorPoolStrategy(
            size=int(numGpus),
            max_tasks_in_flight_per_actor=4,
        ),
        udf_modifying_row_count=True,
    )

    trainPath = cfg.dataset.trainTars
    latentDs.write_webdataset(
        trainPath,
        encoder=True,
        min_rows_per_file=30_000,
        mode=SaveMode.OVERWRITE,
    )
