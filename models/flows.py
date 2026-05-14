from beartype import beartype
import torch
from omegaconf import DictConfig
from torch.func import jvp
from jaxtyping import jaxtyped, Float, Int, Bool
from beartype.typing import Tuple
from models.dit import ImfDIT
from einops import rearrange


class ImprovedMeanFlowTrainer:
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.device = device
        cfgFlow = cfg.models.imFlow
        self.rTMean = cfgFlow.rMinusTDistribution.mean
        self.rTStd = cfgFlow.rMinusTDistribution.std
        self.rTEqual = cfgFlow.rMinusTDistribution.rTEqual

        self.omegaMin = cfgFlow.omegaDistribution.omegaMin
        self.omegaMax = cfgFlow.omegaDistribution.omegaMax
        self.beta = cfgFlow.omegaDistribution.beta

        self.classDrop = cfgFlow.classDrop

        self.lossEps = cfgFlow.lossEps
        self.lossPower = cfgFlow.lossPower

    @jaxtyped(typechecker=beartype)
    def _sampleLogitNormal(self, batch: int) -> Float[torch.Tensor, "b"]:
        return torch.sigmoid(
            self.rTMean + self.rTStd * torch.randn(batch, device=self.device)
        )  # pyrefly:ignore

    @jaxtyped(typechecker=beartype)
    def _sampleH(
        self, batch: int
    ) -> Tuple[
        Float[torch.Tensor, "b"],
        Float[torch.Tensor, "b"],
        Float[torch.Tensor, "b"],
        Bool[torch.Tensor, "b"],
    ]:
        a = self._sampleLogitNormal(batch)
        b = self._sampleLogitNormal(batch)
        # 0<= r <= t <= 1
        t = torch.maximum(a, b)
        r = torch.minimum(a, b)
        rEqualTMask = torch.rand(batch, device=self.device) < self.rTEqual
        r = torch.where(rEqualTMask, t, r)
        h = t - r
        return t, r, h, rEqualTMask

    @jaxtyped(typechecker=beartype)
    def _sampleOmega(self, batch: int) -> Float[torch.Tensor, "b"]:
        u = torch.rand(batch, device=self.device)
        if self.beta == 1:
            omega = self.omegaMin * (self.omegaMax / self.omegaMin) ** u
        else:
            omegaMinBeta = self.omegaMin ** (1 - self.beta)
            omegaMaxBeta = self.omegaMax ** (1 - self.beta)
            exponential = 1 / (1 - self.beta)
            omega = (omegaMinBeta + u * (omegaMaxBeta - omegaMinBeta)) ** exponential
        return omega

    @jaxtyped(typechecker=beartype)
    def _sampleHOmegaTMinMax(
        self, batch: int
    ) -> Tuple[
        Float[torch.Tensor, "b"],
        Float[torch.Tensor, "b"],
        Float[torch.Tensor, "b"],
        Float[torch.Tensor, "b"],
        Float[torch.Tensor, "b"],
        Float[torch.Tensor, "b"],
        Float[torch.Tensor, "b"],
        Bool[torch.Tensor, "b"],
    ]:
        t, r, h, fmMask = self._sampleH(batch=batch)

        omegaRaw = self._sampleOmega(batch=batch)

        tMin = 0.5 * torch.rand(batch, device=self.device)
        tMax = 0.5 + 0.5 * torch.rand(batch, device=self.device)
        tMin = torch.where(fmMask, torch.zeros_like(tMin), tMin)
        tMax = torch.where(fmMask, torch.ones_like(tMax), tMax)

        inside = (t >= tMin) & (t <= tMax)
        omegaEff = torch.where(inside, omegaRaw, torch.ones_like(omegaRaw))

        omegaHatRaw = 1.0 - 1.0 / omegaRaw
        omegaHatEff = 1.0 - 1.0 / omegaEff

        return t, r, h, omegaHatRaw, omegaHatEff, tMin, tMax, fmMask

        # class dropout into embedding table

    def _adaptiveWeightLoss(self, loss):
        weight = (loss + self.lossEps).pow(self.lossPower)

        return loss / weight.detach()

    @jaxtyped(typechecker=beartype)
    def trainer(
        self,
        model: ImfDIT,
        latent: Float[torch.Tensor, "b c h w"],
        classIdx: Int[torch.Tensor, "b"],
    ):
        
        batch = classIdx.shape[0]
        t, r, h, omegaHatRaw, omegaHatEff, tMin, tMax, fmMask = (
            self._sampleHOmegaTMinMax(batch=batch)
        )

        e = torch.randn_like(latent, device=self.device)
        tReshape = rearrange(t, "b -> b 1 1 1")

        z = (1.0 - tReshape) * latent + tReshape * e
        VTarget = e - latent

        nullClasses = torch.zeros_like(classIdx)

        hZero = torch.zeros_like(h)
        omegaHatZero = torch.zeros_like(omegaHatRaw)

        # instant v predict
        tMinV = torch.zeros_like(tMin)
        tMaxV = torch.ones_like(tMax)

        vC = model.forwardV(
            latent=z,
            h=hZero,
            omega=omegaHatEff,
            cfgTStart=tMinV,
            cfgTEnd=tMaxV,
            classIdx=classIdx,
        )

        vU = model.forwardV(
            latent=z,
            h=hZero,
            omega=omegaHatZero,
            cfgTStart=tMinV,
            cfgTEnd=tMaxV,
            classIdx=nullClasses,
        )

        # v guided  for target
        omegaHatEffView = rearrange(omegaHatEff, "b -> b 1 1 1")
        vG = VTarget + omegaHatEffView * (vC - vU)

        dropMask = torch.rand(batch, device=self.device) < self.classDrop
        clsCondition = torch.where(
            dropMask,
            nullClasses,
            classIdx,
        )
        vG = torch.where(
            rearrange(dropMask, "b -> b 1 1 1"),
            VTarget,
            vG,
        )

        def uPredictionFunction(
            z: torch.Tensor,
            h: torch.Tensor,
            omegaHat: torch.Tensor,
        ) -> torch.Tensor:
            return model.forwardU(
                latent=z,
                h=h,
                omega=omegaHat,
                cfgTStart=tMin,
                cfgTEnd=tMax,
                classIdx=clsCondition,
            )

        u, dudt = jvp(
            func=uPredictionFunction,
            primals=(z, h, omegaHatRaw),
            tangents=(
                vC.detach(),
                torch.ones_like(h),
                torch.zeros_like(omegaHatRaw),
            ),
        )

        hExpanded = rearrange(h, "b -> b 1 1 1")

        V = u + hExpanded * dudt.detach()

        vG = vG.detach()

        vAux = model.forwardV(
            latent=z,
            h=h,
            omega=omegaHatRaw,
            cfgTStart=tMin,
            cfgTEnd=tMax,
            classIdx=clsCondition,
        )

        lossURaw = ((V - vG) ** 2).sum(dim=(1, 2, 3))
        lossVRaw = ((vAux - vG) ** 2).sum(dim=(1, 2, 3))

        lossU = self._adaptiveWeightLoss(lossURaw)
        lossV = self._adaptiveWeightLoss(lossVRaw)

        loss = (lossU + lossV).mean() #adaptive weighting 

        logDict = {
        "loss": loss.detach(),
        "lossUMain": lossU.mean().detach(),
        "lossVAux": lossV.mean().detach(),
        "lossURaw": lossURaw.mean().detach(),
        "lossVRaw": lossVRaw.mean().detach(),
        }

        return loss, logDict