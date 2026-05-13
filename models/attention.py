import torch
import torch.nn as nn
from jaxtyping import jaxtyped, Float
from beartype import beartype
from jvp_flash_attention.jvp_attention import JVPAttn
from models.blocks import SwiGlu
from models.embeddings import Conditioning2DRoPE
from einops import rearrange


# add gating mechanism in attention and swiglu paths
class InContextBlock(nn.Module):
    def __init__(self, embedDim: int, numHeads: int, SwiGluMultiplier: int):
        super().__init__()

        self.numHeads = numHeads
        self.headDim = embedDim // numHeads

        # attention
        self.rmsNorm1 = nn.RMSNorm(embedDim, eps=1e-6)
        self.qkvProj = nn.Linear(embedDim, embedDim * 3)
        self.qNorm = nn.RMSNorm(self.headDim, eps=1e-6)
        self.kNorm = nn.RMSNorm(self.headDim, eps=1e-6)
        self.outProjection = nn.Linear(embedDim, embedDim)
        # SwiGlu
        self.rmsNorm2 = nn.RMSNorm(embedDim, eps=1e-6)
        self.swiglu = SwiGlu(embedDim=embedDim, multiplier=SwiGluMultiplier)
        # learnableGate
        self.attnScale = nn.Parameter(torch.ones(embedDim) * 1e-4)
        self.mlpScale = nn.Parameter(torch.ones(embedDim) * 1e-4)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, latent: Float[torch.Tensor, "b seqlen embedDim"], rope: Conditioning2DRoPE
    ) -> Float[torch.Tensor, "b seqlen embedDim"]:
        residual1 = latent
        rmsNorm1 = self.rmsNorm1(latent)
        qkv = self.qkvProj(rmsNorm1)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        q = rearrange(
            q,
            "b seqlen (numHeads headDim) -> b seqlen numHeads headDim",
            numHeads=self.numHeads,
            headDim=self.headDim,
        )
        k = rearrange(
            k,
            "b seqlen (numHeads headDim) -> b seqlen numHeads headDim",
            numHeads=self.numHeads,
            headDim=self.headDim,
        )
        v = rearrange(
            v,
            "b seqlen (numHeads headDim) -> b seqlen numHeads headDim",
            numHeads=self.numHeads,
            headDim=self.headDim,
        )

        qNorm = self.qNorm(q)
        kNorm = self.kNorm(k)

        qRotated, kRotated = rope(qNorm, kNorm)

        # rearrange for jvp kernel
        qJVP = rearrange(
            qRotated,
            "b seqlen numHeads headDim -> b numHeads seqlen headDim",
            numHeads=self.numHeads,
            headDim=self.headDim,
        )
        kJVP = rearrange(
            kRotated,
            "b seqlen numHeads headDim -> b numHeads seqlen headDim",
            numHeads=self.numHeads,
            headDim=self.headDim,
        )
        vJVP = rearrange(
            v,
            "b seqlen numHeads headDim -> b numHeads seqlen headDim",
            numHeads=self.numHeads,
            headDim=self.headDim,
        )

        attention = JVPAttn.fwd_dual(
            q=qJVP.contiguous(),
            k=kJVP.contiguous(),
            v=vJVP.contiguous(),
            attn_mask=None,
        )

        attentionReshape = rearrange(
            attention,
            "b numHeads seqlen headDim -> b seqlen (numHeads headDim)",
            numHeads=self.numHeads,
            headDim=self.headDim,
        )

        attentionProjection = self.outProjection(attentionReshape)

        attentionResidual = residual1 + self.attnScale * attentionProjection

        residual2 = attentionResidual

        rmsNorm2 = self.rmsNorm2(attentionResidual)

        swigluOutput = self.swiglu(rmsNorm2)

        return residual2 + self.mlpScale * swigluOutput
