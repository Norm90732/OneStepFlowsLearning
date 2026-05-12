import torch
import torch.nn as nn
from jaxtyping import Float, jaxtyped
from beartype import beartype
from beartype.typing import Tuple
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, inputDim: int, embedDim: int, patchSize: int, imageSize: int):
        super().__init__()
        self.patchEmbed = nn.Conv2d(
            in_channels=inputDim,
            out_channels=embedDim,
            kernel_size=(patchSize, patchSize),
            stride=(patchSize, patchSize),
        )
        self.embedDim = embedDim
        self.sizePatched = int(imageSize // patchSize)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, latent: Float[torch.Tensor, "b c h w"]
    ) -> Float[torch.Tensor, "b seq EmbedDim"]:
        embedded = self.patchEmbed(latent)
        tokenEmbed = rearrange(
            embedded,
            "b embedDim heightPatched widthPatched -> b (heightPatched widthPatched) embedDim",
            embedDim=self.embedDim,
            heightPatched=self.sizePatched,
            widthPatched=self.sizePatched,
        )
        return tokenEmbed


class UnPatchEmbedding(nn.Module):
    def __init__(self, embedDim: int, inputDim: int, patchSize: int, imageSize: int):
        super().__init__()
        self.linearUnpatch = nn.Linear(
            in_features=embedDim,
            out_features=inputDim * patchSize * patchSize,
        )
        self.patchSize = patchSize
        self.inputDim = inputDim
        self.sizePatched = int(imageSize // patchSize)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, latent: Float[torch.Tensor, " b seq embedDim"]
    ) -> Float[torch.Tensor, "b c h w"]:
        outputLinear = self.linearUnpatch(latent)

        reshapedOuput = rearrange(
            outputLinear,
            "b (embeddedHeight embeddedWidth) (inputDim patchHeight patchWidth) -> b inputDim (embeddedHeight patchHeight) (embeddedWidth patchWidth)",
            embeddedHeight=self.sizePatched,
            embeddedWidth=self.sizePatched,
            inputDim=self.inputDim,
            patchHeight=self.patchSize,
            patchWidth=self.patchSize,
        )
        return reshapedOuput


class MultiConditionEmbed(nn.Module):
    def __init__(self, baseFreq: int, sinusoidalDim: int, embedDim: int):
        super().__init__()
        self.baseFreq = baseFreq
        self.sinusoidalDim = sinusoidalDim

        freqLog = (
            -torch.arange(0, self.sinusoidalDim // 2)
            * torch.log(torch.tensor(self.baseFreq))
            / self.sinusoidalDim
        )
        freq = torch.exp(freqLog)

        self.register_buffer("freq", freq)

        self.projectionMLP = nn.Sequential(
            nn.Linear(sinusoidalDim * 4, embedDim * 4),
            nn.SiLU(),
            nn.Linear(embedDim * 4, embedDim * 4),
        )

    @jaxtyped(typechecker=beartype)
    def _buildTable(
        self, inputTensor: Float[torch.Tensor, "b"]
    ) -> Float[torch.Tensor, "b sinusoidalDim"]:
        angle = torch.outer(inputTensor, self.freq)  # pyrefly:ignore
        embedding = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        print(embedding.shape)
        return embedding

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        h: Float[torch.Tensor, "b"],
        omega: Float[torch.Tensor, "b"],
        cfgTStart: Float[torch.Tensor, "b"],
        cfgTEnd: Float[torch.Tensor, "b"],
    ) -> Tuple[
        Float[torch.Tensor, "b embedDim"],
        Float[torch.Tensor, "b embedDim"],
        Float[torch.Tensor, "b embedDim"],
        Float[torch.Tensor, "b embedDim"],
    ]:

        hTEmbed = self._buildTable(h)
        omegaTEmbed = self._buildTable(omega)
        cfgTStartTEmbed = self._buildTable(cfgTStart)
        cfgTEndTEmbed = self._buildTable(cfgTEnd)

        concat = torch.cat(
            [
                hTEmbed,
                omegaTEmbed,
                cfgTStartTEmbed,
                cfgTEndTEmbed,
            ],
            dim=-1,
        )

        embedding = self.projectionMLP(concat)
        hEmbedding, omegaEmbedding, cfgTStartEmbedding, cfgTEndEmbedding = torch.chunk(
            embedding, chunks=4, dim=-1
        )

        return hEmbedding, omegaEmbedding, cfgTStartEmbedding, cfgTEndEmbedding


class SWIGLU(nn.Module):
    def __init__(self, embedDim: int, multiplier: int):
        super().__init__()
        self.upLinear = nn.Linear(embedDim, embedDim * multiplier * 2, bias=False)
        self.downLinear = nn.Linear(embedDim * multiplier, embedDim, bias=False)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, latent: Float[torch.Tensor, "b seqlen embedDim"]
    ) -> Float[torch.Tensor, "b seqlen embedDim"]:
        wV = self.upLinear(latent)
        w, v = torch.chunk(wV, chunks=2, dim=-1)
        toDown = nn.functional.silu(w) * v
        return self.downLinear(toDown)
