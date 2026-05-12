# IMF paper uses 1D RoPE, im using 2D multi modal rope (2D Rope on images, origin on conditioning)
import torch
import torch.nn as nn
from jaxtyping import jaxtyped, Float
from beartype import beartype
from beartype.typing import Tuple
from einops import rearrange


class Conditioning2DRoPE(nn.Module):
    def __init__(
        self,
        baseFrequency: int,
        numHeads: int,
        embedDim: int,
        conditioningTokensNum: int,
        imageSize: int,
        patchSize: int,
    ):
        super().__init__()
        self.headDim = int(embedDim // numHeads)
        self.headDimInTwo = self.headDim // 2
        self.headDimInFour = self.headDim // 4
        self.conditioningTokensNum = conditioningTokensNum
        self.baseFrequency = baseFrequency
        self.sizePatched = int(imageSize // patchSize)

        angleTable = self._buildAngles()

        cos = torch.cos(angleTable)
        sin = torch.sin(angleTable)

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def _buildAngles(self) -> Float[torch.Tensor, "seqlen headDim"]:
        freqs = 1 / self.baseFrequency ** (
            torch.arange(0, self.headDimInTwo, 2) / self.headDimInTwo
        )
        xCondition = torch.zeros(self.conditioningTokensNum)
        yCondition = torch.zeros(self.conditioningTokensNum)

        yGrid, xGrid = torch.meshgrid(
            torch.arange(1, self.sizePatched + 1, 1),
            torch.arange(1, self.sizePatched + 1, 1),
            indexing="ij",
        )
        # the first 20 tokens should be zero, the last 256 should be image tokens
        yGridFlatten = yGrid.flatten(0, 1)
        xGridFlatten = xGrid.flatten(0, 1)

        yPositions = torch.cat([yCondition, yGridFlatten], dim=0)
        xPositions = torch.cat([xCondition, xGridFlatten], dim=0)

        yAngles = torch.outer(yPositions, freqs)
        xAngles = torch.outer(xPositions, freqs)

        angles = torch.cat([yAngles, xAngles], dim=1)
        anglesFull = torch.cat([angles, angles], dim=1)
        return anglesFull

    @jaxtyped(typechecker=beartype)
    def _rotateSwap(
        self, latent: Float[torch.Tensor, "b seq numHeads headDim"]
    ) -> Float[torch.Tensor, "b seq numHeads headDim"]:
        a, b = torch.chunk(latent, chunks=2, dim=-1)
        return torch.cat([-b, a], dim=-1)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        q: Float[torch.Tensor, "b seq numHeads headDim"],
        k: Float[torch.Tensor, "b seq numHeads headDim"],
    ) -> Tuple[
        Float[torch.Tensor, "b seq numHeads headDim"],
        Float[torch.Tensor, "b seq numHeads headDim"],
    ]:
        cos = rearrange(
            self.cos, "seq headDim -> 1 seq 1 headDim", headDim=self.headDim
        )
        sin = rearrange(
            self.sin, "seq headDim -> 1 seq 1 headDim", headDim=self.headDim
        )

        qRotated = (q * cos) + (self._rotateSwap(q) * sin)  # pyrefly:ignore
        kRotated = (k * cos) + (self._rotateSwap(k) * sin)  # pyrefly:ignore

        return qRotated, kRotated
