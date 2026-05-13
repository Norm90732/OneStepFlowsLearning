from beartype import beartype
import torch
import torch.nn as nn
from jaxtyping import Float, jaxtyped, Int
from models.attention import InContextBlock
from models.embeddings import Conditioning2DRoPE
from models.blocks import PatchEmbedding, UnPatchEmbedding, MultiConditionEmbed
from omegaconf import DictConfig
from einops import rearrange
from beartype.typing import Tuple


class ImfDIT(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # config constants
        cfgModels = cfg.models
        self.embedDim = cfgModels.embedDim
        numHeads = cfgModels.numHeads
        patchSize = cfgModels.patchSize
        imageSize = cfgModels.imageSize
        inputDim = cfgModels.inputDim
        numClasses = cfgModels.numClasses + 1  # plus null class
        sinusoidalBaseFreq = cfgModels.sinusoidalBaseFreq
        ropeBaseFreq = cfgModels.ropeBaseFreq
        sinusoidalDim = cfgModels.sinusoidalDim
        SwiGluMultiplier = cfgModels.SwiGluMultiplier

        # transformer block
        self.sharedBackboneBlocks = cfgModels.sharedBackboneBlocks
        self.numUBlocks = cfgModels.numUBlocks
        self.numVBlocks = cfgModels.numVBlocks

        # token numbers
        classTokensNum = cfgModels.classTokensNum
        timeTokensNum = cfgModels.timeTokensNum
        omegaTokensNum = cfgModels.omegaTokensNum
        tMinTokensNum = cfgModels.tMinTokensNum
        tMaxTokensNum = cfgModels.tMaxTokensNum
        self.totalTokens = (
            classTokensNum
            + timeTokensNum
            + omegaTokensNum
            + tMinTokensNum
            + tMaxTokensNum
        )

        self.patchEmbedding = PatchEmbedding(
            inputDim=inputDim,
            embedDim=self.embedDim,
            patchSize=patchSize,
            imageSize=imageSize,
        )

        self.embeddingTable = nn.Embedding(
            num_embeddings=numClasses, embedding_dim=self.embedDim
        )
        self.constantGroupProjection = MultiConditionEmbed(
            baseFreq=sinusoidalBaseFreq,
            sinusoidalDim=sinusoidalDim,
            embedDim=self.embedDim,
        )

        # tokenCreation
        self.classTokens = nn.Parameter(
            torch.empty(classTokensNum, self.embedDim)
        )  # add embedding table to this

        self.timeTokens = nn.Parameter(
            torch.empty(timeTokensNum, self.embedDim)
        )  # h addition

        self.omegaTokens = nn.Parameter(
            torch.empty(omegaTokensNum, self.embedDim)
        )  # cfg addition (1- 1/w)

        self.tMinTokens = nn.Parameter(
            torch.empty(tMinTokensNum, self.embedDim)
        )  # tmin addition

        self.tMaxTokens = nn.Parameter(
            torch.empty(tMaxTokensNum, self.embedDim)
        )  # tmax addition

        nn.init.trunc_normal_(self.classTokens, std=0.02)
        nn.init.trunc_normal_(self.timeTokens, std=0.02)
        nn.init.trunc_normal_(self.omegaTokens, std=0.02)
        nn.init.trunc_normal_(self.tMinTokens, std=0.02)
        nn.init.trunc_normal_(self.tMaxTokens, std=0.02)

        # rope Creation
        self.rope = Conditioning2DRoPE(
            baseFrequency=ropeBaseFreq,
            numHeads=numHeads,
            embedDim=self.embedDim,
            conditioningTokensNum=self.totalTokens,
            imageSize=imageSize,
            patchSize=patchSize,
        )

        self.sharedBackboneBlocks = nn.ModuleList(
            [
                InContextBlock(
                    embedDim=self.embedDim,
                    numHeads=numHeads,
                    SwiGluMultiplier=SwiGluMultiplier,
                )
                for _ in range(self.sharedBackboneBlocks)
            ]
        )
        self.uBlocks = nn.ModuleList(
            [
                InContextBlock(
                    embedDim=self.embedDim,
                    numHeads=numHeads,
                    SwiGluMultiplier=SwiGluMultiplier,
                )
                for _ in range(self.numUBlocks)
            ]
        )
        self.vBlocks = nn.ModuleList(
            [
                InContextBlock(
                    embedDim=self.embedDim,
                    numHeads=numHeads,
                    SwiGluMultiplier=SwiGluMultiplier,
                )
                for _ in range(self.numVBlocks)
            ]
        )

        self.uRmsNorm = nn.RMSNorm(self.embedDim, eps=1e-6)
        self.vRmsNorm = nn.RMSNorm(self.embedDim, eps=1e-6)
        self.uProjection = nn.Linear(self.embedDim, self.embedDim)

        self.vProjection = nn.Linear(self.embedDim, self.embedDim)

        self.unpatchUEmbedding = UnPatchEmbedding(
            embedDim=self.embedDim,
            inputDim=inputDim,
            patchSize=patchSize,
            imageSize=imageSize,
        )
        self.unpatchVEmbedding = UnPatchEmbedding(
            embedDim=self.embedDim,
            inputDim=inputDim,
            patchSize=patchSize,
            imageSize=imageSize,
        )

        nn.init.zeros_(self.unpatchUEmbedding.linearUnpatch.weight)
        nn.init.zeros_(self.unpatchUEmbedding.linearUnpatch.bias)
        nn.init.zeros_(self.unpatchVEmbedding.linearUnpatch.weight)
        nn.init.zeros_(self.unpatchVEmbedding.linearUnpatch.bias)

    @jaxtyped(typechecker=beartype)
    def _buildSequence(
        self,
        latent: Float[torch.Tensor, "b c h w"],
        h: Float[torch.Tensor, "b"],
        omega: Float[torch.Tensor, "b"],
        cfgTStart: Float[torch.Tensor, "b"],
        cfgTEnd: Float[torch.Tensor, "b"],
        classIdx: Int[torch.Tensor, "b"],
    ) -> Float[torch.Tensor, "b seqlen embedDim"]:

        embeddedLatent = self.patchEmbedding(latent)

        # construct embedding for Tokens
        hEmbedding, omegaEmbedding, cfgTStartEmbedding, cfgTEndEmbedding = (
            self.constantGroupProjection(h, omega, cfgTStart, cfgTEnd)
        )
        classEmbeddingLookup = self.embeddingTable(classIdx.long())

        classEmbedding = rearrange(
            classEmbeddingLookup, "b embedDim -> b 1 embedDim", embedDim=self.embedDim
        )
        hEmbedding = rearrange(
            hEmbedding, "b embedDim -> b 1 embedDim", embedDim=self.embedDim
        )
        omegaEmbedding = rearrange(
            omegaEmbedding, "b embedDim -> b 1 embedDim", embedDim=self.embedDim
        )
        cfgTStartEmbedding = rearrange(
            cfgTStartEmbedding, "b embedDim -> b 1 embedDim", embedDim=self.embedDim
        )
        cfgTEndEmbedding = rearrange(
            cfgTEndEmbedding, "b embedDim -> b 1 embedDim", embedDim=self.embedDim
        )

        classTokens = rearrange(
            self.classTokens,
            "numTokens embedDim -> 1 numTokens embedDim",
            embedDim=self.embedDim,
        )
        timeTokens = rearrange(
            self.timeTokens,
            "numTokens embedDim -> 1 numTokens embedDim",
            embedDim=self.embedDim,
        )
        omegaTokens = rearrange(
            self.omegaTokens,
            "numTokens embedDim -> 1 numTokens embedDim",
            embedDim=self.embedDim,
        )
        tMinTokens = rearrange(
            self.tMinTokens,
            "numTokens embedDim -> 1 numTokens embedDim",
            embedDim=self.embedDim,
        )
        tMaxTokens = rearrange(
            self.tMaxTokens,
            "numTokens embedDim -> 1 numTokens embedDim",
            embedDim=self.embedDim,
        )

        classTokens = classTokens + classEmbedding
        timeTokens = timeTokens + hEmbedding
        omegaTokens = omegaTokens + omegaEmbedding
        tMinTokens = tMinTokens + cfgTStartEmbedding
        tMaxTokens = tMaxTokens + cfgTEndEmbedding

        inputSequence = torch.cat(
            [
                classTokens,
                omegaTokens,
                tMinTokens,
                tMaxTokens,
                timeTokens,
                embeddedLatent,
            ],
            dim=1,
        )

        return inputSequence

    @jaxtyped(typechecker=beartype)
    def _forwardBackBone(
        self, inputSequence: Float[torch.Tensor, "b seqlen embedDim"]
    ) -> Float[torch.Tensor, "b seqlen embedDim"]:
        for layer in self.sharedBackboneBlocks:
            inputSequence = layer(inputSequence, self.rope)
        return inputSequence

    @jaxtyped(typechecker=beartype)
    def _forwardULayers(
        self, inputSequence: Float[torch.Tensor, "b seqlen embedDim"]
    ) -> Float[torch.Tensor, "b seqlen embedDim"]:
        for ulayer in self.uBlocks:
            inputSequence = ulayer(inputSequence, self.rope)
        return inputSequence

    @jaxtyped(typechecker=beartype)
    def _forwardVLayers(
        self, inputSequence: Float[torch.Tensor, "b seqlen embedDim"]
    ) -> Float[torch.Tensor, "b seqlen embedDim"]:
        for vlayer in self.vBlocks:
            inputSequence = vlayer(inputSequence, self.rope)
        return inputSequence

    @jaxtyped(typechecker=beartype)
    def _sliceImageTokens(
        self, inputSequence: Float[torch.Tensor, "b seqlen embedDim"]
    ) -> Float[torch.Tensor, "b slicedSeqlen embedDim"]:
        return inputSequence[:, self.totalTokens :, :]

    @jaxtyped(typechecker=beartype)
    def _finalLayerU(
        self, uImageTokens: Float[torch.Tensor, "b seqlen embedDim"]
    ) -> Float[torch.Tensor, "b c h w"]:
        uImageTokens = self.uRmsNorm(uImageTokens)
        uImageTokens = self.uProjection(uImageTokens)
        uLatent = self.unpatchUEmbedding(uImageTokens)
        return uLatent

    @jaxtyped(typechecker=beartype)
    def _finalLayerV(
        self, vImageTokens: Float[torch.Tensor, "b seqlen embedDim"]
    ) -> Float[torch.Tensor, "b c h w"]:
        vImageTokens = self.vRmsNorm(vImageTokens)
        vImageTokens = self.vProjection(vImageTokens)
        vLatent = self.unpatchVEmbedding(vImageTokens)
        return vLatent

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        latent: Float[torch.Tensor, "b c h w"],
        h: Float[torch.Tensor, "b"],
        omega: Float[torch.Tensor, "b"],
        cfgTStart: Float[torch.Tensor, "b"],
        cfgTEnd: Float[torch.Tensor, "b"],
        classIdx: Int[torch.Tensor, "b"],
    ) -> Tuple[
        Float[torch.Tensor, "b c h w"],
        Float[torch.Tensor, "b c h w"],
    ]:
        inputSequence = self._buildSequence(
            latent=latent,
            h=h,
            omega=omega,
            cfgTStart=cfgTStart,
            cfgTEnd=cfgTEnd,
            classIdx=classIdx,
        )
        inputSequence = self._forwardBackBone(inputSequence=inputSequence)

        uSequence = self._forwardULayers(inputSequence=inputSequence)
        vSequence = self._forwardVLayers(inputSequence=inputSequence)

        uImageTokens = self._sliceImageTokens(inputSequence=uSequence)
        vImageTokens = self._sliceImageTokens(inputSequence=vSequence)

        uLatent = self._finalLayerU(uImageTokens=uImageTokens)
        vLatent = self._finalLayerV(vImageTokens=vImageTokens)

        # add to tokens
        return uLatent, vLatent

    @jaxtyped(typechecker=beartype)
    def forwardU(
        self,
        latent: Float[torch.Tensor, "b c h w"],
        h: Float[torch.Tensor, "b"],
        omega: Float[torch.Tensor, "b"],
        cfgTStart: Float[torch.Tensor, "b"],
        cfgTEnd: Float[torch.Tensor, "b"],
        classIdx: Int[torch.Tensor, "b"],
    ) -> Float[torch.Tensor, "b c h w"]:

        inputSequence = self._buildSequence(
            latent=latent,
            h=h,
            omega=omega,
            cfgTStart=cfgTStart,
            cfgTEnd=cfgTEnd,
            classIdx=classIdx,
        )
        inputSequence = self._forwardBackBone(inputSequence=inputSequence)

        uSequence = self._forwardULayers(inputSequence=inputSequence)

        uImageTokens = self._sliceImageTokens(inputSequence=uSequence)

        uLatent = self._finalLayerU(uImageTokens=uImageTokens)
        return uLatent

    @jaxtyped(typechecker=beartype)
    def forwardV(
        self,
        latent: Float[torch.Tensor, "b c h w"],
        h: Float[torch.Tensor, "b"],
        omega: Float[torch.Tensor, "b"],
        cfgTStart: Float[torch.Tensor, "b"],
        cfgTEnd: Float[torch.Tensor, "b"],
        classIdx: Int[torch.Tensor, "b"],
    ) -> Float[torch.Tensor, "b c h w"]:

        inputSequence = self._buildSequence(
            latent=latent,
            h=h,
            omega=omega,
            cfgTStart=cfgTStart,
            cfgTEnd=cfgTEnd,
            classIdx=classIdx,
        )
        inputSequence = self._forwardBackBone(inputSequence=inputSequence)

        vSequence = self._forwardVLayers(inputSequence=inputSequence)

        vImageTokens = self._sliceImageTokens(inputSequence=vSequence)

        vLatent = self._finalLayerV(vImageTokens=vImageTokens)
        return vLatent


if __name__ == "__main__":
    from omegaconf import OmegaConf

    device = torch.device("cuda")
    cfg = OmegaConf.load("configs/config.yaml")
    cfg.models = OmegaConf.load("configs/models/imfDIT.yaml")

    dit = ImfDIT(cfg=cfg).to(device)  # pyrefly:ignore

    latent = torch.randn(1, 32, 32, 32).to(device)
    h = torch.randn(1).to(device)
    omega = torch.randn(1).to(device)
    tStart = torch.randn(1).to(device)
    tEnd = torch.randn(1).to(device)
    classIdx = torch.randint(
        low=0, high=cfg.models.numClasses + 1, size=(1,), device=device
    )

    u, v = dit.forward(latent, h, omega, tStart, tEnd, classIdx)
    print("normal forward")
    print(u.shape)
    print(v.shape)

    print("u forward")
    u = dit.forwardU(latent, h, omega, tStart, tEnd, classIdx)
    print(u.shape)
    print("v forward")
    v = dit.forwardV(latent, h, omega, tStart, tEnd, classIdx)
    print(v.shape)
