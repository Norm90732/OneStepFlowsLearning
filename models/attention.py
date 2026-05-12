import torch
import torch.nn as nn
from jaxtyping import jaxtyped, Float
from beartype import beartype
from jvp_flash_attention.jvp_attention import JVPAttn
from models.blocks import SwiGlu
from models.embeddings import Conditioning2DRoPE
from einops import rearrange


class InContextBlock(nn.Module):
    def __init__(self,
                 embedDim:int,
                 numHeads:int,
                 SwiGluMultiplier:int):
        super().__init__()
        
        self.numHeads = numHeads
        self.headDim = embedDim// numHeads
        
        #attention 
        self.rmsNorm1 = nn.RMSNorm(embedDim,eps=1e-6)
        self.qkvProj = nn.Linear(embedDim,embedDim*3)
        self.qNorm = nn.RMSNorm(self.headDim,eps=1e-6)
        self.kNorm = nn.RMSNorm(self.headDim,eps=1e-6)
        self.outProjection = nn.Linear(embedDim,embedDim)
        #SwiGlu
        self.rmsNorm2 = nn.RMSNorm(embedDim,eps=1e-6)
        self.swiglu = SwiGlu(
            embedDim=embedDim,
            multiplier=SwiGluMultiplier
        )
        
        
    @jaxtyped(typechecker=beartype)
    def forward(self,latent:Float[torch.Tensor,"b seqlen embedDim"],rope:Conditioning2DRoPE) -> Float[torch.Tensor,"b seqlen embedDim"]:
        residual1 = latent
        rmsNorm1 = self.rmsNorm1(latent)
        qkv = self.qkvProj(rmsNorm1)
        q,k,v = torch.chunk(qkv,chunks=3,dim=-1)
        q = rearrange(q,"b seqlen (numHeads headDim) -> b seqlen numHeads headDim",numHeads=self.numHeads,headDim=self.headDim) 
        k = rearrange(k,"b seqlen (numHeads headDim) -> b seqlen numHeads headDim",numHeads=self.numHeads,headDim=self.headDim)
        v = rearrange(v,"b seqlen (numHeads headDim) -> b seqlen numHeads headDim",numHeads=self.numHeads,headDim=self.headDim)
        
        qNorm = self.qNorm(q)
        kNorm = self.kNorm(k)
        
        qRotated, kRotated = rope(qNorm,kNorm)
        
        #rearrange for jvp kernel 
        qJVP = rearrange(qRotated,"b seqlen numHeads headDim -> b numHeads seqlen headDim",numHeads=self.numHeads,headDim=self.headDim)
        kJVP = rearrange(kRotated,"b seqlen numHeads headDim -> b numHeads seqlen headDim",numHeads=self.numHeads,headDim=self.headDim)
        vJVP = rearrange(v,"b seqlen numHeads headDim -> b numHeads seqlen headDim",numHeads=self.numHeads,headDim=self.headDim)

        
        attention=JVPAttn.fwd_dual(
                        q=qJVP.contiguous(),
                         k=kJVP.contiguous(),
                         v=vJVP.contiguous(),
                         attn_mask=None)
        
        attentionReshape = rearrange(
            attention, 
            "b numHeads seqlen headDim -> b seqlen (numHeads headDim)", 
            numHeads=self.numHeads, 
            headDim=self.headDim
        )
        
        attentionProjection = self.outProjection(attentionReshape)
        
        attentionResidual = residual1 + attentionProjection
        
        residual2 = attentionResidual
        
        rmsNorm2 = self.rmsNorm2(attentionResidual)
        
        swigluOutput = self.swiglu(rmsNorm2)
        
        return swigluOutput + residual2
    
    
        
        
        
if __name__ == "__main__":
    device = torch.device("cuda")
    rope = Conditioning2DRoPE(
        baseFrequency=10000,
        numHeads=12,
        embedDim=768,
        conditioningTokensNum=20,
        imageSize=32,
        patchSize=2,
    ).to(device)
    
    
    modelBlock = InContextBlock(
        embedDim=768,
        numHeads=12,
        SwiGluMultiplier=3,
    ).to(device)
    
    testTensor = torch.randn(64,276,768).to(device)
    tanget = torch.randn_like(testTensor).to(device)
    
    def wrapperFunction(latent):
        return modelBlock(latent,rope)
    
    print(f"testTensor:{testTensor.shape}")
    output = modelBlock(testTensor,rope)
    print(f"output:{output.shape}")
    
    
    primal, tangetOut = torch.func.jvp(
        func=wrapperFunction,
        primals=(testTensor,),
        tangents=(tanget,)
    )
    print(primal.shape)
    print(tangetOut.shape)
    print(primal)
    print(tangetOut)