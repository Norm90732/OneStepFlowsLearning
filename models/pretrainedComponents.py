from jaxtyping._decorator import jaxtyped
from beartype import beartype
from jaxtyping import Float     
import torch.nn as nn 
from diffusers import AutoencoderKL
import torch
#Models are between -1 and 1 for the VAE 
class FluxVAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(
             "black-forest-labs/FLUX.2-klein-9B",
            subfolder="vae"
        )
        self.encoder = vae.encoder #pyrefly:ignore
        self.quantCovolution = vae.quant_conv #pyrefly:ignore 
        scalingFactor = getattr(vae.config, "scaling_factor", None) 
        shiftFactor = getattr(vae.config, "shift_factor", None)

        self.scalingFactor = 1.0 if scalingFactor is None else float(scalingFactor)
        self.shiftFactor = 0.0 if shiftFactor is None else float(shiftFactor)

        
        del vae 
    @torch.inference_mode()
    @jaxtyped(typechecker=beartype)
    def forward(self,x:Float[torch.Tensor,"b c h w"]) -> Float[torch.Tensor,"b vaeChannel heightVae widthVae"]:
        output = self.encoder(x)
        output = self.quantCovolution(output) # pyrefly:ignore 
        mean, logvar = torch.chunk(output,chunks=2,dim=1) 
        return (mean -self.shiftFactor)* self.scalingFactor

class FluxVAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(
             "black-forest-labs/FLUX.2-klein-9B",
            subfolder="vae"
        )
        self.decoder = vae.decoder #pyrefly:ignore
        self.postQuant = vae.post_quant_conv #pyrefly:ignore
        scalingFactor = getattr(vae.config, "scaling_factor", None) 
        shiftFactor = getattr(vae.config, "shift_factor", None)

        self.scalingFactor = 1.0 if scalingFactor is None else float(scalingFactor)
        self.shiftFactor = 0.0 if shiftFactor is None else float(shiftFactor)
        del vae
        
    @torch.inference_mode()
    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[torch.Tensor, "b latentChannels heightVae widthVae"]) -> Float[torch.Tensor, "b c h w"]:
        x = x / self.scalingFactor + self.shiftFactor
        x = self.postQuant(x) # pyrefly:ignore 
        x = self.decoder(x)
        return x