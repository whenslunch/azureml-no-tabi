import torch
from transformers import T5EncoderModel
from diffusers import FluxPipeline, FluxTransformer2DModel
from PIL import Image
import io


nf4_model_id = "sayakpaul/flux.1-dev-nf4-pkg"
text_encoder_2 = T5EncoderModel.from_pretrained(nf4_model_id, subfolder="text_encoder_2", torch_dtype=torch.float16)
transformer = FluxTransformer2DModel.from_pretrained(nf4_model_id, subfolder="transformer", torch_dtype=torch.float16)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", text_encoder_2=text_encoder_2, transformer=transformer, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

image = pipe(
            "In a big country, dreams stay with you, like a lovers voice fires the mountainside",
            guidance_scale=3.5,
            num_inference_steps=20,
            max_sequence_length=256,
            generator=torch.Generator("cuda").manual_seed(8657309)
        ).images[0]

image.save("output.png", format="PNG")

pipe.save_pretrained("./saved_model")


