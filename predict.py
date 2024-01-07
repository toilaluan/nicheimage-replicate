# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import requests
from PIL import Image
import base64
import io


def base64_to_pil_image(base64_image: str) -> Image.Image:
    image_stream = io.BytesIO(base64.b64decode(base64_image))
    image = Image.open(image_stream)

    return image


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=1.5
        ),
        prompt: str = Input(description="Prompt to generate image from"),
        seed: int = Input(description="Seed for random number generator", default=0),
        width: int = Input(
            description="Width of generated image", ge=0, le=2048, default=512
        ),
        height: int = Input(
            description="Height of generated image", ge=0, le=2048, default=512
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps to run",
            ge=0,
            le=50,
            default=30,
        ),
        model_name: str = Input(
            description="Name of model to use",
            choices=["RealisticVision", "SDXLTurbo"],
            default="RealisticVision",
        ),
        guidance_scale: float = Input(
            description="CFG. SDXLTurbo should be 0.0 -> 1.0", ge=0, le=1, default=7
        ),
        key: str = Input(description="API key", default="replicate"),
    ) -> Path:
        data = {
            "key": key,
            "model_name": model_name,
            "prompt": prompt,
            "pipeline_params": {
                "guidance_scale": scale,
                "seed": seed,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        }
        response = requests.post(
            "http://proxy_client_nicheimage.nichetensor.com:10003/generate", json=data
        )
        base64 = response.json()
        image = base64_to_pil_image(base64)
        image.save("output.png")
        return [Path("output.png")]
