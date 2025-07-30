import base64
import os
from torchvision.transforms.functional import to_pil_image
from openai import OpenAI
import io
import torch
import numpy as np
from PIL import Image
from ...data_processing import tensor_to_pil, pil_to_tensor


class GPTImage:
    def __init__(self, data_params):
        self.client = OpenAI(organization=os.getenv("OPENAI_ORG_KEY", None))
        self.prompt = "Please recreate the exact same image without any alterations. Please preserve the original resolution (1024*1024)."
        self.data_params = data_params

    def eval(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input):
        results = []
        for image in input:
            image = tensor_to_pil(image, **self.data_params)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            image_file = ("image.png", buffer, "image/png")

            try:
                result = self.client.images.edit(
                    model="gpt-image-1",
                    image=image_file,
                    prompt=self.prompt,
                    n=1,
                    size="1024x1024",
                )
                image_base64 = result.data[0].b64_json
                image_bytes = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_bytes))
                results.append(pil_to_tensor(image, **self.data_params))
            except Exception as e:
                print("💥 Unexpected error occurred:", e)
                results.append(None)

        return results, None, None
