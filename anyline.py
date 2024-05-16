import os
import torch
import numpy as np
import kornia as kn
from pathlib import Path
from .teed_module import MTEED
from huggingface_hub import hf_hub_download
import cv2  # Ensure you have OpenCV installed

class AnyLine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("anyline_image",)

    FUNCTION = "get_anyline"
    CATEGORY = "TheMisto/image/preprocessor"

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()

    def load_model(self):
        checkpoint_filename = "MTEED.safetetensors"
        here = Path(__file__).parent.resolve()
        checkpoint_dir = here / "checkpoints"
        checkpoint_path = checkpoint_dir / checkpoint_filename

        if not checkpoint_path.is_file():
            print("Model not found locally, downloading from HuggingFace...")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = hf_hub_download(repo_id="TheMistoAI/MistoLine", filename=checkpoint_filename, cache_dir=checkpoint_dir)

        model = MTEED().to(self.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        return model


    def download_model(self):
        hf_hub_download(repo_id=self.REPO_ID, filename=self.FILENAME, local_dir="./checkpoints", force_download=True)

    def get_anyline(self, image):
        image_tensor = torch.tensor(image).to(self.device)
        original_tensor = image_tensor.clone()

        with torch.no_grad():
            preds = self.model(image_tensor, original=original_tensor, is_eval=True)
            img_height, img_width = image.shape[2], image.shape[3]
            image_vis = kn.utils.tensor_to_image(torch.sigmoid(preds[0]))
            image_vis = (255.0 * (1.0 - image_vis)).astype(np.uint8)
            image_vis = cv2.resize(image_vis, (img_width, img_height))
        
        return (image_vis,)

NODE_CLASS_MAPPINGS = {
    "AnyLine": AnyLine
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyLine": "TheMisto Anyline"
}
