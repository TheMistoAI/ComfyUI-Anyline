import torch
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from skimage import morphology

# Requires comfyui_controlnet_aux funcsions and classes
from custom_nodes.comfyui_controlnet_aux.utils import common_annotator_call
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.teed import TEDDetector
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.teed.ted import TED
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.lineart_standard import LineartStandardDetector

class AnyLine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "get_anyline"
    CATEGORY = "TheMisto/image/preprocessor"

    def __init__(self):
        self.device = "cpu"

    def load_model(self):
        subfolder = "Anyline"
        checkpoint_filename = "MTEED.pth"
        checkpoint_dir = Path(__file__).parent.resolve() / "checkpoints" / subfolder
        checkpoint_path = checkpoint_dir / checkpoint_filename

        # Download the model if it's not present
        if not checkpoint_path.is_file():
            print("Model not found locally, downloading from HuggingFace...")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = hf_hub_download(repo_id="TheMistoAI/MistoLine", filename=checkpoint_filename, subfolder=subfolder, local_dir=checkpoint_dir)

        model = TED()
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        return model

    def get_anyline(self, image):
        # Process the image with MTEED model
        mteed_model = TEDDetector(model=self.load_model()).to(self.device)
        mteed_result = common_annotator_call(mteed_model, image, resolution=1280)
        mteed_result = mteed_result.squeeze(0).numpy()


        # Process the image with the lineart standard preprocessor
        lineart_standard_detector = LineartStandardDetector()
        lineart_result  = common_annotator_call(lineart_standard_detector, image, guassian_sigma=2, intensity_threshold=3, resolution=1280).squeeze().numpy()
        lineart_result  = get_intensity_mask(lineart_result, lower_bound=0, upper_bound=1)
        cleaned = morphology.remove_small_objects(lineart_result .astype(bool), min_size=36, connectivity=1)
        lineart_result = lineart_result *cleaned

        # Combine the results
        final_result = combine_layers(mteed_result, lineart_result)

        del mteed_model
        return (torch.tensor(final_result).unsqueeze(0),)

def get_intensity_mask(image_array, lower_bound, upper_bound):
    mask = image_array[:, :, 0]
    mask = np.where((mask >= lower_bound) & (mask <= upper_bound), mask, 0)
    mask = np.expand_dims(mask, 2).repeat(3, axis=2)
    return mask

def combine_layers(base_layer, top_layer):
    mask = top_layer.astype(bool)
    temp = 1 - (1 - top_layer) * (1 - base_layer)
    result = base_layer * (~mask) + temp * mask
    return result
