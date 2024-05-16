import os
import torch
import numpy as np
import kornia as kn
from pathlib import Path
from .teed_module import MTEED
from huggingface_hub import hf_hub_download
import cv2
from skimage import morphology
from PIL import Image
from einops import rearrange

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
        self.mteed_detector = self.MTEEDDetector(self.model, self.device)

    def load_model(self):
        checkpoint_filename = "MTEED.pth"
        here = Path(__file__).parent.resolve()
        checkpoint_dir = here / "checkpoints"
        checkpoint_path = checkpoint_dir / checkpoint_filename

        # Download the model if it's not present
        if not checkpoint_path.is_file():
            print("Model not found locally, downloading from HuggingFace...")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = hf_hub_download(repo_id="TheMistoAI/MistoLine", filename=checkpoint_filename, local_dir=checkpoint_dir)

        model = MTEED().to(self.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        return model

    def get_anyline(self, image):
        # Process the image with MTEED model
        mteed_result = self.mteed_detector(image[0], detect_resolution=image[0].shape[0])
        print("mteed_result: ", mteed_result)

        # Process the image with the lineart standard preprocessor
        lineart_result = process_line_art(image[0])
        print("lineart_result: ", lineart_result)
        
        # Combine the results
        

        return (torch.tensor(lineart_result),)
    
    class MTEEDDetector:
        def __init__(self, model, device):
            self.model = model
            self.device = device
        def __call__(self, input_image, detect_resolution=512, safe_steps=2, upscale_method="INTER_CUBIC", output_type=None, **kwargs):
            input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
            input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

            H, W, _ = input_image.shape
            with torch.no_grad():
                image_teed = torch.from_numpy(input_image.copy()).float().to(self.device)
                image_teed = rearrange(image_teed, 'h w c -> 1 c h w')
                edges = self.model(image_teed)
                edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
                edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
                edges = np.stack(edges, axis=2)
                edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
                if safe_steps != 0:
                    edge = safe_step(edge, safe_steps)
                edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

            detected_map = remove_pad(HWC3(edge))

            return  detected_map
    
def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y
def lineart_standard_preprocessor(image, guassian_sigma=6.0, intensity_threshold=8, resolution=1024):
    """
    Standard lineart
    :param image: Input image (should be a single image tensor)
    :param guassian_sigma: Gaussian blur sigma
    :param intensity_threshold: Threshold for intensity to define edges
    :param resolution: Resolution for detection
    :return: Processed image tensor
    """
    model = LineartStandardDetector()
    kwargs = {'guassian_sigma': guassian_sigma, 'intensity_threshold': intensity_threshold, 'resolution': resolution}
    
    detect_resolution = kwargs.pop('resolution', 512) if isinstance(kwargs.get('resolution', 512), int) and kwargs.get('resolution', 512) >= 64 else 512
    
    # Convert tensor image to numpy array, apply model, and convert back to tensor
    np_image = np.asarray(image.cpu() * 255., dtype=np.uint8)  # Assuming the input image tensor is in the range [0, 1]
    np_result = model(np_image, output_type="np", detect_resolution=detect_resolution, **kwargs)
    tensor_result = torch.from_numpy(np_result.astype(np.float32) / 255.0)  # Convert back to tensor in the range [0, 1]
    
    return tensor_result

class LineartStandardDetector:
    def __call__(self, input_image=None, guassian_sigma=6.0, intensity_threshold=8, detect_resolution=512,
                 output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        x = input_image.astype(np.float32)
        g = cv2.GaussianBlur(x, (0, 0), guassian_sigma)
        intensity = np.min(g - x, axis=2).clip(0, 255)
        intensity /= max(16, np.median(intensity[intensity > intensity_threshold]))
        intensity *= 127
        detected_map = intensity.clip(0, 255).astype(np.uint8)

        detected_map = HWC3(remove_pad(detected_map))

        return detected_map
def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
def common_input_validate(input_image, output_type, **kwargs):
    if "img" in kwargs:
        # warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
        input_image = kwargs.pop("img")

    if "return_pil" in kwargs:
        # warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
        output_type = "pil" if kwargs["return_pil"] else "np"

    if type(output_type) is bool:
        # warnings.warn( "Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
        if output_type:
            output_type = "pil"

    if input_image is None:
        raise ValueError("input_image must be defined.")

    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
        output_type = output_type or "pil"
    else:
        output_type = output_type or "np"

    return (input_image, output_type)

def resize_image_with_pad(input_image, resolution, upscale_method = "", skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=get_upscale_method(upscale_method) if k > 1 else cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad
UPSCALE_METHODS = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]
def get_upscale_method(method_str):
    assert method_str in UPSCALE_METHODS, f"Method {method_str} not found in {UPSCALE_METHODS}"
    return getattr(cv2, method_str)

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()
def get_color_range(numpy_0_1, grey_num_min_0_1, grey_num_max_0_1, other_color=0, mode=1):
    # 取灰度在某一范围内的像素grey_num_min__0_1,grey_num_max_0_1为灰度最大值最小值（包括等于）注意误差！！，other_color是所有不符合像素设成的颜色，mode=1时所有符合像素设成1，mode=2时所有符合像素保持不变
    result = numpy_0_1[:, :, 0]
    if mode == 1:
        result = np.where((result >= grey_num_min_0_1) & (result <= grey_num_max_0_1), 1, other_color)
    elif mode == 2:
        result = np.where((result >= grey_num_min_0_1) & (result <= grey_num_max_0_1), result, other_color)
    result = np.expand_dims(result, 2).repeat(3, axis=2)
    return result
def process_line_art(img,threshold=0.15,gaussian_sigma=2,intensity_threshold=3,min_size=36):
    # threshold=0.5,gaussian_sigma=3,intensity_threshold=15,min_size=81
    lineart_standard_res = lineart_standard_preprocessor(image=img,guassian_sigma=gaussian_sigma, intensity_threshold=intensity_threshold,resolution=img.shape[0]).squeeze().numpy()
    print("lineart_standard_res: ", lineart_standard_res)
    lineart_standard_res = get_color_range(lineart_standard_res,threshold,1,other_color=0,mode=2)
    print("lineart_standard_res 2: ", lineart_standard_res)
    cleaned = morphology.remove_small_objects(lineart_standard_res.astype(bool), min_size=min_size, connectivity=1)
    return lineart_standard_res*cleaned
