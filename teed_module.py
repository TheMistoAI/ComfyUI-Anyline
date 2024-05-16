# This module is strongly reference from https://arxiv.org/abs/2308.06468
# Thanks for the great work and the magic activation function
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import morphology
import numpy as np
import cv2
from PIL import Image, ImageOps

@torch.jit.script
def mish_(input):
    return input * torch.tanh(F.softplus(input))

@torch.jit.script
def Fsmish(input):
    return input * torch.tanh(torch.log(1+torch.sigmoid(input)))

class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Mish.html
    """
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if torch.__version__ >= "1.9":
            return F.mish(input)
        else:
            return mish_(input)

class Smish(nn.Module):
    """
    the same thing as Mish
    """
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return Fsmish(input)



def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class CoFusion(nn.Module):
    # from LDC

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3,stride=1, padding=1) # before 64
        self.conv3= nn.Conv2d(32, out_ch, kernel_size=3,
                               stride=1, padding=1)# before 64  instead of 32
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 32) # before 64

    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = F.softmax(self.conv3(attn), dim=1)
        return ((x * attn).sum(1)).unsqueeze(1)

class CoFusion2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CoFusion2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3,stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3,stride=1, padding=1)
        self.smish = Smish()

def forward(self, x):
    attn = self.conv1(self.smish(x))
    attn = self.conv3(self.smish(attn))
    return ((x * attn).sum(1)).unsqueeze(1)

class DoubleFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleFusion, self).__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch*8, kernel_size=3,stride=1, padding=1, groups=in_ch)
        self.PSconv1 = nn.PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(24, 24*1, kernel_size=3,stride=1, padding=1,groups=24)
        self.AF= Smish()


    def forward(self, x):
        attn = self.PSconv1(self.DWconv1(self.AF(x)))
        attn2 = self.PSconv1(self.DWconv2(self.AF(attn)))
        return Fsmish(((attn2 +attn).sum(1)).unsqueeze(1))

class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('smish1', Smish()),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True))
    def forward(self, x):
        x1, x2 = x
        new_features = super(_DenseLayer, self).forward(Fsmish(x1))  # F.relu()
        return 0.5 * (new_features + x2), x2

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features

class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(Smish())
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, use_ac=False):
        super(SingleConvBlock, self).__init__()
        # self.use_bn = use_bs
        self.use_ac=use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        if self.use_ac:
            self.smish = Smish()

    def forward(self, x):
        x = self.conv(x)
        if self.use_ac:
            return self.smish(x)
        else:
            return x

class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish= Smish()#nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        return x
def process_line_art(img, threshold=0.15, gaussian_sigma=2, intensity_threshold=3, min_size=36):
    lineart_standard_res = lineart_standard_preprocessor(image=img, guassian_sigma=gaussian_sigma, intensity_threshold=intensity_threshold, resolution=img.shape[0]).squeeze().numpy()
    lineart_standard_res = get_color_range(lineart_standard_res, threshold, 1, other_color=0, mode=2)
    cleaned = morphology.remove_small_objects(lineart_standard_res.astype(bool), min_size=min_size, connectivity=1)
    return lineart_standard_res * cleaned

def lineart_standard_preprocessor(image, guassian_sigma=6.0, intensity_threshold=8, resolution=1024):
    """
    标准lineart
    :param image:
    :param guassian_sigma:
    :param intensity_threshold:
    :param resolution:
    :return:
    """
    model=LineartStandardDetector()
    tensor_image=image
    input_batch = False
    kwargs= {'guassian_sigma':guassian_sigma, 'intensity_threshold':intensity_threshold, 'resolution':resolution}
    if "detect_resolution" in kwargs:
        del kwargs["detect_resolution"] #Prevent weird case?

    if "resolution" in kwargs:
        detect_resolution = kwargs["resolution"] if type(kwargs["resolution"]) == int and kwargs["resolution"] >= 64 else 512
        del kwargs["resolution"]
    else:
        detect_resolution = 512

    if input_batch:
        np_images = np.asarray(tensor_image * 255., dtype=np.uint8)
        np_results = model(np_images, output_type="np", detect_resolution=detect_resolution, **kwargs)
        return torch.from_numpy(np_results.astype(np.float32) / 255.0)

    out_list = []
    for image in tensor_image:
        np_image = np.asarray(image.cpu() * 255., dtype=np.uint8)
        np_result = model(np_image, output_type="np", detect_resolution=detect_resolution, **kwargs)
        out_list.append(torch.from_numpy(np_result.astype(np.float32) / 255.0))
    return torch.stack(out_list, dim=0)

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
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
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
def combine_images(original, lineart):
    # Logic to combine original and lineart images
    pass

class MTEED(nn.Module):
    def __init__(self):
        super(MTEED, self).__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2,)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)
        self.block_cat = DoubleFusion(3,3)
        self.apply(weight_init)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        img_h, img_w = slice_shape
        if img_w!=t_shape[-1] or img_h!=t_shape[2]:
            new_tensor = F.interpolate(
                tensor, size=(img_h, img_w), mode='bicubic',align_corners=False)
        else:
            new_tensor=tensor
        return new_tensor
    def resize_input(self,tensor):
        t_shape = tensor.shape
        if t_shape[2] % 8 != 0 or t_shape[3] % 8 != 0:
            img_w= ((t_shape[3]// 8) + 1) * 8
            img_h = ((t_shape[2] // 8) + 1) * 8
            new_tensor = F.interpolate(
                tensor, size=(img_h, img_w), mode='bicubic', align_corners=False)
        else:
            new_tensor = tensor
        return new_tensor

    def crop_bdcn(data1, h, w, crop_h, crop_w):
        _, _, h1, w1 = data1.size()
        assert (h <= h1 and w <= w1)
        data = data1[:, :, crop_h:crop_h + h, crop_w:crop_w + w]
        return data


    def forward(self, x, original=None, is_eval=False):
        assert original is not None
        assert x.ndim == 4, x.shape
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        results = [out_1, out_2, out_3]
        block_cat = torch.cat(results, dim=1)
        block_cat = self.block_cat(block_cat)
        results.append(block_cat)
        if is_eval:
            # add post processing
            line_art_results = process_line_art(original)
            combined_results = combine_images(results[0], line_art_results)
            return combined_results
        else:
            return results