import torch
import numpy as np
import kornia as kn

class AnyLine:

    @classmethod
    def INPUT_TYPES(s):

        return {
            # TODO need to be other thing
            "required": {
                "E-mail":("STRING",{
                    "default": "Enter your resign email（输入你注册的email地址）"
                }),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("anyline_image",)

    FUNCTION = "get_anyline"
    CATEGORY = "TheMisto/image/preprocessor"

    def get_anyline(self, email, image):
        return (image,)

NODE_CLASS_MAPPINGS = {
    "AnyLine": AnyLine
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyLine": "TheMisto Anyline"
}


if __name__ == "__main__":
    # loader
    checkpoint_path = ""
    model = TED().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        images = sample_batched['images'].to(device)
        preds = model(images, single_test=resize_input)

        img_height, img_width = img_shape[0].item(), img_shape[1].item()
        image_vis = kn.utils.tensor_to_image(torch.sigmoid(tensor_image))
        image_vis = (255.0 * (1.0 - image_vis)).astype(np.uint8)

        image_vis = cv2.resize(image_vis, (img_width, img_height))