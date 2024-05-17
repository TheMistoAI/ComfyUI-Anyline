from .anyline import AnyLine

NODE_CLASS_MAPPINGS = {
    "AnyLinePreprocessor": AnyLine
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyLinePreprocessor": "TheMisto.ai Anyline"
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]