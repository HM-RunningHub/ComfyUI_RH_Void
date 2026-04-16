import os

from .nodes import NODE_CLASS_MAPPINGS

NODE_DISPLAY_NAME_MAPPINGS = {k: k for k, v in NODE_CLASS_MAPPINGS.items()}
WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "js")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
