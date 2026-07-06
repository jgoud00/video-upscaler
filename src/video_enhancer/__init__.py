"""video-enhancer package.

Fine-tune pretrained deep learning models for video enhancement.
"""

__version__ = "0.1.0"

import sys

# Monkey-patch basicsr compatibility for newer torchvision versions
try:
    import torchvision.transforms.functional as TF
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        sys.modules["torchvision.transforms.functional_tensor"] = sys.modules["torchvision.transforms.functional"]
except ImportError:
    pass
