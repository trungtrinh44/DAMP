from .vit import VisionTransformer
from .config import get_s16_config, get_s1k16_config, get_b1k16_config, get_l1k16_config
from functools import partial

__all__ = ["ViT_S16", "ViT_S1k16", "ViT_B1k16", "ViT_L1k16"]

ViT_S16 = partial(VisionTransformer, **get_s16_config())
ViT_S1k16 = partial(VisionTransformer, **get_s1k16_config())
ViT_B1k16 = partial(VisionTransformer, **get_b1k16_config())
ViT_L1k16 = partial(VisionTransformer, **get_l1k16_config())
