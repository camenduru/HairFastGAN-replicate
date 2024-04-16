import os, sys
from cog import BasePredictor, Input, Path
from typing import List
sys.path.append('/content/HairFastGAN')
os.chdir('/content/HairFastGAN')

import torchvision.transforms as transforms

class Predictor(BasePredictor):
    def setup(self) -> None:
        from hair_swap import HairFast, get_parser
        model_args = get_parser()
        self.hair_fast = HairFast(model_args.parse_args([]))
    def predict(
        self,
        face_image: Path = Input(description="Face image"),
        shape_image: Path = Input(description="Shape image"),
        color_image: Path = Input(description="Color image"),
    ) -> Path:
        final_image = self.hair_fast.swap(face_image, shape_image, color_image)
        image_pil = transforms.ToPILImage()(final_image)
        image_pil.save("/content/final_image.png")
        return Path("/content/final_image.png")