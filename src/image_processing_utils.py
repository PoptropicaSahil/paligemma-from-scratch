from typing import List, Optional, Union, Tuple, Iterable

from PIL import Image
import numpy as np


class IMAGE_UTILS:

    def resize(
        self,
        image: Image.Image, # changed
        size: Tuple[int, int],
        resample: Optional[Image.Resampling] = None, # changed
        reducing_gap: Optional[int] = None,
    ) -> Image.Image: # changed
        height, width = size
        return image.resize(
            (width, height), resample=resample, reducing_gap=reducing_gap
        )
    
    def rescale(
        self,
        image: np.ndarray,
        scale: float,
        dtype: np.dtype = np.dtype(np.float32), # corrected
    ) -> np.ndarray:
        rescaled_image = image * scale
        return rescaled_image.astype(dtype)
    
    def normalize(
        self,
        image: np.ndarray,
        mean: Optional[Union[float, Iterable[float]]] = None,
        std: Optional[Union[float, Iterable[float]]] = None,
    ) -> np.ndarray:
        mean = np.array(mean, dtype=image.dtype)
        std = np.array(std, dtype=image.dtype)
        image = (image - mean) / std
        return image
    
    def process_images(
        self,
        images: List[Image.Image],
        size: Tuple[int, int], # Given Dict[str, int],
        resample: Image.Resampling,
        rescale_factor: float,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
    ) -> List[np.ndarray]:
        height, width = size[0], size[1]
        images = [self.resize(image=image, size=(height, width), resample=resample) for image in images]
        
        # Giving new names because type annotations go crazy if image = [np.array(i) for i in images]
        # Because we defined it as List[Image.Image], so now converting to List of np array raises type errors
        # Convert Each image to numpy array
        images_sized = [np.array(image) for image in images]
        
        images_scaled = [self.rescale(image, scale=rescale_factor) for image in images_sized]
        images_norm = [self.normalize(image, mean=image_mean, std=image_std) for image in images_scaled]

        # Move the channel to the first dimension. Model expects
        # image (Channel, Height, Width)
        images_return = [image.transpose((2, 0, 1)) for image in images_norm]
        return images_return
