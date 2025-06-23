"""
Transformations d'images pour YOLO v1
"""

import numpy as np
from PIL import Image, ImageEnhance
from typing import List, Tuple
import random


class SimpleTransforms:
    """Version simple sans augmentations"""
    
    def __init__(self, target_size: int = 448):
        self.target_size = target_size
    
    def __call__(self, image: Image.Image, bboxes: List[List[float]], classes: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        image_resized = image.resize((self.target_size, self.target_size), Image.LANCZOS)
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        return image_array, bboxes, classes


class YOLOTransforms:
    """Transformations avec augmentations YOLO v1 et adaptation des bboxes"""
    
    def __init__(self, target_size: int = 448, training: bool = True):
        self.target_size = target_size
        self.training = training
    
    def random_hsv_adjustment(self, image: Image.Image) -> Image.Image:
        if not self.training:
            return image
            
        exposure_factor = random.uniform(0.67, 1.5)
        saturation_factor = random.uniform(0.67, 1.5)
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(exposure_factor)
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)
        
        return image
    
    def random_scale_translate(self, image: Image.Image, bboxes: List[List[float]]) -> Tuple[Image.Image, List[List[float]]]:
        if not self.training:
            return image, bboxes
            
        orig_width, orig_height = image.size
        
        scale_factor = random.uniform(0.8, 1.2)
        translate_x = random.uniform(-0.2, 0.2)
        translate_y = random.uniform(-0.2, 0.2)
        
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        
        image_scaled = image.resize((new_width, new_height), Image.LANCZOS)
        
        final_image = Image.new('RGB', (orig_width, orig_height), (128, 128, 128))
        
        paste_x = int((orig_width - new_width) / 2 + translate_x * orig_width)
        paste_y = int((orig_height - new_height) / 2 + translate_y * orig_height)
        
        final_image.paste(image_scaled, (paste_x, paste_y))
        
        # Adapter les bboxes
        new_bboxes = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            
            # Convertir en pixels absolus
            xmin_px = xmin * orig_width
            ymin_px = ymin * orig_height
            xmax_px = xmax * orig_width
            ymax_px = ymax * orig_height
            
            # Appliquer scale
            xmin_scaled = xmin_px * scale_factor
            ymin_scaled = ymin_px * scale_factor
            xmax_scaled = xmax_px * scale_factor
            ymax_scaled = ymax_px * scale_factor
            
            # Appliquer translate
            xmin_final = xmin_scaled + paste_x
            ymin_final = ymin_scaled + paste_y
            xmax_final = xmax_scaled + paste_x
            ymax_final = ymax_scaled + paste_y
            
            # Clipper aux limites
            xmin_final = max(0, min(xmin_final, orig_width))
            ymin_final = max(0, min(ymin_final, orig_height))
            xmax_final = max(0, min(xmax_final, orig_width))
            ymax_final = max(0, min(ymax_final, orig_height))
            
            # Reconvertir en normalisé
            xmin_norm = xmin_final / orig_width
            ymin_norm = ymin_final / orig_height
            xmax_norm = xmax_final / orig_width
            ymax_norm = ymax_final / orig_height
            
            # Vérifier que la bbox est encore valide
            if xmax_norm > xmin_norm and ymax_norm > ymin_norm:
                new_bboxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
        
        return final_image, new_bboxes
    
    def __call__(self, image: Image.Image, bboxes: List[List[float]], classes: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        image_aug = self.random_hsv_adjustment(image)
        image_aug, bboxes_aug = self.random_scale_translate(image_aug, bboxes)
        
        # Filtrer les classes correspondantes si des bboxes ont été supprimées
        if len(bboxes_aug) < len(bboxes):
            classes_filtered = classes[:len(bboxes_aug)]  # Simple: garder les premières
        else:
            classes_filtered = classes
        
        image_resized = image_aug.resize((self.target_size, self.target_size), Image.LANCZOS)
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        
        return image_array, bboxes_aug, classes_filtered


def denormalize_simple(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)