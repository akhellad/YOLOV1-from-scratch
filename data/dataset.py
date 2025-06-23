"""
Dataset class for PASCAL VOC 2007
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional


class VOCDataset:
    """Dataset class for PASCAL VOC 2007"""
    
    # Les 20 classes de PASCAL VOC (dans l'ordre alphabétique)
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(self, voc_root: str, split: str = 'train', transform=None):
        """
        Args:
            voc_root: Chemin vers VOCdevkit/VOC2007/
            split: 'train', 'val', ou 'test'
            transform: Fonction de transformation à appliquer (ex: YOLOTransforms)
        """
        self.voc_root = voc_root
        self.split = split
        self.transform = transform
        
        # Mapping classe -> index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.VOC_CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Chemins vers les dossiers
        self.images_dir = os.path.join(voc_root, 'JPEGImages')
        self.annotations_dir = os.path.join(voc_root, 'Annotations')
        self.imagesets_dir = os.path.join(voc_root, 'ImageSets', 'Main')
        
        # Charger la liste des IDs d'images pour ce split
        self.image_ids = self._load_image_ids()
        
    def _load_image_ids(self) -> List[str]:
        """Charge la liste des IDs d'images pour le split donné"""
        split_file = os.path.join(self.imagesets_dir, f'{self.split}.txt')
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]
            
        print(f"Chargé {len(image_ids)} images pour le split {self.split}")
        return image_ids
    
    def _parse_annotation(self, annotation_path: str) -> Tuple[List[List[float]], List[int]]:
        """
        Parse un fichier XML d'annotation
        
        Returns:
            bboxes: Liste de [xmin, ymin, xmax, ymax] normalisées entre 0 et 1
            classes: Liste des indices de classes correspondants
        """
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Taille de l'image
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        bboxes = []
        classes = []
        
        for obj in root.findall('object'):
                
            # Nom de la classe
            class_name = obj.find('name').text
            if class_name not in self.class_to_idx:
                print(f"Classe inconnue: {class_name}, ignorée.")
                continue  # Classe inconnue, on ignore
                
            class_idx = self.class_to_idx[class_name]
            
            # Bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Normaliser les coordonnées entre 0 et 1
            xmin_norm = xmin / img_width
            ymin_norm = ymin / img_height
            xmax_norm = xmax / img_width
            ymax_norm = ymax / img_height
            
            bboxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
            classes.append(class_idx)
            
        return bboxes, classes
    
    def __len__(self) -> int:
        """Retourne le nombre d'images dans le dataset"""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Retourne un échantillon du dataset
        
        Returns:
            Si transform=None: (image: PIL Image, bboxes: List, classes: List)
            Si transform fourni: (image: np.ndarray, bboxes: List, classes: List)
        """
        image_id = self.image_ids[idx]
        
        # Charger l'image
        image_path = os.path.join(self.images_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        
        # Charger les annotations
        annotation_path = os.path.join(self.annotations_dir, f'{image_id}.xml')
        bboxes, classes = self._parse_annotation(annotation_path)
        
        # Appliquer les transformations si fournies
        if self.transform:
            image, bboxes, classes = self.transform(image, bboxes, classes)
        
        return image, bboxes, classes
    
    def get_class_name(self, class_idx: int) -> str:
        """Retourne le nom de la classe pour un index donné"""
        return self.idx_to_class[class_idx]


def create_voc_datasets(voc_root: str, transforms: Dict[str, object] = None) -> Dict[str, VOCDataset]:
    """
    Crée les datasets train, val et test
    
    Args:
        voc_root: Chemin vers VOCdevkit/VOC2007/
        transforms: Dict avec les transformations pour chaque split 
                   ex: {'train': YOLOTransforms(training=True), 'val': SimpleTransforms()}
        
    Returns:
        Dict avec les datasets pour chaque split
    """
    if transforms is None:
        transforms = {}
        
    datasets = {}
    for split in ['train', 'val', 'test']:
        transform = transforms.get(split, None)
        datasets[split] = VOCDataset(voc_root, split=split, transform=transform)
    return datasets