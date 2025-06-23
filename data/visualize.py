"""
Utilitaires pour visualiser les annotations YOLO v1
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import seaborn as sns

# Classes VOC pour les couleurs
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Couleurs pour chaque classe
COLORS = sns.color_palette("husl", len(VOC_CLASSES))


def yolo_target_to_bboxes(target: np.ndarray, grid_size: int = 7, num_classes: int = 20) -> List[Tuple]:
    """
    Convertit un target YOLO vers une liste de bounding boxes
    
    Args:
        target: Target YOLO de shape (grid_size, grid_size, 30)
        
    Returns:
        Liste de tuples (xmin, ymin, xmax, ymax, class_idx, confidence)
    """
    bboxes = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell = target[i, j]
            
            # Vérifier s'il y a des objets dans cette cellule
            # Trouver la classe active
            class_probs = cell[:num_classes]
            if np.max(class_probs) > 0:
                class_idx = np.argmax(class_probs)
                
                # Vérifier les deux boîtes
                for box_idx in range(2):
                    offset = num_classes + box_idx * 5
                    x_rel = cell[offset + 0]
                    y_rel = cell[offset + 1]
                    w_rel = cell[offset + 2]
                    h_rel = cell[offset + 3]
                    conf = cell[offset + 4]
                    
                    if conf > 0:  # Il y a un objet dans cette boîte
                        # Convertir vers coordonnées absolues
                        center_x_abs = (j + x_rel) / grid_size
                        center_y_abs = (i + y_rel) / grid_size
                        
                        # Convertir centre+taille vers coins
                        xmin = center_x_abs - w_rel / 2
                        ymin = center_y_abs - h_rel / 2
                        xmax = center_x_abs + w_rel / 2
                        ymax = center_y_abs + h_rel / 2
                        
                        bboxes.append((xmin, ymin, xmax, ymax, class_idx, conf))
    
    return bboxes


def visualize_yolo_annotations(image: Image.Image, target: np.ndarray, 
                              title: str = "YOLO Annotations", 
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[str] = "./annotation") -> None:
    """
    Visualise une image avec ses annotations YOLO
    
    Args:
        image: Image PIL
        target: Target YOLO de shape (7, 7, 30)
        title: Titre de la figure
        figsize: Taille de la figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Image originale
    ax1.imshow(image)
    ax1.set_title("Image Originale")
    ax1.axis('off')
    
    # Image avec annotations
    ax2.imshow(image)
    ax2.set_title(title)
    ax2.axis('off')
    
    # Récupérer les bounding boxes
    bboxes = yolo_target_to_bboxes(target)
    
    img_width, img_height = image.size
    
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, class_idx, conf = bbox
        
        # Convertir en pixels
        xmin_px = xmin * img_width
        ymin_px = ymin * img_height
        xmax_px = xmax * img_width
        ymax_px = ymax * img_height
        
        # Créer le rectangle
        width_px = xmax_px - xmin_px
        height_px = ymax_px - ymin_px
        
        rect = patches.Rectangle(
            (xmin_px, ymin_px), width_px, height_px,
            linewidth=2, edgecolor=COLORS[class_idx], facecolor='none'
        )
        ax2.add_patch(rect)
        
        # Ajouter le label
        label = f"{VOC_CLASSES[class_idx]}: {conf:.2f}"
        ax2.text(xmin_px, ymin_px - 5, label, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS[class_idx], alpha=0.7),
                fontsize=10, color='white', weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Image sauvée: {save_path}")


def visualize_yolo_grid(target: np.ndarray, figsize: Tuple[int, int] = (10, 10), save_path: Optional[str] = "./grid") -> None:
    """
    Visualise la grille YOLO avec les cellules actives
    
    Args:
        target: Target YOLO de shape (7, 7, 30)
        figsize: Taille de la figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    grid_size = target.shape[0]
    
    # Créer la grille
    for i in range(grid_size + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
        ax.axvline(i, color='gray', linewidth=0.5)
    
    # Marquer les cellules avec des objets
    for i in range(grid_size):
        for j in range(grid_size):
            cell = target[i, j]
            
            # Vérifier s'il y a des objets
            conf1 = cell[20 + 4]  # première boîte
            conf2 = cell[25 + 4]  # deuxième boîte
            
            if conf1 > 0 or conf2 > 0:
                # Trouver la classe
                class_probs = cell[:20]
                class_idx = np.argmax(class_probs)
                
                # Colorier la cellule
                rect = patches.Rectangle(
                    (j, grid_size-1-i), 1, 1,  # Inverser i pour avoir (0,0) en haut à gauche
                    facecolor=COLORS[class_idx], alpha=0.6
                )
                ax.add_patch(rect)
                
                # Ajouter le texte
                ax.text(j + 0.5, grid_size-1-i + 0.5, 
                       f"{VOC_CLASSES[class_idx]}\n{int(conf1 + conf2)} obj",
                       ha='center', va='center', fontsize=8, weight='bold')
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.set_title('Grille YOLO - Cellules Actives')
    ax.set_xlabel('Colonne')
    ax.set_ylabel('Ligne')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Grille YOLO sauvée: {save_path}")