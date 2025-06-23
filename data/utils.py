"""
Utilitaires pour convertir les annotations VOC vers format YOLO v1
"""

import numpy as np
from typing import List, Tuple


def convert_to_yolo_target(bboxes: List[List[float]], classes: List[int], 
                          grid_size: int = 7, num_classes: int = 20) -> np.ndarray:
    """
    Convertit les bounding boxes VOC vers le format target YOLO v1
    
    Args:
        bboxes: Liste de [xmin, ymin, xmax, ymax] normalisées (0-1)
        classes: Liste des indices de classes correspondants
        grid_size: Taille de la grille (7 pour YOLO v1)
        num_classes: Nombre de classes (20 pour VOC)
    
    Returns:
        target: Array de shape (grid_size, grid_size, 30)
                30 = 20 classes + 2*(x,y,w,h,confidence)
    """
    
    # Initialiser le target: (7, 7, 30)
    target = np.zeros((grid_size, grid_size, num_classes + 5 * 2))
    
    # Traiter chaque objet
    for bbox, class_idx in zip(bboxes, classes):
        xmin, ymin, xmax, ymax = bbox
        
        # 1. Convertir vers centre + taille
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        # 2. Déterminer la cellule responsable (celle qui contient le centre)
        cell_x = int(center_x * grid_size)  # 0 à 6
        cell_y = int(center_y * grid_size)  # 0 à 6
        
        # S'assurer qu'on reste dans la grille
        cell_x = min(cell_x, grid_size - 1)
        cell_y = min(cell_y, grid_size - 1)
        
        # 3. Coordonnées relatives à la cellule (0-1 dans la cellule)
        x_rel = (center_x * grid_size) - cell_x  # Position dans la cellule
        y_rel = (center_y * grid_size) - cell_y
        
        # 4. Taille relative à l'image entière (déjà entre 0-1)
        w_rel = width
        h_rel = height
        
        # 5. Assigner à la première boîte de cette cellule
        # Layout du target: [20 classes, x1, y1, w1, h1, conf1, x2, y2, w2, h2, conf2]
        box_offset = num_classes  # Les 20 premières sont les classes
        
        # Si aucune boîte n'est assignée dans cette cellule, utiliser la première
        if target[cell_y, cell_x, box_offset + 4] == 0:  # conf1 == 0
            # Première boîte
            target[cell_y, cell_x, box_offset + 0] = x_rel      # x1
            target[cell_y, cell_x, box_offset + 1] = y_rel      # y1
            target[cell_y, cell_x, box_offset + 2] = w_rel      # w1
            target[cell_y, cell_x, box_offset + 3] = h_rel      # h1
            target[cell_y, cell_x, box_offset + 4] = 1.0        # conf1 = 1
        elif target[cell_y, cell_x, box_offset + 9] == 0:  # conf2 == 0
            # Deuxième boîte
            target[cell_y, cell_x, box_offset + 5] = x_rel      # x2
            target[cell_y, cell_x, box_offset + 6] = y_rel      # y2
            target[cell_y, cell_x, box_offset + 7] = w_rel      # w2
            target[cell_y, cell_x, box_offset + 8] = h_rel      # h2
            target[cell_y, cell_x, box_offset + 9] = 1.0        # conf2 = 1
        # Si les 2 boîtes sont prises, on ignore cet objet
        
        # 6. Marquer la classe (one-hot)
        target[cell_y, cell_x, class_idx] = 1.0
        
    return target


def visualize_yolo_target(target: np.ndarray, grid_size: int = 7, num_classes: int = 20) -> None:
    """
    Visualise le contenu d'un target YOLO pour debugging
    
    Args:
        target: Target YOLO de shape (grid_size, grid_size, 30)
    """
    print(f"Target shape: {target.shape}")
    print("Cellules non-vides:")
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell = target[i, j]
            
            # Vérifier s'il y a des objets dans cette cellule
            conf1 = cell[num_classes + 4]  # confidence première boîte
            conf2 = cell[num_classes + 9]  # confidence deuxième boîte
            
            if conf1 > 0 or conf2 > 0:
                # Trouver la classe active
                class_probs = cell[:num_classes]
                active_class = np.argmax(class_probs)
                
                print(f"  Cellule ({i},{j}): classe {active_class}")
                
                if conf1 > 0:
                    x1, y1, w1, h1 = cell[num_classes:num_classes+4]
                    print(f"    Boîte 1: x={x1:.3f}, y={y1:.3f}, w={w1:.3f}, h={h1:.3f}, conf={conf1:.3f}")
                
                if conf2 > 0:
                    x2, y2, w2, h2 = cell[num_classes+5:num_classes+9]
                    print(f"    Boîte 2: x={x2:.3f}, y={y2:.3f}, w={w2:.3f}, h={h2:.3f}, conf={conf2:.3f}")