"""
Script pour télécharger PASCAL VOC 2007 dataset
"""

import os
import urllib.request
import tarfile
from tqdm import tqdm


def download_with_progress(url: str, filepath: str):
    """Télécharge un fichier avec barre de progression"""
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            bar_length = 50
            filled_length = (percent * bar_length) // 100
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r[{bar}] {percent}% ({downloaded // 1024 // 1024}MB)', end='')
    
    print(f"Téléchargement de {url}")
    urllib.request.urlretrieve(url, filepath, progress_hook)
    print()


def extract_tar(tar_path: str, extract_to: str):
    """Extrait un fichier tar"""
    print(f"Extraction de {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_to)
    print("Extraction terminée!")


def download_pascal_voc_2007(data_dir: str = "data"):
    """
    Télécharge PASCAL VOC 2007 dataset
    
    Args:
        data_dir: Dossier où sauvegarder les données
    """
    
    # URLs officielles PASCAL VOC 2007
    urls = {
        'trainval': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'test': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'
    }
    
    # Créer le dossier data
    os.makedirs(data_dir, exist_ok=True)
    
    for split, url in urls.items():
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        
        # Vérifier si déjà téléchargé
        if os.path.exists(filepath):
            print(f"{filename} déjà téléchargé, on passe l'étape...")
            continue
            
        # Télécharger
        try:
            download_with_progress(url, filepath)
            print(f"{filename} téléchargé avec succès")
        except Exception as e:
            print(f"Erreur lors du téléchargement de {filename}: {e}")
            continue
            
        # Extraire
        try:
            extract_tar(filepath, data_dir)
            print(f"{filename} extrait avec succès")
            
            # Supprimer le .tar après extraction
            os.remove(filepath)
            print(f"{filename} supprimé (fichier extrait conservé)")
            
        except Exception as e:
            print(f"Erreur lors de l'extraction de {filename}: {e}")
    
    # Vérifier la structure finale
    voc_path = os.path.join(data_dir, "VOCdevkit", "VOC2007")
    if os.path.exists(voc_path):
        print(f"\nPASCAL VOC 2007 installé avec succès dans: {voc_path}")
        
        # Afficher la structure
        subdirs = ['JPEGImages', 'Annotations', 'ImageSets']
        for subdir in subdirs:
            path = os.path.join(voc_path, subdir)
            if os.path.exists(path):
                count = len(os.listdir(path))
                print(f"  {subdir}: {count} fichiers")
    else:
        print("Problème avec l'installation, vérifier les erreurs ci-dessus")


if __name__ == "__main__":
    print("=== Téléchargement PASCAL VOC 2007 ===")
    download_pascal_voc_2007()
    print("=== Terminé ===")