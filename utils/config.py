"""
Configuration management for YOLO v1 project
"""

import yaml
import os


class Config:
    """Classe simple pour charger et accéder à la configuration YOLO v1"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Charge le fichier de configuration YAML"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def get(self, key_path: str, default=None):
        """
        Récupère une valeur avec un chemin type 'model.input_size'
        
        Args:
            key_path: Chemin vers la valeur (ex: 'model.input_size')
            default: Valeur par défaut si la clé n'existe pas
        """
        keys = key_path.split('.')
        value = self.config
        print(f"Accessing config with keys: {keys}")
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

def load_config(config_path: str = "configs/config.yaml") -> Config:
    """Charge et retourne une instance de Config"""
    return Config(config_path)