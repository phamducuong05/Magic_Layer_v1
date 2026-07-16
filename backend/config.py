import os
import yaml
import torch
from typing import Any, Dict

class ConfigManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = "config.yaml"):
        if not hasattr(self, 'config_dict'):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_dir, config_path)
            
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Config file not found: {full_path}")
                
            with open(full_path, "r", encoding="utf-8") as f:
                self.config_dict = yaml.safe_load(f)

    @property
    def device(self) -> str:
        dev = self.config_dict.get("device", "auto")
        if dev == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return dev

    def get_model_config(self, category: str) -> Dict[str, Any]:
        """
        Lấy cấu hình cho một category cụ thể (vd: matting, inpainting).
        Trả về dict chứa 'name' và các tham số của model đang active.
        """
        cat_config = self.config_dict.get("models", {}).get(category, {})
        active_name = cat_config.get("active")
        
        if not active_name:
            raise ValueError(f"No active model specified for category: {category}")
            
        specific_config = cat_config.get(active_name, {})
        return {"name": active_name, **specific_config}

    def has_active_model(self, category: str) -> bool:
        """Return whether a model category has a concrete active adapter."""
        category_config = self.config_dict.get("models", {}).get(category, {})
        return bool(
            isinstance(category_config, dict)
            and category_config.get("active")
        )

    def get_pipeline_config(self, stage: str) -> Dict[str, Any]:
        """Return configuration owned by one image-pipeline stage."""
        stage_config = self.config_dict.get("pipeline", {}).get(stage, {})
        if not isinstance(stage_config, dict):
            raise ValueError(
                f"Pipeline configuration for {stage!r} must be a mapping"
            )
        return dict(stage_config)

# Global config instance for easy access
config = ConfigManager()
