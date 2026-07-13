from typing import Dict, Type, Any

class ModelRegistry:
    """
    Registry Pattern để quản lý và tự động tìm kiếm các model classes.
    """
    _registry: Dict[str, Dict[str, Type]] = {
        "matting": {},
        "inpainting": {},
        "segmentation": {}
    }

    @classmethod
    def register(cls, category: str, name: str):
        """
        Đăng ký một model class vào registry.
        """
        def wrapper(wrapped_class: Type):
            if category not in cls._registry:
                cls._registry[category] = {}
            cls._registry[category][name] = wrapped_class
            return wrapped_class
        return wrapper

    @classmethod
    def get_class(cls, category: str, name: str) -> Type:
        """
        Lấy class tương ứng với name trong category.
        """
        if category not in cls._registry:
            raise ValueError(f"Category '{category}' is not supported.")
        if name not in cls._registry[category]:
            raise ValueError(f"Model '{name}' not found in category '{category}'. Available: {list(cls._registry[category].keys())}")
        
        return cls._registry[category][name]
