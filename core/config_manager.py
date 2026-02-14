import yaml
import os
import threading

_config_lock = threading.Lock()
_global_config_instance = None


class Config:
    def __init__(self, config_path="config/config.yaml"):
        self._config_path = config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
            
        self._load_config()
            
    def _load_config(self):
        with open(self._config_path, 'r', encoding='utf-8') as f:
            self._cfg = yaml.safe_load(f) or {}
            
    @property
    def recognition(self):
        # 兼容旧配置 key 'model' -> 'recognition'
        return self._cfg.get('recognition', self._cfg.get('model', {}))
        
    @property
    def detector(self):
        return self._cfg.get('detector', {})
        
    @property
    def preprocess(self):
        return self._cfg.get('preprocess', {})


def get_global_config():
    global _global_config_instance
    with _config_lock:
        if _global_config_instance is None:
            try:
                _global_config_instance = Config()
            except Exception as e:
                import logging
                logging.getLogger("systems.config").warning(f"Failed to load config: {e}")
        return _global_config_instance


# 向后兼容
try:
    global_config = get_global_config()
except Exception as e:
    import logging
    logging.getLogger("systems.config").warning(f"Warning: Failed to load config: {e}")
    global_config = None
