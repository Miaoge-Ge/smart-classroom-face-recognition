import torch.nn as nn
from models.backbones.resnet import ResNet10, ResNet50
from models.backbones.nexnet import nexnet

class ModelFactory:
    """
    模型工厂：负责创建 Backbone 实例
    """
    _creators = {
        'resnet10': ResNet10,
        'resnet50': ResNet50,
        'nexnet': nexnet,
        'fastcontextface': nexnet,
    }

    @classmethod
    def create_backbone(cls, backbone_type, embedding_size=512):
        backbone_type = backbone_type.lower()
        if backbone_type not in cls._creators:
            raise ValueError(f"不支持的 Backbone 类型: {backbone_type}. 可选: {list(cls._creators.keys())}")
        
        creator = cls._creators[backbone_type]
        return creator(embedding_size=embedding_size)
