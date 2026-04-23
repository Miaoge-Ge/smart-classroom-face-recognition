from models.backbones.fastcontextface import FastContextFace


def nexnet(embedding_size=512):
    return FastContextFace(embedding_size=embedding_size, depths=[2, 2, 6, 2], dims=[48, 96, 192, 384])
