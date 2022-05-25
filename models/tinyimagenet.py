# ===========================================================================
# Project:      Compression-aware Training of Neural Networks using Frank-Wolfe
# File:         models_pytorch/tinyimagenet.py
# Description:  TinyImagenet Models
# ===========================================================================
import torchvision


def ResNet50():
    return torchvision.models.resnet50(pretrained=False, num_classes=200)
