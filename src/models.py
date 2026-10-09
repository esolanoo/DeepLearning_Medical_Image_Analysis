
import torch.nn as nn
from torchvision import models


def build_model(name, num_classes=8, pretrained=True):
    """
    Create a pretrained image classifier with a replaceable head.
    Supported models:
        resnet18
        efficientnet_b0
        vit_b_16
    """
    if name == "resnet18":
        weights = (models.ResNet18_Weights.DEFAULT if pretrained else None)
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        # Freeze the backbone; train the classification head.
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    elif name == "efficientnet_b0":
        weights = (models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes) # type: ignore

        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif name == "vit_b_16":
        weights = ( models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features # type: ignore
        model.heads.head = nn.Linear(in_features, num_classes) # type: ignore

        for param in model.parameters():
            param.requires_grad = False
        for param in model.heads.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unsupported model (yet): {name}")

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum( p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}