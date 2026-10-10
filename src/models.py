import torch.nn as nn
from torchvision import models
import torch


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
            
    elif name == "small_cnn":
        return SmallCNN(num_classes=num_classes)
    
    elif name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)  # type: ignore

        return model
    
    else:
        raise ValueError(f"Unsupported model (yet): {name}")

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum( p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}



class SmallCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)