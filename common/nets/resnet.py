import torchvision.models as models
import torch.nn as nn

class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type, pretrained = True):
        super(ResNetBackbone, self).__init__()
        # Dynamically load the pretrained ResNet model
        self.model = getattr(models, f'resnet{resnet_type}')(pretrained=pretrained)
        # Replace the fully connected layer with nn.Identity
        self.model.fc = nn.Identity()

    def forward(self, x):
        # Forward pass through the modified ResNet model
        return self.model(x)
