from torchvision.models import resnet50
import torch.nn as nn
import torch
from collections import OrderedDict

class Classifier(nn.Module):
    def __init__(self, num_class, dropout_rate=0.5):
        super().__init__()
        self.drop_out = nn.Dropout(dropout_rate)  # Dropout layer for regularization
        self.linear = nn.Linear(2048, num_class)  # Linear layer for classification

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.drop_out(x)  # Apply dropout
        x = self.linear(x)  # Apply the linear layer
        return x

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50()  # Load the pre-trained ResNet-50 model
        encoder_layers = list(base_model.children())  # Extract the layers of the base model
        self.backbone = nn.Sequential(*encoder_layers[:9])

    def forward(self, x):
        return self.backbone(x)  # Forward pass through the backbone

def split_state_dict(state_dict):
    backbone_state_dict = OrderedDict()  # State dict for the backbone
    classifier_state_dict = OrderedDict()  # State dict for the classifier
    
    for key, value in state_dict.items():
        if key.startswith('0.backbone'):
            new_key = key.replace('0.', '', 1)  # Adjust the key for backbone
            backbone_state_dict[new_key] = value
        elif key.startswith('1.linear'):
            new_key = key.replace('1.', '', 1)  # Adjust the key for classifier
            classifier_state_dict[new_key] = value

    return backbone_state_dict, classifier_state_dict

def get_resnet_model(model_load, num_classes=2, freeze_backbone=False, own_model=False):
    backbone = Backbone()  # Initialize the backbone
    classifier = Classifier(num_classes)  # Initialize the classifier

    if own_model:
        state_dict = torch.load(model_load)  # Load the custom model state dict
        backbone_state_dict, classifier_state_dict = split_state_dict(state_dict)  # Split the state dict

        backbone.load_state_dict(backbone_state_dict)  # Load the backbone state dict
        classifier.load_state_dict(classifier_state_dict)  # Load the classifier state dict
    else:
        backbone.load_state_dict(torch.load(model_load))  # Load the state dict directly if not custom
    
    if freeze_backbone:
        # Freeze all layers in the backbone
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last layer of the backbone for fine-tuning
        for param in backbone.backbone[-1].parameters():
            param.requires_grad = True

    net = nn.Sequential(backbone, classifier)  # Combine the backbone and classifier

    return net
