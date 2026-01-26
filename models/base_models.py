import torch.nn as nn
from torchvision import models 
 
TEXT_FEATURE_DIM = 768
IMAGE_FEATURE_DIM = 512  
NUM_CLASSES = 3 

# 1.文本基线模型。
class TextClassifier(nn.Module): 
    def __init__(self, feature_dim=TEXT_FEATURE_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, text_features):
        return self.fc(text_features)
# 2. 图像基线模型 
class ImageClassifier(nn.Module):  
    def __init__(self, feature_dim=IMAGE_FEATURE_DIM, num_classes=NUM_CLASSES):
        super().__init__() 
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
         
        for param in self.resnet.parameters():
            param.requires_grad = False
         
        num_ftrs = self.resnet.fc.in_features  
        self.resnet.fc = nn.Identity()
 
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, feature_dim),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, image_input): 
        features = self.resnet(image_input)
        return self.classifier(features)
        