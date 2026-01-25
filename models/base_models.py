import torch.nn as nn

# 假设文本特征维度为 768
TEXT_FEATURE_DIM = 768
NUM_CLASSES = 3 

# 1. Text Baseline Model（仅使用文本特征进行分类的模型。）
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