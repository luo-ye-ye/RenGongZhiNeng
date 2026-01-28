import torch.nn as nn
from torchvision import models 
from transformers import AutoModel, AutoTokenizer
import torch
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
# 3. 早期融合模型 
class EarlyFusionClassifier(nn.Module):  
    def __init__(self, text_feature_dim=TEXT_FEATURE_DIM, 
                 image_feature_dim=IMAGE_FEATURE_DIM, 
                 num_classes=NUM_CLASSES):
        super().__init__()
        
        fused_dim = text_feature_dim + image_feature_dim  
        
        self.fused_classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, text_features, image_features): 
        fused_features = torch.cat((text_features, image_features), dim=1) 
        return self.fused_classifier(fused_features)    
# 交叉注意力模块 
class CrossAttentionBlock(nn.Module): 
    def __init__(self, d_model=TEXT_FEATURE_DIM, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
         
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, target, source): 
        attn_output, _ = self.cross_attn(
            query=target, 
            key=source, 
            value=source,
            need_weights=False
        )
         
        output = self.norm1(target + self.dropout1(attn_output))
         
        ffn_output = self.ffn(output)
        output = self.norm2(output + self.dropout2(ffn_output))
        
        return output
 
#多模态Transformer融合分类器 
class TransformerFusionClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, 
                 text_model_name='bert-base-chinese', 
                 img_feature_dim=IMAGE_FEATURE_DIM):
        super().__init__()
        
        #BERT 
        self.bert_encoder = AutoModel.from_pretrained(text_model_name)
        bert_output_dim = self.bert_encoder.config.hidden_size # 768
        
        #ResNet 
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        #将ResNet转换为输出网格特征 
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2])) 
        #维度转换
        self.image_proj = nn.Linear(512, bert_output_dim) 

        #交叉注意力融合核心 
        self.text_to_image_attn = CrossAttentionBlock(d_model=bert_output_dim)  
        self.image_to_text_attn = CrossAttentionBlock(d_model=bert_output_dim)  
         
        self.classifier = nn.Sequential(
            nn.Linear(bert_output_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, image_input):
        #文本特征提取 
        text_outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask) 
        text_features = text_outputs.last_hidden_state 
        
        #图像特征提取  
        image_grid_features = self.resnet(image_input)  
        image_features = image_grid_features.flatten(2).transpose(1, 2) 
        image_features = self.image_proj(image_features)

        #双向交叉注意力融合 
        enhanced_image_features = self.image_to_text_attn(
            target=image_features,
            source=text_features
        ) 
        enhanced_text_features = self.text_to_image_attn(
            target=text_features,
            source=image_features
        )
         
        #文本融合特征:使用增强文本序列的[CLS]token 
        text_fused_feature = enhanced_text_features[:, 0, :]  
        
        #图像融合特征:对增强图像序列进行平均池化
        image_fused_feature = torch.mean(enhanced_image_features, dim=1) 
         
        fused_vector = torch.cat((text_fused_feature, image_fused_feature), dim=1) 
        
        return self.classifier(fused_vector)
#堆叠多层交叉注意力
class FusionEncoder(nn.Module): 
    def __init__(self, d_model=768, num_layers=2, nhead=8, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        
        # 堆叠num_layers个双向交叉注意力层
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'text_to_image': CrossAttentionBlock(d_model, nhead, dropout),  
                'image_to_text': CrossAttentionBlock(d_model, nhead, dropout)  
            })
            for _ in range(num_layers)
        ])

    def forward(self, text_features, image_features):
         
        for layer in self.encoder_layers:
            #文本增强 
            new_text_features = layer['image_to_text'](
                target=text_features, 
                source=image_features
            )
            
            #图像增强  
            new_image_features = layer['text_to_image'](
                target=image_features, 
                source=text_features
            )
             
            text_features = new_text_features
            image_features = new_image_features
            
        return text_features, image_features

 
#多模态Transformer融合分类器pro 
class TransformerFusionClassifierpro(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, 
                 text_model_name='bert-base-chinese', 
                 img_feature_dim=IMAGE_FEATURE_DIM,
                 num_fusion_layers=2): #核心变化 
        super().__init__()
         
        self.bert_encoder = AutoModel.from_pretrained(text_model_name)
        bert_output_dim = self.bert_encoder.config.hidden_size # 768
         
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) 
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-2]))  
        self.image_proj = nn.Linear(512, bert_output_dim) 
 
        self.fusion_encoder = FusionEncoder(d_model=bert_output_dim, num_layers=num_fusion_layers) 
         
        self.classifier = nn.Sequential(
            nn.Linear(bert_output_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, image_input):
         
        text_outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state 
         
        image_grid_features = self.resnet(image_input) 
        image_features = image_grid_features.flatten(2).transpose(1, 2)
        image_features = self.image_proj(image_features)
 
        enhanced_text_features, enhanced_image_features = self.fusion_encoder(
            text_features=text_features, 
            image_features=image_features
        )
         
        text_fused_feature = enhanced_text_features[:, 0, :]  
        image_fused_feature = torch.mean(enhanced_image_features, dim=1) 
        
        fused_vector = torch.cat((text_fused_feature, image_fused_feature), dim=1) 
        
        return self.classifier(fused_vector)