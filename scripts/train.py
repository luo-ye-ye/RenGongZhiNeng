import torch 
import torch.nn as nn
from models.base_models import TextClassifier 
import scripts.data_loader as dl
from sklearn.metrics import accuracy_score
import numpy as np
from models.base_models import TextClassifier, ImageClassifier 
from models.base_models import TextClassifier, ImageClassifier, EarlyFusionClassifier
#核心函数 
def train_and_validate(model, train_loader, val_loader, model_type='Text'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    print(f"Starting training for {model_type} Model on {device}...") # 确保这里打印出 'cuda'
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    num_epochs = 10 
    image_extractor = None
    if model_type == 'EarlyFusion':
         
        temp_img_model = ImageClassifier()  
         
        image_extractor = temp_img_model.resnet 
        image_extractor.eval()
        image_extractor.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0 
         
         
        for text_f, image_in, labels in train_loader: 
             
            text_f, image_in, labels = text_f.to(device), image_in.to(device), labels.to(device)
            
            optimizer.zero_grad()
             
            if model_type == 'Text':
                outputs = model(text_f)  
            elif model_type == 'Image':
                outputs = model(image_in)
            elif model_type == 'EarlyFusion':  
                with torch.no_grad():  
                    image_features_4D = image_extractor(image_in) 
                outputs = model(text_f, image_features_4D)  
            else: 
                raise ValueError(f"Unknown model type: {model_type}")

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for text_f, image_in, labels in val_loader: 
                
                text_f, image_in, labels = text_f.to(device), image_in.to(device), labels.to(device)
                
                if model_type == 'Text':
                    outputs = model(text_f)
                elif model_type == 'Image':  
                    outputs = model(image_in)
                elif model_type == 'EarlyFusion':  
                    image_features_2D = image_extractor(image_in) 
             
                    outputs = model(text_f, image_features_2D)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {total_loss/len(train_loader):.4f} - Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            
    print(f"\n{model_type} Model Best Validation Accuracy: {best_acc:.4f}")
    return best_acc


# 主运行逻辑
if __name__ == '__main__':
    train_loader, val_loader = dl.get_train_val_loaders()

     
     #3. 早期融合基线 
    print("\n--- Running Early Fusion Baseline (Ablation C) ---")
    early_fusion_model = EarlyFusionClassifier()
    train_and_validate(early_fusion_model, train_loader, val_loader, model_type='EarlyFusion')