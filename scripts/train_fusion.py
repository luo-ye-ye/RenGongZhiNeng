import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import os
import sys
import scripts.data_loader_fusion as dlf 
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from models.base_models import TransformerFusionClassifier  
from models.base_models import TransformerFusionClassifierpro
import scripts.data_loader  as dl
DEVICE = dl.DEVICE 
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5  

def train_and_validate_fusion(model, train_loader, val_loader): 
    model_type = 'TransformerFusion'
    model.to(DEVICE)
    print(f"Starting training for {model_type} Model on {DEVICE}...") 
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
         
        for input_ids, attention_mask, image_in, labels in train_loader: 
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            image_in = image_in.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            #前向传播 
            outputs = model(input_ids, attention_mask, image_in) 

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            #反向传播 
            loss.backward()
            optimizer.step()
            
        #验证
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for input_ids, attention_mask, image_in, labels in val_loader: 
                
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                image_in = image_in.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(input_ids, attention_mask, image_in)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Avg Loss: {total_loss/len(train_loader):.4f} - Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc 

    print(f"\n{model_type} Model Best Validation Accuracy: {best_acc:.4f}")
    return best_acc

 
if __name__ == '__main__': 
    train_loader, val_loader = dlf.get_train_val_loaders() 
    
    print("\n--- Running Transformer Cross-Attention Fusion Model ---")
     
    #fusion_model = TransformerFusionClassifier() 
     
    #train_and_validate_fusion(fusion_model, train_loader, val_loader)
     
    fusion_model = TransformerFusionClassifierpro() 
     
    train_and_validate_fusion(fusion_model, train_loader, val_loader)