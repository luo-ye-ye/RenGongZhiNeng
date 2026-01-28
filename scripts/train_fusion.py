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
import pandas as pd 
from models.base_models import NUM_CLASSES 
CHECKPOINT_PATH = 'models/best_roberta_fusion.pth' 

DEVICE = dl.DEVICE 
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5 

ID_TO_LABEL = {v: k for k, v in dlf.LABEL_MAP.items()}

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
            #ǰ�򴫲� 
            outputs = model(input_ids, attention_mask, image_in) 

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            #���򴫲� 
            loss.backward()
            optimizer.step()
            
        #��֤
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
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)  
            torch.save(model.state_dict(), CHECKPOINT_PATH) 
 

    print(f"\n{model_type} Model Best Validation Accuracy: {best_acc:.4f}")
    print(f"New best model saved with Acc: {best_acc:.4f}")
    return best_acc

def predict_and_generate_file(model, test_loader, guids, output_file='test_results.txt', checkpoint_path='models/best_model.pth'):
 
    model.to(DEVICE)
    
    #�������ģ��Ȩ��
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"Successfully loaded best model weights from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}. Using current model state.") 
    
    model.eval()
    all_predictions = []
    
    print("Starting test set prediction...")

    with torch.no_grad():
        for batch in test_loader: 
            input_ids, attention_mask, image_in, _ = batch 
             
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            image_in = image_in.to(DEVICE)
             
            outputs = model(input_ids, attention_mask, image_in)
             
            preds_id = torch.argmax(outputs, dim=1).cpu().numpy()
            preds_label = [ID_TO_LABEL[id] for id in preds_id]
            all_predictions.extend(preds_label)
 
    if len(guids) != len(all_predictions):
        raise RuntimeError("GUIDs count does not match predictions count.")
        
    results_df = pd.DataFrame({
        'guid': guids,
        'tag': all_predictions
    })
     
    results_df.to_csv(output_file, index=False, header=True, sep=',') 
    
    print(f"\nPrediction complete! Results saved to {output_file}")
 
if __name__ == '__main__': 
    train_loader, val_loader = dlf.get_train_val_loaders() 
    
    print("\n--- Running Transformer Cross-Attention Fusion Model ---")
     
    #fusion_model = TransformerFusionClassifier() 
     
    #train_and_validate_fusion(fusion_model, train_loader, val_loader)
     
    fusion_model = TransformerFusionClassifierpro() 
     
    train_and_validate_fusion(fusion_model, train_loader, val_loader)

    #Ԥ��
    fusion_model_best = TransformerFusionClassifierpro( num_classes=NUM_CLASSES,num_fusion_layers=2, text_model_name='hfl/chinese-roberta-wwm-ext')
     
    test_loader, test_guids = dlf.get_test_loader()
     
    predict_and_generate_file(
        model=fusion_model_best, 
        test_loader=test_loader, 
        guids=test_guids,
        output_file='shiyanjieguo.txt', 
        checkpoint_path='models/best_roberta_fusion.pth' 
    )