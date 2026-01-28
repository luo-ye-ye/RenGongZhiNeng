import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from PIL import Image 
from torchvision import transforms   
MODEL_NAME = 'bert-base-chinese'  
MAX_SEQ_LENGTH = 128
IMAGE_MEAN = [0.485, 0.456, 0.406] 
IMAGE_STD = [0.229, 0.224, 0.225]
# 数据集路径常量 
DATA_DIR = 'data'
TRAIN_FILE = 'train.txt'
LABEL_MAP = {'positive': 0, 'neutral': 1, 'negative': 2}
#图像处理
IMAGE_TRANSFORMS = transforms.Compose([ 
    transforms.Resize((224, 224)),       
    transforms.ToTensor(),             
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)  
])  
try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME) 
    BERT_ENCODER = AutoModel.from_pretrained(MODEL_NAME)
    BERT_ENCODER.to(DEVICE)
    BERT_ENCODER.eval()   
    for param in BERT_ENCODER.parameters():
        param.requires_grad = False
except Exception as e:
    print(f"Warning: Failed to load BERT model. Using dummy features. Error: {e}")
    BERT_ENCODER = None


class MultimodalDataset(Dataset):  
    #数据集类。在提交 A 中只处理文本。 
    def __init__(self, guids, labels=None, is_train=True):
        self.guids = guids
        self.labels = labels
        self.is_train = is_train
        
    def _get_text_feature(self, guid): 
        #实现BERT特征提取。 
        if BERT_ENCODER is None: 
            from models.base_models import TEXT_FEATURE_DIM
            return torch.randn(TEXT_FEATURE_DIM) 

        text_path = os.path.join(DATA_DIR, f"{guid}.txt")
        raw_text = ""
 
        if not os.path.exists(text_path): 
             raise FileNotFoundError(f"Required data file not found: {text_path}")
        try:
            with open(text_path, 'r', encoding='gb18030') as f: 
                raw_text = f.read().strip()
        except UnicodeDecodeError: 
            print(f"Warning: GBK failed for {guid}.txt. Trying utf-8...")
            with open(text_path, 'r', encoding='CP866') as f:
                raw_text = f.read().strip()
        except Exception as e: 
            print(f"An unexpected error occurred while reading {text_path}: {e}")
            raise   
        inputs = TOKENIZER(
            raw_text, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH
        )
         
        with torch.no_grad():
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
             
            outputs = BERT_ENCODER(
                input_ids=inputs['input_ids'].to(DEVICE), 
                attention_mask=inputs['attention_mask'].to(DEVICE)
            )
            cls_feature = outputs.last_hidden_state[0, 0, :] 
            
        return cls_feature
    def _get_image_input(self, guid):   
        image_path = os.path.join(DATA_DIR, f"{guid}.jpg")
        
        if not os.path.exists(image_path):
             raise FileNotFoundError(f"Required data image not found: {image_path}")
              
        image = Image.open(image_path).convert('RGB')
         
        image_tensor = IMAGE_TRANSFORMS(image)
        
        return image_tensor
    def __len__(self):
        return len(self.guids)

    def __getitem__(self, idx): 
        guid = self.guids[idx]
        text_data = self._get_text_feature(guid)
        image_data = self._get_image_input(guid)
        if self.is_train:
            label = self.labels[idx]
            label_id = LABEL_MAP[label]
            label_tensor = torch.tensor(label_id, dtype=torch.long)
            return text_data,image_data,label_tensor
        else:
            return text_data,image_data, guid

#加载数据并划分 
def get_train_val_loaders(batch_size=32, val_split=0.1):
    df = pd.read_csv(os.path.join(TRAIN_FILE), sep=',', header=0 , dtype={'guid': str, 'tag': str})
     
    train_guids, val_guids, train_tags, val_tags = train_test_split(
        df['guid'].tolist(), 
        df['tag'].tolist(), 
        test_size=val_split, 
        random_state=42,   
    )

    train_dataset = MultimodalDataset(train_guids, train_tags, is_train=True)
    val_dataset = MultimodalDataset(val_guids, val_tags, is_train=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader