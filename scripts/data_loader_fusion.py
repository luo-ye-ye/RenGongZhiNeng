import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer  
from PIL import Image 
from torchvision import transforms   
#model
MODEL_NAME = 'bert-base-chinese'  
MAX_SEQ_LENGTH = 128
IMAGE_MEAN = [0.485, 0.456, 0.406] 
IMAGE_STD = [0.229, 0.224, 0.225]
#数据集路径常量 
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
    #只加载Tokenizer 
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME) 
except Exception as e:
    print(f"Warning: Failed to load Tokenizer. Please ensure 'transformers' is installed. Error: {e}")
    TOKENIZER = None

 
#数据集 
class MultimodalDatasetFusion(Dataset):   
    def __init__(self, guids, labels=None, is_train=True):
        self.guids = guids
        self.labels = labels
        self.is_train = is_train
        if TOKENIZER is None:
             raise RuntimeError("Tokenizer failed to load. Cannot proceed with data loading.")

    def _get_raw_text(self, guid):  
        text_path = os.path.join(DATA_DIR, f"{guid}.txt")
        raw_text = ""
 
        if not os.path.exists(text_path): 
             raise FileNotFoundError(f"Required data file not found: {text_path}")
         
        try:
            with open(text_path, 'r', encoding='gb18030') as f: 
                raw_text = f.read().strip()
        except UnicodeDecodeError:  
            with open(text_path, 'r', encoding='CP866') as f:
                raw_text = f.read().strip()
        except Exception as e: 
            print(f"An unexpected error occurred while reading {text_path}: {e}")
            raise   
            
        return raw_text

    def _get_text_tokens(self, raw_text): 
        inputs = TOKENIZER(
            raw_text, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH
        ) 
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)
    
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
        
        #文本处理 
        raw_text = self._get_raw_text(guid)
        input_ids, attention_mask = self._get_text_tokens(raw_text)
        
        #图像处理 
        image_data = self._get_image_input(guid)
        
        if self.is_train:
            label = self.labels[idx]
            label_id = LABEL_MAP[label]
            label_tensor = torch.tensor(label_id, dtype=torch.long) 
            return input_ids, attention_mask, image_data, label_tensor 
        else: 
            return input_ids, attention_mask, image_data, guid
 
def get_train_val_loaders(batch_size=32, val_split=0.1):  
    df = pd.read_csv(os.path.join(TRAIN_FILE), sep=',', header=0 , dtype={'guid': str, 'tag': str})
     
    train_guids, val_guids, train_tags, val_tags = train_test_split(
        df['guid'].tolist(), 
        df['tag'].tolist(), 
        test_size=val_split, 
        random_state=42,   
    )
 
    train_dataset = MultimodalDatasetFusion(train_guids, train_tags, is_train=True)
    val_dataset = MultimodalDatasetFusion(val_guids, val_tags, is_train=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
 