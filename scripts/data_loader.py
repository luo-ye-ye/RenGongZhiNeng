import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel  
# 假设您使用一个基础的中文 BERT 模型
MODEL_NAME = 'bert-base-chinese' 
# 或者 'hfl/chinese-roberta-wwm-ext' 等
MAX_SEQ_LENGTH = 128

# 定义数据集路径常量 (与您的目录结构一致)
DATA_DIR = 'data'
TRAIN_FILE = 'train.txt'
LABEL_MAP = {'positive': 0, 'neutral': 1, 'negative': 2}

# 在模块级别初始化模型和分词器，避免重复加载
# 注意：在实际训练中，如果内存不足，需要考虑将模型和分词器封装在单例模式或只在需要时加载
try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 使用 AutoModel 提取特征，只加载编码器部分
    # 注意：这里我们是冻结特征提取，只训练分类头
    BERT_ENCODER = AutoModel.from_pretrained(MODEL_NAME)
    BERT_ENCODER.eval() # 特征提取时设置为评估模式
    # 冻结 BERT 权重
    for param in BERT_ENCODER.parameters():
        param.requires_grad = False
except Exception as e:
    print(f"Warning: Failed to load BERT model. Using dummy features. Error: {e}")
    BERT_ENCODER = None


class MultimodalDataset(Dataset):
    """
    数据集类。在提交 A 中只处理文本。
    """
    def __init__(self, guids, labels=None, is_train=True):
        self.guids = guids
        self.labels = labels
        self.is_train = is_train
        
    def _get_text_feature(self, guid):
        """ 
        实现真正的 BERT 特征提取。
        """
        if BERT_ENCODER is None:
            # 如果模型加载失败，使用模拟特征（这是最后的保障）
            from models.base_models import TEXT_FEATURE_DIM
            return torch.randn(TEXT_FEATURE_DIM) 

        text_path = os.path.join(DATA_DIR, f"{guid}.txt")
        raw_text = ""

        # 核心：确保文件存在，并使用多重尝试处理编码
        if not os.path.exists(text_path):
             # 如果文件确实不存在，直接抛出错误
             raise FileNotFoundError(f"Required data file not found: {text_path}")
        # 1. 读取原始文本
        try:
            with open(text_path, 'r', encoding='gb18030') as f: # <--- 关键修改
                raw_text = f.read().strip()
        except UnicodeDecodeError:
            # 如果 GBK 也不行，可以尝试 'gb18030' 或 'latin-1'
            print(f"Warning: GBK failed for {guid}.txt. Trying utf-8...")
            with open(text_path, 'r', encoding='CP866') as f:
                raw_text = f.read().strip()
        except Exception as e:
            # 捕获其他任何文件读取错误（如权限问题等）
            print(f"An unexpected error occurred while reading {text_path}: {e}")
            raise # 重新抛出异常
        # 2. Tokenize 和编码
        inputs = TOKENIZER(
            raw_text, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH
        )
        
        # 3. 通过 BERT 提取特征 (在 eval 模式下，不需要梯度)
        with torch.no_grad():
            inputs = inputs.to(DEVICE)
            # inputs['input_ids'] 是 (1, MAX_SEQ_LENGTH)
            outputs = BERT_ENCODER(
                 
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask']
            )
            # 提取 [CLS] 对应的特征 (Batch=1, CLS=0)
            cls_feature = outputs.last_hidden_state[0, 0, :] 
            
        return cls_feature
    
    def __len__(self):
        return len(self.guids)

    def __getitem__(self, idx):
        # ... (与之前一致，返回 text_data, label_tensor)
        guid = self.guids[idx]
        text_data = self._get_text_feature(guid)
        
        if self.is_train:
            label = self.labels[idx]
            label_id = LABEL_MAP[label]
            label_tensor = torch.tensor(label_id, dtype=torch.long)
            return text_data, label_tensor
        else:
            return text_data, guid

# 核心函数：加载数据并划分 (与之前一致)
def get_train_val_loaders(batch_size=32, val_split=0.1):
    df = pd.read_csv(os.path.join(TRAIN_FILE), sep=',', header=0 , dtype={'guid': str, 'tag': str})
    
    # 划分数据集 (使用 sklearn 的 train_test_split 保证可复现)
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