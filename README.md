import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import numpy as np

# 配置参数
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
MODEL_NAME = 'bert-base-chinese'

# 示例数据集（替换为实际数据）
texts = [
    "帮我画一个昏暗的房间",
    "查询明天的天气",
    "播放周杰伦的歌曲",
    "预定北京到上海的机票"
]
labels = [0, 1, 2, 3]  # 对应不同意图的编号
label_names = ["IMAGE_GENERATE", "WEATHER_QUERY", "MUSIC_PLAY", "FLIGHT_BOOK"]

# 划分训练集验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42)

# 自定义数据集类
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 初始化组件
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_names))

# 创建DataLoader
train_dataset = IntentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = IntentDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 训练准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练循环
best_accuracy = 0
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    model.eval()
    val_accuracy = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            val_accuracy += torch.sum(preds == labels).item()
    
    # 保存最佳模型
    val_accuracy = val_accuracy / len(val_dataset)
    if val_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_model.bin')
        best_accuracy = val_accuracy
    
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'Train Loss: {total_loss/len(train_loader):.4f}')
    print(f'Val Acc: {val_accuracy:.4f}\n')

# 预测函数
def predict_intent(text):
    model.load_state_dict(torch.load('best_model.bin', map_location=device))
    model.eval()
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
    return label_names[prediction]

# 测试预测
test_text = "帮我画一只会飞的猫"
print(f"预测结果: {predict_intent(test_text)}")
