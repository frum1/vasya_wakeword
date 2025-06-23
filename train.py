import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Загрузка данных
negative_features = np.load("negative_features.npy")
positive_features = np.load("vasya.npy")
X = np.vstack((negative_features, positive_features))
y = np.array([0]*len(negative_features) + [1]*len(positive_features)).astype(np.float32)[...,None]

# Разделение на тренировочный и валидационный наборы
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Расчет весов классов для функции потерь
pos_weight = len(negative_features) / len(positive_features)
neg_weight = len(positive_features) / len(negative_features)
class_weights = torch.tensor([neg_weight, pos_weight], device=device)

# Кастомная Focal Loss для бинарной классификации
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss

# Создание DataLoader
batch_size = 512
train_data = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=batch_size,
    shuffle=True
)
val_data = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
    batch_size=batch_size,
    shuffle=False
)

# Улучшенная архитектура модели
class VasyaClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim//4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.main(x)

# Инициализация модели
input_dim = X.shape[1] * X.shape[2]
model = VasyaClassifier(input_dim).to(device)

# Функция потерь и оптимизатор
criterion = FocalLoss(alpha=0.7, gamma=2.0)  # Комбинация Focal Loss и BCE
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# Ранняя остановка
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_f1):
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_f1
            self.counter = 0

# Обучение модели
n_epochs = 100
history = collections.defaultdict(list)
early_stopping = EarlyStopping(patience=10)

for epoch in tqdm(range(n_epochs), total=n_epochs):
    # Тренировочная эпоха
    model.train()
    epoch_loss = 0
    all_preds = []
    all_targets = []
    
    for batch in train_data:
        x, y_batch = batch[0].to(device), batch[1].to(device)
        
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        all_preds.append(predictions.detach().cpu())
        all_targets.append(y_batch.detach().cpu())
    
    # Сбор метрик тренировки
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    train_preds = (all_preds > 0.5).float()
    train_acc = (train_preds == all_targets).float().mean()
    train_recall = ((train_preds == 1) & (all_targets == 1)).float().sum() / (all_targets == 1).float().sum()
    train_f1 = f1_score(all_targets.numpy(), train_preds.numpy())
    
    # Валидационная эпоха
    model.eval()
    val_preds = []
    val_targets = []
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_data:
            x, y_batch = batch[0].to(device), batch[1].to(device)
            predictions = model(x)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()
            val_preds.append(predictions.cpu())
            val_targets.append(y_batch.cpu())
    
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)
    val_preds_bin = (val_preds > 0.5).float()
    val_acc = (val_preds_bin == val_targets).float().mean()
    val_recall = ((val_preds_bin == 1) & (val_targets == 1)).float().sum() / (val_targets == 1).float().sum()
    val_f1 = f1_score(val_targets.numpy(), val_preds_bin.numpy())
    
    # Логирование метрик
    history['train_loss'].append(epoch_loss/len(train_data))
    history['val_loss'].append(val_loss/len(val_data))
    history['train_acc'].append(train_acc.item())
    history['val_acc'].append(val_acc.item())
    history['train_recall'].append(train_recall.item())
    history['val_recall'].append(val_recall.item())
    history['train_f1'].append(train_f1)
    history['val_f1'].append(val_f1)
    
    # Обновление scheduler и ранняя остановка
    scheduler.step(val_f1)
    early_stopping(val_f1)
    
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break

# Визуализация результатов
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(2, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(2, 2, 3)
plt.plot(history['train_recall'], label='Train Recall')
plt.plot(history['val_recall'], label='Val Recall')
plt.legend()
plt.title('Recall')

plt.subplot(2, 2, 4)
plt.plot(history['train_f1'], label='Train F1')
plt.plot(history['val_f1'], label='Val F1')
plt.legend()
plt.title('F1 Score')

plt.tight_layout()
plt.savefig("train_metrics.png")

best_val_f1 = max(history['val_f1'])
best_val_recall = max(history['val_recall'])
best_val_acc = max(history['val_acc'])

metrics_report = f"""
## Результаты обучения

| Метрика       | Значение |
|---------------|----------|
| F1-мера       | {best_val_f1:.4f} |
| Recall        | {best_val_recall:.4f} |
| Точность      | {best_val_acc:.4f} |
| Эпох          | {epoch}/{n_epochs} |
"""

with open("metrics/training_metrics.md", "w") as f:
    f.write(metrics_report)

print(metrics_report)

# Сохранение лучшей модели
torch.save(model.state_dict(), "vasya.pth")

# Экспорт в ONNX
dummy_input = torch.zeros((1, X.shape[1], X.shape[2])).to(device)
onnx_path = "vasya.onnx"
torch.onnx.export(
    model.cpu(), 
    dummy_input.cpu(), 
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)