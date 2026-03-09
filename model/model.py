import torch.optim as optim
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_HAR_KS(nn.Module):
    def __init__(self):
        super(CNN_HAR_KS, self).__init__()
        
        # --- Layer 1: Convolution ---
        # Input: (Batch, 1, 16, 16)
        # Output: (Batch, 32, 16, 16)
        # Padding=1 ensures the size remains 16x16 with a 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        # --- Layer 2: Convolution ---
        # Input: (Batch, 32, 16, 16)
        # Output: (Batch, 64, 16, 16)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # --- Layer 3: Max Pooling ---
        # Input: (Batch, 64, 16, 16)
        # Output: (Batch, 64, 8, 8) -> Flat features: 64 * 8 * 8 = 4096
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Layer 4: Dropout ---
        # Paper Appendix C: Dropout = 0.3
        self.dropout = nn.Dropout(p=0.3)
        
        # --- Layer 5: Fully Connected (MLP) ---
        # Input: 4096 flattened features
        # Output: 64
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.relu3 = nn.ReLU()
        
        # --- Layer 6: Output ---
        # Output: 2 (Class 0: Volatility Increase, Class 1: Decrease)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # Convolution Block
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        
        # Regularization
        x = self.dropout(x)
        
        # Flattening: (Batch, 64, 8, 8) -> (Batch, 4096)
        x = x.view(x.size(0), -1)
        
        # Dense Layers
        x = self.relu3(self.fc1(x))
        
        # Output Logits
        # Note: We return raw logits. nn.CrossEntropyLoss will handle Softmax internally.
        x = self.fc2(x)
        return x


def train_model(model, train_loader, test_loader, num_epochs=50, device='cpu'):
    """
    训练 CNN_HAR_KS 模型的完整循环
    """
    model = model.to(device)
    
    # --- Hyperparameters from Paper Appendix C ---
    # Optimizer: Adam (standard choice)
    # Learning Rate: 0.001 (search space start)
    # L2 Regularization (Weight Decay): 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    
    # Loss Function: Cross Entropy (Standard for classification)
    criterion = nn.CrossEntropyLoss()
    
    # Learning Rate Scheduler: Reduce LR on Plateau
    # 论文提到: "Reduce LR On Plateau, min lr = 0.0001"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, min_lr=0.0001, #verbose=True
    )

    print(f"Starting training on {device}...")
    
    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        model.train() # Set to training mode (enables Dropout)
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 1. Zero gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            outputs = model(images)
            
            # 3. Calculate Loss
            loss = criterion(outputs, labels)
            
            # 4. Backward pass
            loss.backward()
            
            # 5. Optimization step
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train
        
        # --- Validation Phase ---
        test_acc, test_loss = evaluate_model(model, test_loader, criterion, device)
        
        # Update Scheduler based on Test Loss
        scheduler.step(test_loss)
        
        history['train_loss'].append(epoch_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    return history

def evaluate_model(model, data_loader, criterion, device):
    """
    评估模型在测试集上的表现
    """
    model.eval() # Set to evaluation mode (disables Dropout)
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad(): # No need to track gradients
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = correct / total
    return accuracy, avg_loss
# 实例化并打印模型结构以验证
if __name__ == "__main__":
    model = CNN_HAR_KS()
    # 假设输入一个 Batch size 为 32 的数据
    dummy_input = torch.randn(32, 1, 16, 16)
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}") # Should be [32, 2]
    print(model)