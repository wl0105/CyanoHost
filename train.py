import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch.utils.data.sampler import WeightedRandomSampler
import sys
import random
from collections import Counter

class classify_net(nn.Module):
    def __init__(self, input_dim):
        super(classify_net, self).__init__()
        self.input_dim = input_dim

        self.conv1 = SAGEConv(self.input_dim, 1550)
        self.conv2 = SAGEConv(1550, 512)
        self.fc = nn.Linear(512, 50)
        self.bn1 = nn.BatchNorm1d(num_features=1550)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.8, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.fc(x) 
        return F.log_softmax(x, dim=1)

def train(dataset, model, batch_size, learning_rate=1e-4, epoch_n=100, random_seed=111, val_split=0.2, weighted_sampling=False, model_name="comb_model.pt", device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    random.seed(random_seed)
    data_list = list(range(0, len(dataset)))
    test_list = random.sample(data_list, int(len(dataset) * val_split))
    train_dataset = [dataset[i] for i in data_list if i not in test_list]
    val_dataset = [dataset[i] for i in data_list if i in test_list]

    if weighted_sampling:
        label_count = Counter([int(data.y) for data in dataset])
        weights = [100/label_count[int(data.y)] for data in train_dataset]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model = None
    best_val_acc = 0
    
    for epoch in range(epoch_n):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (pred.argmax(dim=1) == batch.y).float().mean().item()
        
        val_acc = evaluation(val_loader, model, device)
        if val_acc > best_val_acc:
            torch.save(model, model_name)
            best_model = model
            best_val_acc = val_acc
            
        print(f"Epoch {epoch}| Loss: {train_loss/len(train_loader):.4f}| "
              f"Train accuracy: {train_acc/len(train_loader):.4f}| "
              f"Validation accuracy: {val_acc:.4f}")

    return best_model, best_val_acc

def evaluation(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    
    return correct / total

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    print("Dataset Loading")
    train_label_path = sys.argv[1]
    pre_data_pt = sys.argv[2]
    
    pre_dataset = []
    with open(train_label_path, 'r') as rf:
        for line in rf:
            pt_path = pre_data_pt + line.split('\t')[0] + '.pt'
            pre_dataset.append(torch.load(pt_path))
    
    val_split = 0.2
    data_list = list(range(0, len(pre_dataset)))
    test_list = random.sample(data_list, int(len(pre_dataset) * val_split))
    train_dataset = [pre_dataset[i] for i in data_list if i not in test_list]
    test_dataset = [pre_dataset[i] for i in data_list if i in test_list]
    
    del pre_dataset

    f_dim = train_dataset[0].x.shape[1]
    model = classify_net(input_dim=f_dim).to(device)
    
    best_model, best_val_acc = train(train_dataset, model, 64, model_name="model.pt")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    test_acc = evaluation(test_loader, best_model, device)
    
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("Done!")



    
