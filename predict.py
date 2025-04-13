import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
import sys
import os

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

def predict(model, data_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred = model(batch)
            pred_class = pred.argmax(dim=1)
            predictions.extend(pred_class.cpu().numpy().tolist())
    
    return predictions

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python predict.py <model_path> <data_path> <output_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    output_path = sys.argv[3]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    # Load model
    print("Loading model...")
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    
    # Load prediction data
    print("Loading prediction data...")
    predict_dataset = []
    with open(data_path, 'r') as rf:
        for line in rf:
            pt_path = line.strip() + '.pt'
            if os.path.exists(pt_path):
                predict_dataset.append(torch.load(pt_path))
            else:
                print(f"Warning: File {pt_path} does not exist")
    
    if not predict_dataset:
        print("Error: No valid prediction data found")
        sys.exit(1)
    
    # Create data loader
    predict_loader = DataLoader(predict_dataset, batch_size=64, shuffle=False)
    
    # Make predictions
    print("Starting prediction...")
    predictions = predict(model, predict_loader, device)
    
    # Save prediction results
    print("Saving prediction results...")
    with open(output_path, 'w') as wf:
        for i, pred in enumerate(predictions):
            wf.write(f"{i}\t{pred}\n")
    
    print(f"Prediction completed! Results saved to {output_path}") 