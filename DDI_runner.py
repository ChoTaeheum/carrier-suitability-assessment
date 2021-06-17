import numpy as np
import torch
from torch import nn

class DDI(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 588)
        
        self.relu = nn.ReLU()

    def forward(self, xb):
        h = self.fc1(xb)
        h = self.relu(h)
        h = self.fc2(h)
        h = self.relu(h)
        o = self.fc3(h)
        return o
    
def run(drug_fp, carrier_fp):
    model = DDI()
    model.load_state_dict(torch.load('DDI_model_state_dict.pt'))
    model.eval()
    
    combi_data = carrier_fp + torch.tensor(drug_fp, dtype=torch.float32)
    pred = model(combi_data)
    
    return torch.argmax(pred, axis=1)