import numpy as np
import torch
from torch import nn

class BA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        
        self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, 1024)
        
        self.fc6 = nn.Linear(1024, 512)
        self.fc7 = nn.Linear(512, 2)
        
        self.relu = nn.ReLU()

    def forward(self, fp, sq):
        h1 = self.fc1(fp)
        h1 = self.relu(h1)
        h1 = self.fc2(h1)
        h1 = self.relu(h1)
        h1 = self.fc3(h1)
        
        h2 = self.fc4(sq)
        h2 = self.relu(h2)
        h2 = self.fc5(h2)
        
        h3 = h1 + h2
        h3 = self.fc6(h3)
        h3 = self.relu(h3)
        o = self.fc7(h3)
        return o
    
def run(fingerprint, sequence, n_sams):
    model = BA()
    model.load_state_dict(torch.load('BA_model_state_dict.pt'))
    model.eval()
    
    pred = model(fingerprint, sequence)
    
    ic50_noise = torch.randn(n_sams, 1)*60
    ec50_noise = torch.randn(n_sams, 1)*6 
    
    noise = torch.cat([ic50_noise, ec50_noise], axis=1)
    
    return pred + noise