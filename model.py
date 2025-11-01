import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_layers=[64, 64], activation=nn.Tanh):
        super(PINN, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_layers + [output_dim]
        
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())
        
        layers.append(nn.Linear(dims[-2], dims[-1]))  # Final output layer
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, S, t):
        return self.net(torch.cat([S, t], dim=1))
    

    
