import torch
import torch.nn as nn
import torch.nn.functional as F

class GlutenSubstitutionNet(nn.Module):
    """
    MLP with two outputs:
     - gluten flag logits (2 classes)
     - substitute logits (num_substitutes classes, 0 means 'no substitute')
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_substitutes: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_gluten = nn.Linear(hidden_dim, 2)
        self.out_substitute = nn.Linear(hidden_dim, num_substitutes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        gluten_logits = self.out_gluten(x)
        substitute_logits = self.out_substitute(x)
        return gluten_logits, substitute_logits

