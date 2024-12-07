import json
import os
import torch

class LabelDict:
    
    def __init__(self, label_path : str):
        if not os.path.exists(label_path):
            raise FileNotFoundError("Label file not found.")
        elif not label_path.endswith('.json'):
            raise ValueError("Label file must be a json file.")
        
        
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        
    def __call__(self, class_name : str) -> torch.Tensor:
        if not class_name in self.label_dict:
            raise ValueError(f"{class_name} not found in label file.")
        
        index = self.label_dict[class_name] - 1
        label = torch.tensor(index, dtype=torch.long)
        return label