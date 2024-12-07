from typing import List
import os
import json
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', 
    '.JPEG','.png', '.PNG', 
]

def is_img_file(filename : str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_all_img_paths(root : str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not is_img_file(filename):
                continue
            paths.append(os.path.join(dirpath, filename))
    return paths


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