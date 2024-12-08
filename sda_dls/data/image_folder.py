import os
import torch

from typing import List, Tuple, Optional
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose
from PIL import Image

from .utils import get_all_img_paths, LabelDict


class SingleDomainImageFolder(Dataset):
    
    def __init__(
        self,
        root : str,
        labels_path : Optional[str] = None,
        transform_list : List[torch.nn.Module] = [],
    ):
        super().__init__()
        self.root = root
        self.transforms = Compose(transform_list) if transform_list else None
        self.paths = get_all_img_paths(root)
        
        self.labels = LabelDict(labels_path) if labels_path else None
        
    def _img_path_to_label(self, path : str) -> torch.Tensor:
        normalized = os.path.normpath(path)
        parts = normalized.split(os.sep)
        class_name = parts[-2]
        
        return self.labels(class_name)
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index : int):
        path = self.paths[index % len(self.paths)]
        img = Image.open(path).convert('RGB')
        
        img = self.transforms(img) if self.transforms else img

        if self.labels:
            label = self._img_path_to_label(path)
            return img, label

        return img
        

class TwoDomainImageFolder(Dataset):
    
    def __init__(
        self,
        root_A : str,
        root_B : str,
        labels_path : Optional[str] = None,
        use_domain_labels : bool = False,
        transform_list : List[torch.nn.Module] = [],
    ):
        super().__init__()
        self.img_folder_A = SingleDomainImageFolder(root_A, labels_path, transform_list)
        self.img_folder_B = SingleDomainImageFolder(root_B, labels_path, transform_list)
        
        self.use_domain_labels = use_domain_labels
        self.get_labels = labels_path is not None
        self.len_A = len(self.img_folder_A)
        self.len_B = len(self.img_folder_B)
        
    def __len__(self) -> int:
        return self.len_A + self.len_B
    
    def __getitem__(self, index : int) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        is_domain_A = index < self.len_A
        dataset = self.img_folder_A if is_domain_A else self.img_folder_B
        
        outputs = dataset[index % len(dataset)]

        if self.use_domain_labels:
            if not self.get_labels:
                outputs = [outputs]
            domain_label = torch.zeros(1) if is_domain_A else torch.ones(1)
            outputs = *outputs, domain_label

        return outputs


class UnpairedImageFolder(Dataset):
    
    def __init__(
        self,
        root_A : str,
        root_B : str,
        labels_path : str = None,
        transform_list : List[torch.nn.Module] = [],
    ):
        super().__init__()

        self.img_folder_A = SingleDomainImageFolder(
            root_A,
            labels_path=labels_path, 
            transform_list=transform_list
        )
        self.img_folder_B = SingleDomainImageFolder(
            root_B, 
            labels_path=labels_path,
            transform_list=transform_list
        )
        
        self.use_labels = labels_path is not None
        self.len_A = len(self.img_folder_A)
        self.len_B = len(self.img_folder_B)
            
    def __len__(self) -> int:
        return max(self.len_A, self.len_B)
    
    def __getitem__(self, index : int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_labels:
            img_A, label_A = self.img_folder_A[index % self.len_A]
            img_B, label_B = self.img_folder_B[index % self.len_B]
            return img_A, img_B, label_A, label_B
        else:
            img_A = self.img_folder_A[index % self.len_A]
            img_B = self.img_folder_B[index % self.len_B]
            return img_A, img_B
    


        
        