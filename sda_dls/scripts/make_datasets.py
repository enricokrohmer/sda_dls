from typing import Dict
from pathlib import Path
import shutil
import logging
import json

from splitfolders import split

from sda_dls.scripts.mappings import MAPPING_MODEL_YEAR, SIMILAR_MODELS_MODEL

logger = logging.Logger(__name__)

def combine_ssb_classes(input_path: str):
    ssb_path = Path(input_path)
    
    # Move every file from src to target
    for target, src in SIMILAR_MODELS_MODEL:
        src_dir = ssb_path / src
        target_dir = ssb_path / target
        
        logger.info(
            f"Moving all files from {src_dir} to {target_dir}"
        )
        for file in src_dir.iterdir():
            file.rename(target_dir / file.name)
            
        target_dir.rmdir()
        

def rename_ccsv(input_path: str, mat_file: str):
    """
    If you download the ccsv dataset from the website, each folder is named after a
    number instead of a class name. This function renames the folders to the class
    given the .mat file provided with the dataset.
    """
    from scipy.io import loadmat
    mat = loadmat(mat_file)['sv_make_model_name']
    names = [str(m[0][0]) + ',' + str(m[1][0]) for m in mat]
    
    for i, name in enumerate(names):
        folder = Path(input_path) / str(i)
        new_folder = Path(input_path) / name
        
        logger.info(f"Renaming {folder} to {new_folder}")
        folder.rename(new_folder)


# Actual dataset maker class        
class DisjointLabelDatasetMaker:
    """
    Creates a disjoint label dataset from two datasets called src and target,
    based on a mapping file that maps classes from src to target. 
    
    The output folder will look like this:
    
    output_dir/
        src/
            train/
            val/ (Optional)
            test/
        target/
            train/
            val/ (Optional)
        target_full/
            train/
            val/
            test/
    
    src only contains classes from the source dataset that are present in the
    mapping file. target contains the classes from the target dataset that are
    not present in the mapping file. Therefore, target and src have disjoint
    label spaces. target_full contains all classes from the target dataset, which
    is used for evaluation or to train a baseline classifier.
    
    Optionally, target/val can be created by moving a fraction of the
    files from target_full/train to target/val.
    """
    
    def __init__(
        self,
        raw_src: str,
        raw_target: str,
        output_dir: str,
        class_mapping: Dict[str, str],
        label_file: str,
        src_to_target: bool = True,
        create_target_val: bool = False,
    ):
        raw_src_ = Path(raw_src)
        raw_target_ = Path(raw_target)
        
        assert raw_src_.exists() and raw_target_.exists()
        
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
            
        # Copy raw_src to output_dir
        logger.info(f"Copying {raw_src_} and {raw_target_} to {output_dir}")
        self.src_dir = output_dir / raw_src_.name
        self.target_full = output_dir / f"{raw_target_.name}_full"
        self.target_dir = output_dir / raw_target_.name
        
        shutil.copytree(raw_src_, self.src_dir)
        shutil.copytree(raw_target_, self.target_full)
        
        self.src_to_target = src_to_target
        
        if not src_to_target:
            self.src_dir, self.target_dir = self.target_dir, self.src_dir
            self.mapping = {v: k for k, v in class_mapping.items()}
        else:
            self.mapping = class_mapping
        
        self.create_target_val = create_target_val
        self.label_file = Path(label_file)
        
        if not self.label_file.exists():
            self.label_file.touch()
        
    def remove_classes_from_src(self):
        for dir_ in self.src_dir.iterdir():
            if dir_.name not in self.mapping:
                logger.info(f"Removing {dir_}")
                shutil.rmtree(dir_)
        
    def remove_classes_from_target(self):
        for dir_ in self.target_dir.iterdir():
            if dir_.name in self.mapping.values():
                logger.info(f"Removing {dir_}")
                shutil.rmtree(dir_)
                
    def make_train_test_split(self, src: bool = True, val: bool = True):
        split.ratio(
            input=self.src_dir if src else self.target_full,
            output=self.src_dir if src else self.target_full,
            ratio=(0.5, 0.2, 0.3) if val else (0.7, 0.3),
            move=True
        )
        
        if not src:
            logger.info(f"Copying {self.target_full / 'train'} to {self.target_dir / 'train'}")
            shutil.copytree(self.target_full / 'train', self.target_dir / 'train')
            
            if self.create_target_val:
                split.ratio(
                    input=self.target_dir / 'train',
                    output=self.target_dir / 'val',
                    ratio=(0.7, 0.3),
                    move=True,
                )
                
    def make_json(self):
        """
        Create json files that maps the class names to integers and vice versa.
        The key and value of the mapping file should map to the same class and
        these should occupy the first integers.
        The remaining integers should be used for the classes that are in target
        """
        label_dict = {}
        for i, (k, v) in enumerate(self.mapping.items()):
            label_dict[k] = i
            label_dict[v] = i
            
        len_ = len(label_dict)
        for class_name in self.target_dir.iterdir():
            if not class_name.is_dir():
                continue
            if class_name.name not in label_dict:
                label_dict[class_name.name] = len_
                len_ += 1
                
        with open(self.label_file, 'w') as f:
            json.dump(label_dict, f)
            
    def make(self):
        self.remove_classes_from_src()
        self.remove_classes_from_target()
        self.make_json()
        self.make_train_test_split()        
        
if __name__ == "__main__":
    # Change accordingly
    src_path = './datasets/raw/ssb'
    target_path = './datasets/raw/ccsv'
    mat_file = './datasets/raw/ccsv/names.mat'
    
    maker = DisjointLabelDatasetMaker(
        raw_src=src_path, raw_target=target_path, 
        output_dir='./datasets/ssb2ccsv/', 
        class_mapping=MAPPING_MODEL_YEAR,
        label_file='./datasets/ssb2ccsv/labels.json',
        src_to_target=False,
    )
    
    # Only required for ssb and ccsv. Remove for other pairings.
    if maker.src_to_target:
        combine_ssb_classes(maker.src_dir)
        rename_ccsv(maker.target_full, mat_file)
    else:
        combine_ssb_classes(maker.target_full)
        rename_ccsv(maker.src_dir, mat_file)
    
    maker.make()
                

    
            
        
        
