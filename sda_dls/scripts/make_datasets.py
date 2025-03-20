from typing import Dict, Tuple, List
from pathlib import Path
import shutil
import logging
import json
import csv

from splitfolders import split

from sda_dls.scripts.mappings import MAPPING_MODEL_YEAR, SIMILAR_MODELS_MODEL

logger = logging.Logger("__main__")


def append_suffixes(path: str, suffixes: List[str] = ["train", "test", "val"]):
    return [Path(path) / suffix for suffix in suffixes]


def make_train_test(
    input_path: str,
    output_path: str,
    train_test_files: Tuple[str, str],
    make_val: bool = True,
):
    input_folder = Path(input_path)
    assert input_folder.exists()

    def copy_files(input_folder: Path, output_folder: Path, split_file: str):
        with open(split_file) as f:
            rows = csv.reader(f)
            for row in rows:
                (output_folder / row[0]).parent.mkdir(parents=True, exist_ok=True)

                shutil.copy(input_folder / row[0], output_folder / row[0])

    train = Path(output_path) / "train"
    train.mkdir(parents=True)
    copy_files(input_folder, train, train_test_files[0])

    if make_val:
        temp_dir = Path(output_path) / "temp"
        split.ratio(input=train, output=temp_dir, ratio=(0.8, 0.2), move=True)

        val = Path(output_path) / "val"
        shutil.move(temp_dir / "val", val)

        shutil.rmtree(train)
        shutil.move(temp_dir / "train", Path(output_path))

        temp_dir.rmdir()

    test = Path(output_path) / "test"
    test.mkdir(parents=True)
    copy_files(input_folder, test, train_test_files[1])


def _combine_ssb_classes(input_path: str):
    input_path = Path(input_path)

    # Move every file from src to target
    for target, src in SIMILAR_MODELS_MODEL:
        src_dir = input_path / src
        target_dir = input_path / target

        logger.info(f"Moving all files from {src_dir} to {target_dir}")
        for file in src_dir.iterdir():
            file.rename(target_dir / file.name)

        src_dir.rmdir()


def combine_ssb(src_dir: str, target_dir: str):
    for p in append_suffixes(src_dir, ["train", "val", "test"]):
        _combine_ssb_classes(p)


def _rename_ccsv(input_path: str, mat_file: str):
    """
    If you download the ccsv dataset from the website, each folder is named after a
    number instead of a class name. This function renames the folders to the class
    given the .mat file provided with the dataset.
    """
    from scipy.io import loadmat

    mat = loadmat(mat_file)["sv_make_model_name"]
    names = [str(m[0][0]) + "," + str(m[1][0]) for m in mat]

    for i, name in enumerate(names):
        folder = Path(input_path) / str(i + 1)
        new_folder = Path(input_path) / name

        logger.info(f"Renaming {folder} to {new_folder}")
        folder.rename(new_folder)


def rename_ccsv(input_path: str, mat_file: str):
    for p in append_suffixes(input_path, ["train", "val", "test"]):
        _rename_ccsv(p, mat_file)


class DisjointLabelDatasetMaker:
    """
    Creates a disjoint label dataset from two datasets called src and target,
    based on a mapping file that maps classes from src to target.

    The input folder structure is expected to look like this

    input_dir/
        src/
            train/
            val/
            test/
        target/
            train/
            val/
            test/

    The output folder will look like this:

    output_dir/
        src/
            train/
            test/
        target/
            train/
        target_split/
            train/
            val/
        target_full/
            train/
            val/
            test/
        label_file.json

    src only contains classes from the source dataset that are present in the
    mapping file. target contains the classes from the train split of the target
    dataset that are not present in the mapping file. Therefore, target and src
    have disjoint label spaces. target_full contains all classes from the target
    dataset, which is used for evaluation or to train a baseline classifier.
    target_split is the same as target/train, but with a train/val split.
    Latter is used for two-step training.
    """

    def __init__(
        self,
        raw_src: str,
        raw_target: str,
        output_dir: str,
        class_mapping: Dict[str, str],
        src_to_target: bool = True,
    ):
        raw_src_ = Path(raw_src)
        raw_target_ = Path(raw_target)

        if not src_to_target:
            raw_src_, raw_target_ = raw_target_, raw_src_

        assert raw_src_.exists() and raw_target_.exists()

        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

        # Copy raw_src to output_dir
        self.src_dir = output_dir / "source"
        shutil.copytree(raw_src_, self.src_dir)

        # Copy raw_target to output_dir
        self.target_full = output_dir / "target_full"
        shutil.copytree(raw_target_, self.target_full)

        # Copy raw_target to output_dir
        self.target_dir = output_dir / "target/train"
        shutil.copytree(raw_target_ / "train", self.target_dir)

        self.src_to_target = src_to_target

        # Invert mapping if necessary
        if not src_to_target:
            self.mapping = {v: k for k, v in class_mapping.items()}
        else:
            self.mapping = class_mapping

        # Create label file
        self.label_file = Path(output_dir) / "labels.json"
        if not self.label_file.exists():
            self.label_file.touch()

    def remove_classes_from_src(self):
        for p in append_suffixes(self.src_dir):
            for dir_ in p.iterdir():
                if dir_.name not in self.mapping:
                    logger.info(f"Removing {dir_}")
                    shutil.rmtree(dir_)

    def remove_classes_from_target(self):
        for dir_ in self.target_dir.iterdir():
            if dir_.name in self.mapping.values():
                logger.info(f"Removing {dir_}")
                shutil.rmtree(dir_)

    def make_target_split(self):
        split.ratio(
            input=self.target_dir,
            output=Path(self.target_dir).parent.parent / "target_split",
            ratio=(0.8, 0.2),
            move=False,
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

        len_ = len(label_dict) / 2
        for class_name in self.target_dir.iterdir():
            if not class_name.is_dir():
                continue
            if class_name.name not in label_dict:
                label_dict[class_name.name] = len_
                len_ += 1

        with open(self.label_file, "w") as f:
            json.dump(label_dict, f)

    def make(self):
        self.remove_classes_from_src()
        self.remove_classes_from_target()
        self.make_target_split()
        self.make_json()


if __name__ == "__main__":
    # Change accordingly
    src_path = "./datasets/raw/ssb"
    src_train_test = ("./datasets/raw/train_ssb.csv", "./datasets/raw/test_ssb.csv")

    target_path = "./datasets/raw/ccsv"
    mat_file = "./datasets/raw/ccsv.mat"
    target_train_test = (
        "./datasets/raw/train_ccsv.txt",
        "./datasets/raw/test_ccsv.txt",
    )

    src_to_target = True

    make_train_test(
        input_path=src_path,
        output_path="./datasets/formatted/ssb",
        train_test_files=src_train_test,
        make_val=True,
    )
    combine_ssb("./datasets/formatted/ssb", "./datasets/formatted/ssb")

    make_train_test(
        input_path=target_path,
        output_path="./datasets/formatted/ccsv",
        train_test_files=target_train_test,
        make_val=True,
    )
    rename_ccsv("./datasets/formatted/ccsv", mat_file)

    DisjointLabelDatasetMaker(
        raw_src="./datasets/formatted/ssb",
        raw_target="./datasets/formatted/ccsv",
        output_dir="./datasets/ssb2ccsv/",
        class_mapping=MAPPING_MODEL_YEAR,
        src_to_target=True,
    ).make()

    DisjointLabelDatasetMaker(
        raw_src="./datasets/formatted/ssb",
        raw_target="./datasets/formatted/ccsv",
        output_dir="./datasets/ccsv2ssb/",
        class_mapping=MAPPING_MODEL_YEAR,
        src_to_target=False,
    ).make()
