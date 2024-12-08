from typing import Optional
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_set: Dataset,
        batch_size_train: int,
        num_workers_train: int,
        val_set: Dataset,
        batch_size_val: int,
        num_workers_val: int,
        test_set: Optional[Dataset] = None,
        batch_size_test: Optional[int] = None,
        num_workers_test: Optional[int] = None,
    ):
        super().__init__()
        self.train_set = train_set
        self.batch_size_train = batch_size_train
        self.num_workers_train = num_workers_train
        self.val_set = val_set
        self.batch_size_val = batch_size_val
        self.num_workers_val = num_workers_val
        self.test_set = test_set
        self.batch_size_test = batch_size_test
        self.num_workers_test = num_workers_test

    def setup(self, stage):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers_train,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size_val,
            shuffle=False,
            num_workers=self.num_workers_val,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if any(
            x is None
            for x in [self.test_set, self.batch_size_test, self.num_workers_test]
        ):
            raise ValueError("No test set provided")

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=self.num_workers_test,
            pin_memory=True,
        )


class TwoStageDataModule(BaseDataModule):
    def __init__(
        self,
        switch_epoch: int,
        first_train_set: Dataset,
        snd_train_set: Dataset,
        batch_size_train_first: int,
        batch_size_train_snd: int,
        num_workers_train: int,
        val_set: Dataset,
        batch_size_val: int,
        num_workers_val: int,
        test_set: Optional[Dataset] = None,
        batch_size_test: Optional[int] = None,
        num_workers_test: Optional[int] = None,
    ):
        super().__init__(
            train_set=first_train_set,
            batch_size_train=batch_size_train_first,
            num_workers_train=num_workers_train,
            val_set=val_set,
            batch_size_val=batch_size_val,
            num_workers_val=num_workers_val,
            test_set=test_set,
            batch_size_test=batch_size_test,
            num_workers_test=num_workers_test,
        )
        self.snd_train_set = snd_train_set
        self.batch_size_train_snd = batch_size_train_snd
        self.switch_epoch = switch_epoch

    def train_dataloader(self) -> DataLoader:
        if self.trainer.current_epoch < self.switch_epoch:
            return super().train_dataloader()
        else:
            print("SWITCH EPOCH \n")
            return DataLoader(
                self.snd_train_set,
                batch_size=self.batch_size_train_snd,
                shuffle=True,
                num_workers=self.num_workers_train,
                pin_memory=True,
            )
