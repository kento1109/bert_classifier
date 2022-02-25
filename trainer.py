""" this module is for BERT training """
import copy
import os
from typing import Dict, List

import numpy as np
import torch
from logzero import logger
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW


def build_data_loader(
    example_list: List[Dict], batch_size: int, shuffle: bool = False
) -> DataLoader:
    return DataLoader(
        example_list, batch_size=batch_size, shuffle=shuffle, collate_fn=to_tensor
    )


def to_tensor(_batch: List):
    features = _batch[0].keys()
    batch = dict()
    for f in features:
        batch[f] = torch.tensor([example[f] for example in _batch], dtype=torch.long)
    return batch


class Trainer:
    def __init__(self, model, config):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"running on device {self.device}")

        self.model = model
        self.model.to(self.device)

        self.config = config

        self._create_optimizer_and_scheduler()

    def _create_optimizer_and_scheduler(self):

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.config.lr, betas=(0.9, 0.999)
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, factor=0.1, patience=2, mode="min"
        )

    def train(self, train_loader: DataLoader, valid_loader: DataLoader = None):

        torch.backends.cudnn.benchmark = True

        dataloaders_dict = {"train": train_loader}

        if valid_loader is not None:
            dataloaders_dict["val"] = valid_loader

        best_acc = 0
        # store initial model
        best_model = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.config.num_epochs):
            for phase in dataloaders_dict:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                epoch_loss = 0.0
                epoch_acc = 0
                num_examples = 0

                # for batch in tqdm(dataloaders_dict[phase]):
                for batch in dataloaders_dict[phase]:

                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    with torch.set_grad_enabled(phase == "train"):

                        output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )

                        _, preds = torch.max(output.logits, 1)

                        if phase == "train":
                            self.optimizer.zero_grad()
                            output.loss.backward()
                            self.optimizer.step()

                        epoch_loss += output.loss.item()
                        epoch_acc += torch.sum(preds == labels).item()

                        num_examples += input_ids.size(0)

                epoch_loss = epoch_loss / len(dataloaders_dict[phase])
                epoch_acc = epoch_acc / num_examples

                logger.info(
                    f"Epoch {epoch + 1} | {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
                )

                if phase == "val":
                    self.scheduler.step(epoch_acc)

                    # deep copy the current best model
                    if epoch_acc > best_acc:
                        best_model = copy.deepcopy(self.model.state_dict())
                        best_acc = epoch_acc

        if valid_loader is not None:
            # load best model
            self.model.load_state_dict(best_model)

        torch.cuda.empty_cache()

    def predict(self, pred_df, batch_size):

        pred_loader = self.build_data_loader(
            pred_df, batch_size, shuffle=False, only_inputs=True
        )

        self.model.eval()

        preds = list()
        with torch.no_grad():
            for batch in tqdm(pred_loader):

                inputs = batch[0].to(self.device)
                logit = self.model(input_ids=inputs)[0]
                prob = logit.sigmoid()
                _pred = (prob > 0.5).float().tolist()
                preds.extend(_pred)
        return np.array(preds, dtype="object")

    def evaluate(self, test_loader: DataLoader, label_dict: Dict = None):

        self.model.eval()

        trues_list = list()
        preds_list = list()

        with torch.no_grad():
            for batch in test_loader:

                inputs_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                trues_list.extend(
                    [
                        label_dict[label] if label_dict is not None else label
                        for label in labels.tolist()
                    ]
                )

                output = self.model(input_ids=inputs_ids, attention_mask=attention_mask)

                _, preds = torch.max(output.logits, 1)

                preds_list.extend(
                    [
                        label_dict[label] if label_dict is not None else label
                        for label in preds.tolist()
                    ]
                )

        # convert to 2d
        trues_arr = np.array(trues_list)
        preds_arr = np.array(preds_list)

        print(classification_report(trues_arr, preds_arr, digits=4))
