import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, st_maps, labels, tokenizer, max_length, train_flag=True):
        self.st_maps = torch.tensor(st_maps).view(-1, 30, 100).float()
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_flag = train_flag
        self.__split_tokenize__()

    def __len__(self):
        return len(self.st_maps)

    def __getitem__(self, idx):
        return {
            "st_maps": self.st_maps[idx],
            "inst_input_ids": self.inst_input_ids[idx],
            "decoder_input_ids": self.decoder_input_ids[idx],
            "decoder_attention_mask": self.decoder_attention_mask[idx],
        }

    def __split_tokenize__(self):
        if self.train_flag:
            self.st_maps = self.st_maps[: int(0.8 * len(self.st_maps))]
            self.labels = self.labels[: int(0.8 * len(self.labels))]
        else:
            self.st_maps = self.st_maps[int(0.8 * len(self.st_maps)) :]
            self.labels = self.labels[int(0.8 * len(self.labels)) :]

        labels = ["<pad>" + label for label in self.labels]
        tokenized_labels = self.tokenizer.batch_encode_plus(
            labels,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.decoder_input_ids = tokenized_labels.input_ids
        self.decoder_attention_mask = tokenized_labels.attention_mask

        inst = ["Generate a caption for the given spatial-temporal data" for _ in range(len(self.st_maps))]
        tokenized_inst = self.tokenizer.batch_encode_plus(
            inst,
            max_length=16,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.inst_input_ids = tokenized_inst.input_ids
