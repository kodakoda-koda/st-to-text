from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, st_maps, labels, tokenizer, max_length, train_flag=True):
        self.st_maps = st_maps
        self.labels = labels
        self.tokenizer = tokenizer
        self.__split_tokenize__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "st_maps": self.st_maps[idx],
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
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
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = tokenized_labels.input_ids
        self.attention_mask = tokenized_labels.attention_mask
