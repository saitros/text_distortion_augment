import torch
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset

# class CustomDataset(Dataset):
#     def __init__(self, src_list: list, src_att_list: list, src_seg_list: list = list(),
#                  trg_list: list = None, min_len: int = 4, src_max_len: int = 300):
#         # List setting
#         if src_seg_list == list():
#             src_seg_list = [[0] for _ in range(len(src_list))]
#         self.tensor_list = []

#         # Tensor list
#         for src, src_att, src_seg, trg in zip(src_list, src_att_list, src_seg_list, trg_list):
#             if min_len <= len(src) <= src_max_len:
#                 # Source tensor
#                 src_tensor = torch.zeros(src_max_len, dtype=torch.long)
#                 src_tensor[:len(src)] = torch.tensor(src, dtype=torch.long)
#                 src_att_tensor = torch.zeros(src_max_len, dtype=torch.long)
#                 src_att_tensor[:len(src_att)] = torch.tensor(src_att, dtype=torch.long)
#                 src_seg_tensor = torch.zeros(src_max_len, dtype=torch.long)
#                 src_seg_tensor[:len(src_seg)] = torch.tensor(src_seg, dtype=torch.long)
#                 # Target tensor
#                 trg_tensor = torch.tensor(trg, dtype=torch.float)
#                 #
#                 self.tensor_list.append((src_tensor, src_att_tensor, src_seg_tensor, trg_tensor))

#         self.num_data = len(self.tensor_list)

#     def __getitem__(self, index):
#         return self.tensor_list[index]

#     def __len__(self):
#         return self.num_data

class CustomDataset(Dataset):
    def __init__(self, tokenizer, src_list: list = list(), src_list2: list = None, trg_list: list = None, min_len: int = 4, src_max_len: int = 300):

        self.tokenizer = tokenizer
        self.src_tensor_list = list()
        self.src_tensor_list2 = list()
        self.trg_tensor_list = list()

        self.min_len = min_len
        self.src_max_len = src_max_len

        for src in src_list:
            if min_len <= len(src):
                self.src_tensor_list.append(src)

        if src_list2 != None:
            for src in src_list2:
                if min_len <= len(src):
                    self.src_tensor_list2.append(src)

        self.trg_tensor_list = F.one_hot(torch.tensor(trg_list, dtype=torch.long)).numpy()

        self.num_data = len(self.src_tensor_list)

    def __getitem__(self, index):
        if len(self.src_tensor_list2) == 0:
            encoded_dict = \
            self.tokenizer(
                self.src_tensor_list[index],
                max_length=self.src_max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded_dict['input_ids'].squeeze(0)
            attention_mask = encoded_dict['attention_mask'].squeeze(0)
            if len(encoded_dict.keys()) == 3:
                token_type_ids = encoded_dict['token_type_ids'].squeeze(0)
            else:
                token_type_ids = encoded_dict['attention_mask'].squeeze(0)

        else:
            encoded_dict = \
            self.tokenizer(
                self.src_tensor_list[index], self.src_tensor_list2[index],
                max_length=self.src_max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded_dict['input_ids'].squeeze(0)
            attention_mask = encoded_dict['attention_mask'].squeeze(0)
            if len(encoded_dict.keys()) == 3:
                token_type_ids = encoded_dict['token_type_ids'].squeeze(0)
            else:
                token_type_ids = encoded_dict['attention_mask'].squeeze(0)

        trg_tensor = torch.tensor(self.trg_tensor_list[index], dtype=torch.float)
        # print(input_ids)
        # print()
        # print(trg_tensor)
        return (input_ids, attention_mask, token_type_ids, trg_tensor)

    def __len__(self):
        return self.num_data