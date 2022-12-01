import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, src_list: list, src_att_list: list, src_seg_list: list = None,
                 trg_list: list = None, min_len: int = 4, src_max_len: int = 300):
        self.tensor_list = []
        for src, src_att, src_seg, trg in zip(src_list, src_att_list, src_seg_list, trg_list):
            if min_len <= len(src) <= src_max_len:
                # Source tensor
                src_tensor = torch.zeros(src_max_len, dtype=torch.long)
                src_tensor[:len(src)] = torch.tensor(src, dtype=torch.long)
                src_att_tensor = torch.zeros(src_max_len, dtype=torch.long)
                src_att_tensor[:len(src_att)] = torch.tensor(src_att, dtype=torch.long)
                src_seg_tensor = torch.zeros(src_max_len, dtype=torch.long)
                src_seg_tensor[:len(src_seg)] = torch.tensor(src_seg, dtype=torch.long)
                # Target tensor
                trg_tensor = torch.tensor(trg, dtype=torch.float)
                #
                self.tensor_list.append((src_tensor, src_att_tensor, src_seg_tensor, trg_tensor))
            else:
                print(len(src))

        self.num_data = len(self.tensor_list)

    def __getitem__(self, index):
        return self.tensor_list[index]

    def __len__(self):
        return self.num_data