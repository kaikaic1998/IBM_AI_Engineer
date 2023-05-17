import torch
from torch.utils.data import Dataset

import pandas as pd

# class concrete_data (Dataset):
#     

#     def __len__(self):
#         return len(self.)
    
#     def __getitem__(self, idx):
#         return self.

file_data = pd.read_csv('concrete_data.csv')
print(file_data.shape[0])
print(file_data.shape[1])
