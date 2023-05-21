import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd

Class 

class concrete_data (Dataset):
    def __init__(self, file_name):
        file_data = pd.read_csv(file_name)

        n_row = file_data.shape[0]
        n_column = file_data.shape[1]

        x = file_data.iloc[0:n_row, 0:n_column-1].values
        y = file_data.iloc[0:n_row, n_column-1].values

        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Get dataset
orig_dataset = concrete_data('concrete_data.csv')

# Split image data randomly into train and validation set
size_train = int(0.9 * len(orig_dataset))
size_validation = len(orig_dataset) - size_train
train_dataset, validation_dataset = random_split(orig_dataset, [size_train, size_validation])

# Load datasets into DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=1)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=1)

# Loss function
criterion = nn.MSELoss()

optimizer = torch.optim.Adam()