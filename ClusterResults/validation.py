# Load packages
from DataLoader import *
from Trainer import *
from Model import *
import pandas as pd
import torch
import os

# Ignore warnings
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)

# Set constants
BATCH_SIZE = 32
MAIN_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_DIR = os.path.abspath(os.path.join(MAIN_DIR, "..", "..", "Data"))

# Check if the paths are correct
if os.path.exists(MAIN_DIR) == False or os.path.exists(DATA_DIR) == False:
    raise Exception("Ensure that the paths are correct")

# Set specific options
dtype = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used is: {}".format(DEVICE))

# Split the data into train, validation and test set
dataspliter = DataSpliter(DATA_DIR, DEVICE, train_frac=0.6, val_frac=0.2, test_frac=0.2, use_seed=True, seed_val=42)
train_set = dataspliter.train
val_set = dataspliter.val
test_set = dataspliter.test

# Create the datasets using splits
train_dataset = CustomImageDataset(DATA_DIR, train_set)
val_dataset = CustomImageDataset(DATA_DIR, val_set)
test_dataset = CustomImageDataset(DATA_DIR, test_set)

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Get the model
model = CNNModel3()
trainer = Trainer(model, train_dataloader, val_dataloader, test_dataloader)
_, output, labels = trainer.val_epoch(test_dataloader)

# Validate model using test data
dfresults = pd.DataFrame(np.append(output, labels, axis=1),columns=["sigmapred", "sigmatrue"])
dfresults["RMSE"] = np.sqrt((dfresults["sigmapred"]-dfresults["sigmatrue"])**2)
print(dfresults)
dfresults.to_csv("savefolderpytorch\\results.csv", index=False)