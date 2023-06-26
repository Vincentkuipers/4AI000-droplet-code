from DataLoader import *
from Trainer import *
from Model import *
from RESNET import *
import pandas as pd
import torch
import os

MAIN_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_DIR = os.path.join(MAIN_DIR, "Solid_droplet", "Data")
BATCH_SIZE = 32

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Starting Validation")

# Split the data into train, validation and test set
dataspliter = DataSpliter(DATA_DIR, device, train_frac=0.6, val_frac=0.2, test_frac=0.2, use_seed=True, seed_val=42)

# Load Data from DataLoader
train_set = dataspliter.train
val_set = dataspliter.val
test_set = dataspliter.test

train_dataset = CustomImageDataset(DATA_DIR, train_set)
val_dataset = CustomImageDataset(DATA_DIR, val_set)
test_dataset = CustomImageDataset(DATA_DIR, test_set)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CNNModel3()
trainer = Trainer(model, train_dataloader, val_dataloader, test_dataloader)
trainer.load_model("best_model.pt", "savefolderpytorch")  

_, output, labels = trainer.val_epoch(test_dataloader)

dfresults = pd.DataFrame(np.append(output, labels, axis=1),columns=["sigmapred", "sigmatrue"])
dfresults["RMSE"] = np.sqrt((dfresults["sigmapred"]-dfresults["sigmatrue"])**2)
print(dfresults)
dfresults.to_csv("savefolderpytorch\\results.csv", index=False)