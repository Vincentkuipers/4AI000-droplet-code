from torch.cuda import is_available, get_device_name
from torch import device, save, no_grad, load
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torch.nn import Module, MSELoss
from torch.optim import Adam
from pandas import DataFrame, read_csv
from os.path import join
from sys import stdout
from os import getcwd
from tqdm import tqdm
import numpy as np



class Trainer:
    def __init__(self, model:Module, dltrain:DataLoader, dlval:DataLoader, dltest:DataLoader, learning_rate:float=0.001) -> None:
        """Initialize the Trainer class
            params:
                model: The model to train
                dltrain: The dataloader for the training set
                dlval: The dataloader for the validation set
                dltest: The dataloader for the test set
        """
        self.device = device("cpu")
        print(f"The device that will be used in training is {self.device}")

        self.model = model.float()

        self.train = dltrain
        self.val = dlval
        self.test = dltest

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = MSELoss(reduction = 'sum')

        assert self.criterion is not None, "Please define a loss function"
        assert self.optimizer is not None, "Please define an optimizer"
    
    def train_epoch(self, dl:DataLoader):
        # Put the model in training mode
        self.model.train().float()

    # Store each step's accuracy and loss for this epoch
        epoch_metrics = {
            "loss": [],
        }

        # Iterate over the training dataset
        for inputs, truths in dl:
            # Zero the gradients from the previous step
            self.optimizer.zero_grad()

            # Move the inputs and truths to the target device
            inputs = inputs.to(device=self.device)
            inputs.required_grad = True  # Fix for older PyTorch versions
            truths = truths.reshape(len(truths),1).to(device=self.device)

            # Run model on the inputs
            output = self.model(inputs)

            # Perform backpropagation
            loss = self.criterion(output, truths)
            loss.backward()
            clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            # Store the metrics of this step
            step_metrics = {
                'loss': loss.item(),
            }
            print(f'Training loss is {step_metrics["loss"]}')

            # Add to epoch's metrics
            for k,v in step_metrics.items():
                epoch_metrics[k].append(v)

        # Return metrics
        return epoch_metrics

    def val_epoch(self, dl:DataLoader):
        # Put the model in evaluation mode
        self.model.eval()

        # Store the total loss and accuracy over the epoch
        amount = 0
        total_loss = 0

        out = []
        tru = []

        # Iterate over the validation dataloader
        for inputs, truths in dl:
            # Move the inputs and truths to the target device
            inputs = inputs.to(device=self.device)
            inputs.required_grad = True  # Fix for older PyTorch versions
            truths = truths.reshape(len(truths),1).to(device=self.device)

            # Run model on the inputs
            output = self.model(inputs)
            loss = self.criterion(output, truths)

            # Store the metrics of this step
            step_metrics = {
                'loss': loss.item(),
            }

            amount += 1
            total_loss += step_metrics["loss"]

            for i in range(len(output)):
                out.append(output[i].cpu().detach().numpy())
                tru.append(truths[i].cpu().detach().numpy())
                
        

        # Print mean of metrics
        total_loss /= amount
        print(f'Validation loss is {total_loss/amount}')
        
        out = self.list_of_arr_to_arr(out)
        tru = self.list_of_arr_to_arr(tru)

        # Return mean loss and accuracy
        return {
            "loss": [total_loss],
        }, out, tru

    def fit(self, epochs: int, batch_size:int, continue_training:bool=False):
        # Initialize Dataloaders for the `train` and `val` splits of the dataset. 
        # A Dataloader loads a batch of samples from the each dataset split and concatenates these samples into a batch.
        dl_train = self.train
        dl_val = self.val

        # Store metrics of the training process (plot this to gain insight)
        if not continue_training: # Not -> create new dataframe
            df_train = DataFrame() 
            df_val = DataFrame()
            # set base line for best loss
            epoch_add = 0
            best_loss:float = 1000000000
        else: # Yes -> load dataframe
            df_train = read_csv('savefolderpytorch\\train.csv')
            df_val = read_csv('savefolderpytorch\\val.csv')
            # Set base line for best loss
            epoch_add = df_train['epoch'].max()
            best_loss:float = df_val["loss"].min()
        

        # Train the model for the provided amount of epochs
        for epoch in range(epoch_add+1, epoch_add+epochs+1):
            print(f'Epoch {epoch}', end='\r')
            metrics_train = self.train_epoch(dl_train)
            df_train = df_train.append(DataFrame({'epoch': [epoch for _ in range(len(metrics_train["loss"]))], **metrics_train}), ignore_index=True)

            metrics_val, _, _ = self.val_epoch(dl_val)
            df_val = df_val.append(DataFrame({'epoch': [epoch], **metrics_val}), ignore_index=True)
            print(f'Epoch {epoch} completed')
            
            if metrics_val["loss"][0] < best_loss:
                best_loss = metrics_val["loss"][0]
                self.save_model("best_model_cluster.pt", "savefolderpytorch")
                print(f"Best model saved at epoch {epoch}")

            # Return a dataframe that logs the training process. This can be exported to a CSV or plotted directly.
            df_train.to_csv('savefolderpytorch\\train_cluster.csv')
            df_val.to_csv('savefolderpytorch\\val_cluster.csv')
        
    
    def save_model(self, model_name:str, DIR:str=getcwd()):
        """Save the model"""
        store_path = join(DIR, model_name)
        
        save(self.model.state_dict(), store_path)

    def load_model(self, model_name:str, DIR:str=getcwd()):
        """Load the model"""
        store_path = join(DIR, model_name)
        
        self.model.load_state_dict(load(store_path, map_location=self.device))
        
    def list_of_arr_to_arr(self, output:list):
        for i in range(1, len(output)):
            if i == 1:
                output_arr = np.append(output[0],output[i])
            else:
                output_arr = np.append(output_arr, output[i])
        return output_arr.reshape(int(output_arr.shape[0]/1),1)
