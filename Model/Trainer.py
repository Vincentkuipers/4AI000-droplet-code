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
    """Class for training the model"""

    def __init__(self, model:Module, dltrain:DataLoader, dlval:DataLoader, dltest:DataLoader, learning_rate:float=0.001) -> None:
        """Initialize the Trainer class

        Args:
            model (Module): model to train	
            dltrain (DataLoader): dataloader for training data
            dlval (DataLoader): dataloader for validation data
            dltest (DataLoader): dataloader for test data
            learning_rate (float, optional): learning rate. Defaults to 0.001.
        
        Returns:
            None -> initializes the class
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
        """Train the model for one epoch

        Args:
            dl (DataLoader): dataloader for training data

        Returns:
            epoch_metrics (dict): dictionary with metrics for this epoch
        """

        # Put the model in training mode
        self.model.train().float()

        # Store each step's accuracy and loss for this epoch
        epoch_metrics = {
            "loss": [],
        }

        stdout.flush()
        with tqdm(total=dl.__len__(), leave=True) as pbar:
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
                
                # Update the progress bar
                pbar.update(1)
                pbar.set_description(f'Training loss is {step_metrics["loss"]}')
        stdout.flush()

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

        # Create a progress bar using TQDM
        stdout.flush()
        with no_grad(), tqdm(total=dl.__len__(), leave=True) as pbar:
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
                
                # Update the progress bar
                pbar.update(1)
                pbar.set_description(f'Validation loss is {total_loss/amount}')

        stdout.flush()

        # Print mean of metrics
        total_loss /= amount
        print(f'Validation loss is {total_loss/amount}')
        
        # Convert list of arrays to array (for plotting)
        out = self.list_of_arr_to_arr(out)
        tru = self.list_of_arr_to_arr(tru)

        # Return mean loss and accuracy
        return {
            "loss": [total_loss],
        }, out, tru

    def fit(self, epochs: int, continue_training:bool=False, EarlyStopping:bool=False, patience:int=10):
        """Fit the model to the training data

        Args:
            epochs (int): Number of epochs to train the model for
            continue_training (bool, optional): Continue training from previous model. Defaults to False.
            EarlyStopping (bool, optional): Use early stopping. Defaults to False.
            patience (int, optional): Patience for early stopping. Defaults to 10.

        Returns:
            df_train (DataFrame): Dataframe with training metrics
            df_val (DataFrame): Dataframe with validation metrics
        """

        # Initialize Dataloaders for the `train` and `val` splits of the dataset. 
        # A Dataloader loads a batch of samples from the each dataset split and concatenates these samples into a batch.
        dl_train = self.train
        dl_val = self.val

        # Initialize variables for early stopping
        if EarlyStopping:
            patience_count = 0

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

            # Train and validate the model for one epoch
            metrics_train = self.train_epoch(dl_train)
            df_train = df_train.append(DataFrame({'epoch': [epoch for _ in range(len(metrics_train["loss"]))], **metrics_train}), ignore_index=True)

            metrics_val, _, _ = self.val_epoch(dl_val)
            df_val = df_val.append(DataFrame({'epoch': [epoch], **metrics_val}), ignore_index=True)

            print(f'Epoch {epoch} completed')
            
            # Save the model if it is the best model so far
            if metrics_val["loss"][0] < best_loss:
                best_loss = metrics_val["loss"][0]
                self.save_model("best_model_cluster.pt", "savefolderpytorch")
                print(f"Best model saved at epoch {epoch}")
                
                if EarlyStopping:
                    patience_count = 0

    	    # Update counter for early stopping
            elif EarlyStopping:
                patience_count += 1
                if patience_count >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
            # Return a dataframe that logs the training process. This can be exported to a CSV or plotted directly.
            df_train.to_csv('savefolderpytorch\\train_cluster.csv')
            df_val.to_csv('savefolderpytorch\\val_cluster.csv')
        
    
    def save_model(self, model_name:str, DIR:str=getcwd()):
        """Save the model
        
        Args:
            model_name (str): name of the model
            DIR (str, optional): directory to store the model. Defaults to getcwd().
        
        Returns:
            None -> saves the model
        """
        store_path = join(DIR, model_name)
        
        save(self.model.state_dict(), store_path)

    def load_model(self, model_name:str, DIR:str=getcwd()):
        """Load the model
        
        Args:
            model_name (str): name of the model
            DIR (str, optional): directory to load the model from. Defaults to getcwd().
        
        Returns:
            None -> loads the model
        """

        store_path = join(DIR, model_name)
        
        self.model.load_state_dict(load(store_path, map_location=self.device))
        
    def list_of_arr_to_arr(self, output:list):
        """Convert list of arrays to array, used for validation

        Args:
            output (list): list of arrays

        Returns:
            output_arr (array): array of arrays
        """
        for i in range(1, len(output)):
            if i == 1:
                output_arr = np.append(output[0],output[i])
            else:
                output_arr = np.append(output_arr, output[i])
        return output_arr.reshape(int(output_arr.shape[0]/1),1)
