from torch.nn import Module, MSELoss
from torch.cuda import is_available, get_device_name
from torch.optim import Adam
from torch import device, save, tensor, no_grad
from torch.utils.data import DataLoader
from sys import stdout
from tqdm import tqdm
from os.path import join
from pandas import DataFrame

class Trainer:
    def __init__(self, model:Module, dltrain:DataLoader, dlval:DataLoader, dltest:DataLoader) -> None:
        """Initialize the Trainer class
            params:
                model: The model to train
                dltrain: The dataloader for the training set
                dlval: The dataloader for the validation set
                dltest: The dataloader for the test set
        """
        self.device = device("cuda" if is_available() else "cpu")
        print(f"The device that will be used in training is {get_device_name(self.device)}")

        self.model = model.to(self.device)

        self.train = dltrain
        self.val = dlval
        self.test = dltest

        self.optimizer = Adam(self.model.parameters())
        self.criteria = MSELoss()

        assert self.criteria is not None, "Please define a loss function"
        assert self.optimizer is not None, "Please define an optimizer"

    def train_epochs(self, epochs:int) -> None:
        """Train the model for a number of epochs"""
        self.model.train()
        loss_list = []
        accuracy = []

        stdout.flush()
        with tqdm(total=epochs, desc="Epochs") as pbar:
            for inputs in self.train:
                # Zero grad for previous step
                self.optimizer.zero_grad()

                # Send everything to device
                inputs = inputs.to(self.device)
                inputs.required_grad = True

                # run model on the inputs
                outputs = self.model(inputs)

                # Perform backpropagation
                loss = self.criteria(outputs, self.train_labels)
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.item())
                accuracy.append(MSELoss(outputs, self.train_labels))

                pbar.update(1)
            
        stdout.flush()
        return loss, accuracy
    

    def val_epoch(self, epochs:int) -> None:
        """Validate the model on the validation set"""
        self.model.eval()
        val_loss= []
        val_accuracy= []

        stdout.flush()
        with no_grad(), tqdm(total=epochs, desc="Epochs") as pbar:
            for inputs in self.val_data:
                # Send everything to device
                inputs = self.val.to(self.device)
                inputs.required_grad = True
                self.val_labels.to(self.device)

                # run model on the inputs
                outputs = self.model(inputs)

                # Perform backpropagation
                loss = self.criteria(outputs, self.val_labels)

                val_loss.append(loss.item())
                val_accuracy.append(MSELoss(outputs, self.val_labels))

                pbar.update(1)
            
        stdout.flush()
        return val_loss, val_accuracy
    
    # def fit(self, epochs:int):
    #     for epoch in range(1, epochs+1):
    #         metrics_train = self.train_epochs(epoch)
    #         metrics_val = self.val_epoch(self.val, self.val_labels)
    #         print(f"Epoch: {epoch} | Train Loss: {metrics_train['loss'][-1]} | Val Loss: {metrics_val['val_loss'][-1]}")
    #     return metrics_train, metrics_val
    
    def save_model(self, model_name:str, DIR:str):
        """Save the model"""
        store_path = join(DIR, model_name)
        
        save(self.model.state_dict(), store_path)

    def fit(self, epochs: int, batch_size:int):
        # Initialize Dataloaders for the `train` and `val` splits of the dataset. 
        # A Dataloader loads a batch of samples from the each dataset split and concatenates these samples into a batch.
        dl_train = self.train
        dl_val = self.val

        # Store metrics of the training process (plot this to gain insight)
        df_train = DataFrame()
        df_val = DataFrame()

        # Train the model for the provided amount of epochs
        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch}')
            metrics_train = self.train_epochs(iter(dl_train)._next_data())
            df_train = df_train.append(DataFrame({'epoch': [epoch for _ in range(len(metrics_train["loss"]))], **metrics_train}), ignore_index=True)

            metrics_val = self.val_epoch(iter(dl_val)._next_data())
            df_val = df_val.append(DataFrame({'epoch': [epoch], **metrics_val}), ignore_index=True)

        # Return a dataframe that logs the training process. This can be exported to a CSV or plotted directly.
        return df_train, df_val