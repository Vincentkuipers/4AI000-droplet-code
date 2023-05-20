from torch.nn import Module, MSELoss
from torch.cuda import is_available, get_device_name
from torch.optim import Adam
from torch import device, save, tensor, no_grad
from sys import stdout
from tqdm import tqdm
from os.path import join

class Trainer:
    def __init__(self, model:Module, im_train:tensor, im_val:tensor, im_test:tensor, train_labels:tensor, val_labels:tensor, test_labels:tensor) -> None:
        """Initialize the Trainer class
            params:
                model: The model to train
                im_train:tensor = The training images
                im_val:tensor = The validation images
                im_test:tensor = The test images
                train_labels:tensor = The training labels
                val_labels:tensor = The validation labels
                test_labels:tensor = The test labels
        """
        self.device = device("cuda" if is_available() else "cpu")
        print(f"The device that will be used in training is {get_device_name(self.device)}")

        self.model = model.to(self.device)

        self.train = im_train.to(self.device)
        self.val = im_val.to(self.device)
        self.test = im_test.to(self.device)
        self.train_labels = train_labels.to(self.device)
        self.val_labels = val_labels.to(self.device)
        self.test_labels = test_labels.to(self.device)


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
                inputs = self.val
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
    
    def fit(self, epochs:int):
        for epoch in range(1, epochs+1):
            metrics_train = self.train_epochs(epoch)
            metrics_val = self.val_epoch(self.val, self.val_labels)
            print(f"Epoch: {epoch} | Train Loss: {metrics_train['loss'][-1]} | Val Loss: {metrics_val['val_loss'][-1]}")
        return metrics_train, metrics_val
    
    def save_model(self, model_name:str, DIR:str):
        """Save the model"""
        store_path = join(DIR, model_name)
        
        save(self.model.state_dict(), store_path)