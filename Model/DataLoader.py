from torch import float32, tensor, device
from torch.utils.data import Dataset
from torch.cuda import is_available
from random import seed, shuffle
from numpy import asarray, array
from os import listdir, path
from PIL import Image



class CustomImageDataset(Dataset):
    """Custom dataset for loading images and labels"""
    def __init__(self, DATA_DIR, name_set:str):
        self.root_dir = DATA_DIR
        self.file_list = name_set
        self.device = device("cuda" if is_available() else "cpu")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        image_path = path.join(self.root_dir, filename)

        # Get label:
        label = tensor(array(filename.split(".png")[0].split("_"), dtype=float), device=self.device, dtype=float32) # return list of int/float instead of str

        # Get image
        image = Image.open(image_path).convert('RGB').resize((256,256))
        image = tensor(asarray(image), dtype=float32, device=self.device).permute(2,0,1)

        return image, label
    


class DataSpliter():
    """Class for loading images and the labels"""
    def __init__(self, DATA_DIR:list, device:str, train_frac:float = 0.6, val_frac:float = 0.2, test_frac:float=0.2, use_seed:bool=False, seed_val:int=42):
        self.device = device
        # Split the data into train, validation and test set
        train, val, test = self._split_data(DATA_DIR, train_frac, val_frac, test_frac, use_seed, seed_val)
        
        # Store data
        self.train = train
        self.val = val
        self.test = test
    

    def _split_data(self, DATA_DIR:list, train_frac:float = 0.6, val_frac:float = 0.2, test_frac:float=0.2, use_seed:bool=False, seed_val:int=42):
        """Split data into train, validation and test set"""

        # Check if sum of train_frac, val_frac and test_frac is 1
        if train_frac + val_frac + test_frac != 1:
            raise ValueError("The sum of train_frac, val_frac and test_frac must be 1")
        
        # Check if train_frac, val_frac and test_frac are less than 1
        if train_frac>=1 or val_frac>=1 or test_frac>=1:
            raise ValueError("train_frac, val_frac and test_frac must be less than or equal to 1")
        

        # Make copy of data_labels
        list_of_labels = listdir(DATA_DIR).copy()
        
        # Frac to value
        int_val = len(list_of_labels)
        train_value = int(train_frac*int_val)
        val_value = int(val_frac*int_val)
        
        # Shuffle data
        if use_seed:
            seed(seed_val)

        shuffle(list_of_labels)

        # Split data
        train_split = list_of_labels[:train_value]
        val_split = list_of_labels[train_value:train_value+val_value]
        test_split = list_of_labels[train_value+val_value:]

        return train_split, val_split, test_split

