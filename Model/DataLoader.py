from random import seed, shuffle
from os import listdir
from numpy import array

class DataLoader():
    """Class for loading data"""
    def __init__(self, DATA_DIR:list, train_frac:float = 0.6, val_frac:float = 0.2, test_frac:float=0.2, use_seed:bool=False, seed_val:int=42):
        train, val, test, trainl, vall, testl = self._split_data(DATA_DIR, train_frac, val_frac, test_frac, use_seed, seed_val)
        self.train = train
        self.val = val
        self.test = test
        self.train_label = trainl
        self.val_label = vall
        self.test_label = testl


    def _split_data(self, DATA_DIR:list, train_frac:float = 0.6, val_frac:float = 0.2, test_frac:float=0.2, use_seed:bool=False, seed_val:int=42):
        """Split data into train, validation and test set"""

        # Check if sum of train_frac, val_frac and test_frac is 1
        if train_frac + val_frac + test_frac != 1:
            raise ValueError("The sum of train_frac, val_frac and test_frac must be 1")
        
        # Check if train_frac, val_frac and test_frac are less than 1
        if train_frac>1 or val_frac>1 or test_frac>1:
            raise ValueError("train_frac, val_frac and test_frac must be less than 1")
        

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

        train_labels = self._get_labels(train_split)
        val_labels = self._get_labels(val_split)
        test_labels = self._get_labels(test_split)

        return train_split, val_split, test_split , train_labels, val_labels, test_labels

        
    def _get_labels(self, list:list):
        """Get labels from list"""
        label = []
        
        for list_item in list:
            [sigma, volume, rneedle] = list_item.split(".")[0].split("_")

            # Store data as float instead of string
            sigma = float(sigma)
            volume = float(volume)
            rneedle = float(rneedle)

            # store data
            label.append([sigma, volume, rneedle])
        
        # Store data as numpy array
        label = array(label)                    
        return label

