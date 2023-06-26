from cv2 import imread, resize, cvtColor, COLOR_BGR2GRAY
from random import seed, shuffle
from numpy import array, float16, empty
from os import listdir, path



class DataSpliter():
    """Class for loading images and the labels"""
    def __init__(self, DATA_DIR:list, train_frac:float = 0.6, val_frac:float = 0.2, test_frac:float=0.2, use_seed:bool=False, seed_val:int=42):
        # Split the data into train, validation and test set
        train, val, test, trainl, vall, testl = self._split_data(DATA_DIR, train_frac, val_frac, test_frac, use_seed, seed_val)
        
        # Load images
        train, val, test = self._load_images(train, val, test, DATA_DIR)

        # Store data
        self.train = train.reshape(train.shape[0],train.shape[1],train.shape[2],1)
        self.val = val.reshape(val.shape[0],val.shape[1],val.shape[2],1)
        self.test = test.reshape(test.shape[0],test.shape[1],test.shape[2],1)
        self.train_label = trainl
        self.val_label = vall
        self.test_label = testl


    def _load_images(self, train_split, val_split, test_split, DATA_DIR:str):
        """Load images from DATA_DIR"""
        train_images = self._get_images(train_split, DATA_DIR)
        val_images = self._get_images(val_split, DATA_DIR)
        test_images = self._get_images(test_split, DATA_DIR)
        return train_images, val_images, test_images
    

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

        # Get labels
        train_labels = self._get_labels(train_split)
        val_labels = self._get_labels(val_split)
        test_labels = self._get_labels(test_split)

        return train_split, val_split, test_split, train_labels, val_labels, test_labels

        
    def _get_labels(self, list:list):
        """Get labels from list"""
        label = []
        
        for list_item in list:
            [sigma, volume, rneedle] = [eval(i) for i in list_item.split(".")[0].split("_")] # return list of int/float instead of str
            label.append([sigma, volume, rneedle])
        
        # Store data as numpy array
        label = array(label)                    
        return label

    def _image_resize(self, image, size=(320,256)):
        """Resize image to 320x256"""
        return resize(image, size)

    def _image_color(self, image):
        """Convert image from RGB to GRAY"""
        return cvtColor(image, COLOR_BGR2GRAY).transpose(1,0)
    
    def _image_normalize(self, image):
        """Normalize image"""
        return image.astype(float16)/255.0
    
    def _image_processing(self, image, size=(256,256)):
        """Process image"""
        image = self._image_resize(image, size=size)
        image = self._image_color(image)
        image = self._image_normalize(image)
        return image

    def _get_images(self, list_of_labels:list, DATA_DIR:str, im_size:tuple[int,...] = (256,256)):
        """Get images from list"""
        
        images = empty(shape=(len(list_of_labels),im_size[0], im_size[1]))
        
        for i, filename in enumerate(list_of_labels):
            image = imread(path.join(DATA_DIR, filename))
            image = self._image_processing(image, size=im_size)
            images[i,:,:] = image
        
        return images #