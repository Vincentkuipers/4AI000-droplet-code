from torch import nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
         
        # Flatten the layers
        self.flatten = nn.Flatten()

        # Define the fully connected layers
        self.fc1 = nn.Linear(16384, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 3)
        
    def forward(self, x):
        # Apply the convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layers
        x = self.flatten(x)
        
        # Apply the fully connected layers
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        
        return x
