from torch import nn

class Model(nn.Module):
    def __init__(self, device, dtype) -> None:
        """Initialisation of the Model class"""
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels= 1, out_channels=32, kernel_size=3, padding=2, device=device, dtype=dtype)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, device=device, dtype=dtype)
        self.max1 = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=2, device=device, dtype=dtype)
        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, device=device, dtype=dtype)
        self.max2 = nn.MaxPool2d(kernel_size=(2,2))
        self.flat = nn.Flatten()
        self.lay1 = nn.Linear(3136, 1024, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.lay2 = nn.Linear(1024, 3, device=device, dtype=dtype)
    
    def forward(self, x):
        """Forward pass of the model"""
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.max1(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.max2(x)
        x = self.flat(x)
        x = self.lay1(x)
        x = self.relu(x)
        y = self.lay2(x)
        return y[:,0]
