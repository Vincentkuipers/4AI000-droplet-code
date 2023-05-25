from torch import nn

class RESNET50(nn.Module):
    """Initilize the RESNET50 model"""
    def __init__(self, out_features:int=2, bias:bool=True):
        """RESNET50 setup + forward pass"""
        super(RESNET50, self).__init__()
        self.conv2d_00 = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3),bias=bias),
                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
        )

        self.conv2d_11 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        #nn.Conv2d(64, 256, kernel_size=(1,1), stride=1, bias=False), # If we want to not downsample we should comment out 
                        #nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # these layers
        ) 

        self.conv2d_11 = nn.Sequential(
                        nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
        ) 

        self.conv2d_12 = nn.Sequential(
                        nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
        ) # Last of layer 1

        self.conv2d_20 = nn.Sequential( #init 2nd layer
                        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        ) 

        self.conv2d_21 = nn.Sequential( 
                        nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        ) 

        self.conv2d_22 = nn.Sequential( 
                        nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        ) 

        self.conv2d_23 = nn.Sequential( 
                        nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        )   # Last layer of 2nd layer      

        self.conv2d_30 = nn.Sequential( # Init of third layer
                        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        ) 

        self.conv2d_31 = nn.Sequential( 
                        nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        ) 

        self.conv2d_32 = nn.Sequential( 
                        nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        )

        self.conv2d_33 = nn.Sequential( 
                        nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        )

        self.conv2d_34 = nn.Sequential( 
                        nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        )

        self.conv2d_35 = nn.Sequential( 
                        nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        )


        self.conv2d_40 = nn.Sequential( # Init of fourth layer
                        nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        ) # performed 3 times

        self.conv2d_41 = nn.Sequential( # Init of fourth layer
                        nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        )

        self.conv2d_42 = nn.Sequential( # Init of fourth layer
                        nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=(1,1), bias=bias),
                        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True)
        )

        # To finish resnet50 do this
        self.averagepool = nn.AdaptiveAvgPool2d(kernel_size=(1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(out_features=out_features, bias=True)

    def forward(self, x):
        """Forward Pass of the ResNet50 model"""
        x = self.conv2d_00(x)
        x = self.conv2d_11(x)
        x = self.conv2d_12(x)
        x = self.conv2d_20(x)
        x = self.conv2d_21(x)
        x = self.conv2d_22(x)
        x = self.conv2d_23(x)
        x = self.conv2d_30(x)
        x = self.conv2d_31(x)
        x = self.conv2d_32(x)
        x = self.conv2d_33(x)
        x = self.conv2d_34(x)
        x = self.conv2d_35(x)
        x = self.conv2d_40(x)
        x = self.conv2d_41(x)
        x = self.conv2d_42(x)
        x = self.averagepool(x)
        x = self.fc(x)
        return x