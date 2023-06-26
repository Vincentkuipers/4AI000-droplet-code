# Abstract
The pendant drop is a problem that is currently solved using the positional information of a droplet, which is extracted using a computational heavy algorithm to determine the volume and surface tension.
In this study, a novel approach is proposed, in which data is created using Blender, an open-source 3D-rendering software. The data can be augmented to suit different environments and create large datasets. Furthermore, a convolutional neural network will be used to extract the surface tension from a given image. The trained neural network has an accuracy of 81.20\% for a tolerance range of 0.5 mN/m and a 98.29\% accuracy for a tolerance range of 1 mN/m. The robustness of the model was verified with data augmentation and it was found that the data performed under added noise. The model performed poorly under other augmentations such as changing the background, adding rotation, using a mask of the droplet, and changing the volume. 

# Application
The pendant drop is a problem that is currently solved using the positional information of a droplet, which is extracted using a computational heavy algorithm to determine the volume and surface tension. The solution provided uses convolutional neural networks to solve the problem to instantly detect the properties of a droplet.

The model uses python and a copy is provided to run the code using pytorch as well as tensorflow.

# Installation and Usage
## Installation
### Blender
1. Have blender installed in a folder with no spaces in the name (e.g. C:\blender). We need this otherwise blender is unable to be called from the terminal.
2. Have Blender as an environment variable in the system. To do this, go to the folder where blender is installed and copy the path. Then go to the system environment variables and add a new variable called BLENDER and paste the path. Then edit the path variable and add the path to the blender folder.

### Python with packages:
1. Python 3.9
2. with common packages such as numpy, pandas, matplotlib, etc.
3. pytorch
4. tensorflow
5. opencv

## Usage
Read the documentation file but the general idea is to run the blender file to generate the images and then run the python file to train the model.

# Credits
This project was done by a group of students from the University of Technology Eindhoven. Credits to our professor for guiding us through the project Dr. N.Jaensson

# License
MIT License
