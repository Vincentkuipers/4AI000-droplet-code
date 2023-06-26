# File Documentation
In this document you will find the documentation of the project, the documentation is divided into two parts, the first part is the documentation of the code and the second part is the documentation of the project. For more information about the project, please refer to the [README.md](README.md) file.

## Table of Contents
1. [File Structure](#file-structure)
2. [Code Documentation](#code-documentation)
    1. [Blender Code](#blender-files)
    2. [Model Code](#model-folder)
    3. [Cluster Code](#clusterresults)

## File Structure
The file structure of the project is as follows:
```
.
├── AdditionalDATA
├── ClusterResults
│   ├── savefolderpytorch
│   │   ├── Best Models
│   │   └── Cluster Results (csv files)
│   ├── DataLoader.py
│   ├── Model.py
│   ├── Trainer.py
|   ├── runmodel.py
│   ├── Validation.py
│   └── PlotResults.ipynb
├── Model
│   └── Same strucutre as cluster results 
├── Solid_droplet
|   ├── Additional Files
|   |   ├── To create a movie of images
│   ├── codes_gendrops_py
│   │   └── Files to create the droplet
│   ├── Data
│   │   └── Images
|   ├── CurrentBlenderCode.py
|   ├── NOBackgroundBLenderCode.py
│   ├── Empty.blend (empty blender file with material properties)
│   ├── fun_genSingleDrop.py
│   ├── Pyshell.ipynb
│   └── shellcommands.py
├── README.md
├── documentation.md
└── .gitignore
```


## Code Documentation
The code documentation is divided into two parts, the first part is the documentation of the code in the [blender files](Solid_droplet/) and the second part is the documentation of the code in the [Model Folder](Model/) or [ClusterResults](ClusterResults/).

### Blender Files 
To run blender from the terminal one has to ensure that Blender can be called via command terminal. The easiest way to do this is.

1. Have blender installed in a folder with no spaces in the name (e.g. C:\blender). We need this otherwise blender is unable to be called from the terminal.
2. Have Blender as an environment variable in the system. To do this, go to the folder where blender is installed and copy the path. Then go to the system environment variables and add a new variable called BLENDER and paste the path. Then edit the path variable and add the path to the blender folder. 

With this, blender can be called from the terminal. To run blender from the terminal, one has to run the following command:

```PS
blender -b -P <python file> -- <arguments>
```
or can be run using the python script provided in the folder [Solid_droplet](Solid_droplet/shellcommands.py). To run the python script, one has to run the python file. 

In the folder [Solid_droplet](Solid_droplet/) there are two blender files, the first one is [CurrentBlenderCode.py](Solid_droplet/CurrentBlenderCode.py) and the second one is [NOBackgroundBLenderCode.py](Solid_droplet/NOBackgroundBLenderCode.py). The first one is the blender file that creates the droplet with the background and the second one is the blender file that creates the droplet without the background.

Both can be called using the python script provided.

Other files that are in this folder are:
- Additional Files 
- codes_gendrops_py
- Data
- Empty.blend
- fun_genSingleDrop.py
- Pyshell.ipynb
- shellcommands.py

These files/folders will now be explained.

#### Additional Files
This folder contains additional files that are used to create the droplet. These files are:
- [To create a movie of images](Solid_droplet/AdditionalFiles/movie.ipynb)
- 2 movie files
- 2 additional test files for blender

#### codes_gendrops_py
This folder contains the files that are used to create the droplet. These files are:
- [dif1D.py](Solid_droplet/codes_gendrops_py/dif1D.py) (computes the derivative for the droplet structure)
- [fit_circle_through_3_points.py](Solid_droplet/codes_gendrops_py/fit_circle_through_3_points.py) (fits a circle through 3 points)

#### Data
This folder contains the images that are created by blender. 

#### Empty.blend
This is an empty blender file with the material properties.

#### fun_genSingleDrop.py
This file contains the code to create the droplet. This file is called from the [CurrentBlenderCode.py](Solid_droplet/CurrentBlenderCode.py) and [NOBackgroundBLenderCode.py](Solid_droplet/NOBackgroundBLenderCode.py) files.

#### Pyshell.ipynb
This file is a python shell that can be used to run the blender files. This file is not used in the project. But can aid in checking if the blender files are working and if installation is correct.

#### shellcommands.py
This file is a python script that can be used to create the droplets. Multiple properties can be changed using the arguments (lists) in the script. With this code blender files are created in terminal of the computer.

### Model Folder
The model folder contains the code that is used to train the model. The code is divided into 5 files. These files are:
- [DataLoader.py](Model/DataLoader.py) (load data)
- [Model.py](Model/Model.py) (Create the model)
- [Trainer.py](Model/Trainer.py) (Train the model and its properties)
- [Validation.py](Model/Validation.py) (Inference for a given model)
- [runmodel.py](Model/runmodel.py) (Run the model)

other files that are included are:
- [interpretable AI](Model/interpretableAI/) (Has the results for the interpretable AI/visualising model layer)
- [Results](Model/savefolderpytorch/) (contains all the results of the model)
- [TF](Model/TF/) (contains the tensorflow model if one were to use that)
- [iterpretable AI](Model/Interpretable%20AI.ipynb) (contains the code of the interpretable AI)
- [PlotResults.ipynb](Model/PlotResults.ipynb) (contains the code to plot the results)

### ClusterResults 
Contains the exact same files as the model folder. The only difference is that the files in this folder were used to run the model on the cluster. The files are:
- [DataLoader.py](ClusterResults/DataLoader.py)
- [Model.py](ClusterResults/Model.py)
- [Trainer.py](ClusterResults/Trainer.py)
- [Validation.py](ClusterResults/Validation.py)
- [runmodel.py](ClusterResults/runmodel.py)

