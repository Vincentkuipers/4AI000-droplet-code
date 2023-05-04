from os import system, environ, path, getcwd

# Find PATH for blender
PATH_BLENDER_FOLDER = environ["Blender"]
PATH_BLENDER = path.join(PATH_BLENDER_FOLDER, "blender") 

# Get code path location
CODE_PATH = getcwd()

# Find droplet .blend file (test.blend)
try:
    BLEND_FILE_PATH = path.join(CODE_PATH, "test.blend1")
except:
    print("File has not been found check if file is correct")

# Storage file location
DATA_PATH = path.join(CODE_PATH, "Data\\")

# Code injection into powershell
system(f"{PATH_BLENDER} -b {BLEND_FILE_PATH} -o {DATA_PATH} -f 10")
# -o: accepts storage location
# -b: background (does not open blender)
# -f: Only a single frame (we don't need multiple, since we don't have an animation).