from os import system, environ, path, getcwd

# Find PATH for blender
try:
    PATH_BLENDER_FOLDER = environ["Blender"]
    PATH_BLENDER = path.join(PATH_BLENDER_FOLDER, "blender") 
    print("Blender Directory was found")
except:
    print("Directory not found -> try checking if the directory is correctly specified")


# Get code path location
CODE_PATH = getcwd()

# Find droplet .blend file (test.blend)
try:
    # Blender location
    BLEND_FILE_PATH = path.join(CODE_PATH, "empty.blend")
    BLEND_FILE_CODE = path.join(CODE_PATH, "CurrentBlenderCode.py")
    print("Files were found")
    # BLEND_FILE_PATH = path.join(CODE_PATH, "BlenderTest.blend")
except:
    print("File has not been found check if file is correct")

# Storage file location
DATA_PATH = path.join(CODE_PATH, "Data\\")

# Code injection into powershell
# system(f"{PATH_BLENDER} -b {BLEND_FILE_PATH} -o {DATA_PATH} -f 10")
# -o: accepts storage location -> predefined in CurrentBlenderCode.py
# -b: background (does not open blender) -> speeds up things a ton.
# -f: Only a single frame (we don't need multiple, since we don't have an animation). -> set up such that only one frame is rendered.

# If we have Python code implentation outside we can use this:
sigma = 90
system(f"{PATH_BLENDER} -b -P {BLEND_FILE_CODE} -- {sigma}")
# -P: calls a python file
# -- passes system arguments, needed for sigma, volume, rneedle
