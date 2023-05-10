from os import system, environ, path, getcwd

# Find PATH for blender
try:
    PATH_BLENDER_FOLDER = environ["Blender"]
    PATH_BLENDER = path.join(PATH_BLENDER_FOLDER, "blender") 
    print("Blender Directory was found")
except:
    raise NameError("Directory not found -> try checking if the directory is correctly specified")

# Find code file
try:
    CODE_PATH = getcwd()
    BLEND_FILE_CODE = path.join(CODE_PATH, "CurrentBlenderCode.py")
    print("Files were found")
except:
    raise NameError("File has not been found check if file is correct")

# If we have Python code implentation outside we can use this:
sigma = 101
system(f"{PATH_BLENDER} -b -P {BLEND_FILE_CODE} -- {sigma}")
# -b: background (does not open blender) -> speeds up things a ton.
# -P: calls a python file
# -- passes system arguments, needed for sigma, volume, rneedle
