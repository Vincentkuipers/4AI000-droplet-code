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
    BLEND_BLEND_FILE = path.join(CODE_PATH, "empty.blend")
    BLEND_FILE_CODE = path.join(CODE_PATH, "CurrentBlenderCode.py")
    print("Files were found")
except:
    raise NameError("File has not been found check if file is correct")

# note it will do sigma_min, sigma_min+1, ..., sigma_max-1 
[sigma_min, sigma_max] = [40, 71] 
[volume_min, volume_max] = [32, 51]
[rneedle_min, rneedle_max] = [1,2]

system(f"{PATH_BLENDER} -b {BLEND_BLEND_FILE} -P {BLEND_FILE_CODE} -- {sigma_min} {sigma_max} {volume_min} {volume_max} {rneedle_min} {rneedle_max}")
# -b: background (does not open blender) -> speeds up things a ton.
# -P: calls a python file
# -- passes system arguments, needed for sigma, volume, rneedle
