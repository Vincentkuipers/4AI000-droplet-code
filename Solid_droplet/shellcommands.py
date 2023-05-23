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
[sigma_min, sigma_max, sigma_step] = [66, 80, 0.25] 
[volume_min, volume_max, volume_step] = [32, 33, 1]
[rneedle_min, rneedle_max, rneedle_step] = [1,2,1]

system(f"{PATH_BLENDER} -b {BLEND_BLEND_FILE} -P {BLEND_FILE_CODE} -- {sigma_min} {sigma_max} {sigma_step} {volume_min} {volume_max} {volume_step} {rneedle_min} {rneedle_max} {rneedle_step}")
# -b: background (does not open blender) -> speeds up things a ton.
# -P: calls a python file
# -- passes system arguments, needed for sigma, volume, rneedle
