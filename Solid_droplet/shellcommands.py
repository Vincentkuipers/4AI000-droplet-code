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
    #BLEND_FILE_CODE = path.join(CODE_PATH, "CurrentBlenderCode.py")
    BLEND_FILE_CODE = path.join(CODE_PATH, "NOBackgroundBlenderCode.py")
    print("Files were found")
except:
    raise NameError("File has not been found check if file is correct")

# note it will do sigma_min, sigma_min+1, ..., sigma_max-1 
[sigma_min, sigma_max, sigma_step] = [45, 75, 0.5] 
[volume_min, volume_max, volume_step] = [5, 6, 1]
[rneedle_min, rneedle_max, rneedle_step] = [1,2,1]
[range_min, range_max, range_step] = [10,11,1]
[num_steps] = [60] # number of steps to perform 360 rotation
[vert_steps] = [1] # number of steps with a set angle increase vertically, angle is defined in currentblendercode

image_count = int(((sigma_max - sigma_min) /sigma_step)*((volume_max - volume_min) /volume_step)*((rneedle_max - rneedle_min) /rneedle_step)*((range_max - range_min) /range_step)*num_steps*vert_steps )
print('Number of images:', image_count)

system(f"{PATH_BLENDER} -b {BLEND_BLEND_FILE} -P {BLEND_FILE_CODE} -- {vert_steps} {num_steps} {range_min} {range_max} {range_step} {sigma_min} {sigma_max} {sigma_step} {volume_min} {volume_max} {volume_step} {rneedle_min} {rneedle_max} {rneedle_step}  ")
# -b: background (does not open blender) -> speeds up things a ton.
# -P: calls a python file
# -- passes system arguments, needed for sigma, volume, rneedle
print(image_count, 'Images created')