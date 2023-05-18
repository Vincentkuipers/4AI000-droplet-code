import bpy
import sys
import os
import scipy
import matplotlib
from mathutils import Vector
import mathutils
import math
import bmesh
from bmesh.ops import spin

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir )
from fun_genSingleDrop import *

sigma = int(sys.argv[-1])
#volume = sys.argv
#rneedle = sys.argv

for ob in bpy.data.objects:
    if ob.name in ["Light"]:
        bpy.data.objects[ob.name].select_set(True)
        bpy.ops.object.delete()


r_a, z_a = genSingleDrop(sigma, 32, 1,1 )

edge_points = [0] * (len(r_a)+1)
for i in range(len(r_a)):
    x = 0    
    r = r_a[i]    
    z = z_a[i] +4
    edge_points[i] = Vector((x,r,z))
edge_points[-1] = Vector((0,0,z_a[-1]+4))
        
edges = []

#start_index = len(r_a) 
#end_index = start_index + len(r_a) 
#indices = list(range(start_index, end_index))

# Define the faces for this slice
for i in range(len(edge_points)-1):
    a = i
    b = i+1

    edges.append((a, b))

# Create the mesh
new_mesh = bpy.data.meshes.new("droplet_mesh")
new_mesh.from_pydata(edge_points, edges, [])
new_mesh.update()


new_object = bpy.data.objects.new("droplet_object", new_mesh)
bpy.context.scene.collection.objects.link(new_object)

def lathe_geometry(bm, cent, axis, dvec, angle, steps, remove_doubles=True, dist=0.0001):
    geom = bm.verts[:] + bm.edges[:]

    # super verbose explanation.
    spin(
        bm, 
        geom=geom,         # geometry to use for the spin
        cent=cent,         # center point of the spin world
        axis=axis,         # axis, a (x, y, z) spin axis
        dvec=dvec,         # offset for the center point
        angle=angle,       # how much of the unit circle to rotate around
        steps=steps,       # spin subdivision level 
        use_duplicate=0)   # include existing geometry in returned content

    if remove_doubles:
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=dist)


obj = bpy.data.objects["droplet_object"]
bm = bmesh.new()
bm.from_mesh(obj.data)

axis = (0,0,1)
dvec = (0,0,0)
angle = 2*math.pi
steps = 60
cent =  obj.location



lathe_geometry(bm, cent, axis, dvec, angle, steps, remove_doubles=True, dist=0.0001)

bm.to_mesh(obj.data)
# obj.data.update()   # if you want update to show immediately
bm.free()

# Set the background color to blue
# bpy.context.scene.world.node_tree.nodes['Background'].inputs[0].default_value = (0.0, 0.0, 1.0, 1.0)
# Get a reference to the world node tree
world_node_tree = bpy.context.scene.world.node_tree

# Create a new Environment Texture node
environment_texture_node = world_node_tree.nodes.new(type='ShaderNodeTexEnvironment')

# Load the background image
background_image = bpy.data.images.load('C:/4AI000/Solid_droplet/scythian_tombs_2_4k.exr')

# Set the image as the texture for the Environment Texture node
environment_texture_node.image = background_image

# Connect the Environment Texture node to the Background node
background_node = world_node_tree.nodes['Background']
world_node_tree.links.new(environment_texture_node.outputs['Color'], background_node.inputs['Color'])


# Add a subdivision surface modifier
subsurf = obj.modifiers.new(name='Subdivision Surface', type='SUBSURF')
subsurf.levels = 2 # Increase the number of subdivision levels


# Create a new water material
water_material = bpy.data.materials.new(name="Water")
water_material.use_nodes = True

water_material = bpy.data.materials.new(name="Water")
water_material.use_nodes = True

# Get a reference to the Principled BSDF node in the material's node tree
principled_bsdf_node = water_material.node_tree.nodes["Principled BSDF"]

# Set the roughness of the material to 0.2
principled_bsdf_node.inputs["Roughness"].default_value = 0.2

# Set the transmission to 1.0 (full transparency)
principled_bsdf_node.inputs['Transmission'].default_value = 1.0

# Set the alpha to 1.0 (full transparency)
principled_bsdf_node.inputs['Alpha'].default_value = 1.0

# Set the IOR to 1.33 (refractive index of water)
principled_bsdf_node.inputs['IOR'].default_value = 1.33

# Set the color of the material to white
principled_bsdf_node.inputs["Base Color"].default_value = (0.8, 0.9, 1.0, 1.0)

# Create a new material slot for the object
obj.data.materials.append(water_material)

# ########## Add a new point light to the scene
# light_data = bpy.data.lights.new(name="New Light", type='POINT')
# light_obj = bpy.data.objects.new(name="New Light", object_data=light_data)
# bpy.context.scene.collection.objects.link(light_obj)

# # Set the location of the light
# light_obj.location = (-2, -3, 4.0)

# # Set the intensity of the light
# light_data.energy = 100.0

##########Get the active camera object
camera = bpy.context.scene.camera

# Print the location and rotation of the camera
# print("Camera location:", camera.location) # Camera location: begin (7.3589, -6.9258, 4.9583)>
# print("Camera rotation (Euler angles):", camera.rotation_euler) #Camera rotation (Euler angles):begin (x=1.1093, y=0.0000, z=0.8149), order='XYZ'>

# Set the camera location and rotation
camera.location = (7.3589*1.5,-6.9258*1.5, 4.9583)         #x, y, z are the desired location values
camera.rotation_euler = (1.1093*1.25, 0, 0.8149*1.2)           # rx, ry, rz are the rotation values in radians


bpy.context.scene.frame_end = 0
#bpy.context.scene.render.file_extension = "PNG"
bpy.context.scene.render.filepath = f"//Data//{sigma}"
bpy.ops.render.render(write_still = True)

#         ### Clear Scene except for Light and Camera objects
# for ob in bpy.data.objects:
#     if ob.name not in ["Camera","Light"]:
#         bpy.data.objects[ob.name].select_set(True)
#         bpy.ops.object.delete()

