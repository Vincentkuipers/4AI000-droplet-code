### Import packages and libraries
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

# Add function directories
TEST_DIR = os.getcwd()
FUNC_DIR = os.path.join(TEST_DIR, "codes_gendrops_py\\")
sys.path.insert(1, TEST_DIR) if not TEST_DIR in sys.path else print("Path exists")
sys.path.insert(1, FUNC_DIR) if not FUNC_DIR in sys.path else print("Path exists")

from fun_genSingleDrop import *

### Clear Scene except for Light and Camera objects
for ob in bpy.data.objects:
    if ob.name not in ["Camera","Light"]:
        bpy.data.objects[ob.name].select_set(True)
        bpy.ops.object.delete()


sigma = int(sys.argv[-1])
#volume = sys.argv
#rneedle = sys.argv

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
cent = obj.location

lathe_geometry(bm, cent, axis, dvec, angle, steps, remove_doubles=True, dist=0.0001)

bm.to_mesh(obj.data)
# obj.data.update()   # if you want update to show immediately
bm.free()

# Add a subdivision surface modifier
subsurf = obj.modifiers.new(name='Subdivision Surface', type='SUBSURF')
subsurf.levels = 2  # Increase the number of subdivision levels

# Create a new water material
water_material = bpy.data.materials.new(name="Water")
water_material.use_nodes = True

# Get a reference to the Principled BSDF node in the material's node tree
principled_bsdf_node = water_material.node_tree.nodes["Principled BSDF"]

# Set the color of the material to blue
principled_bsdf_node.inputs["Base Color"].default_value = (0.0, 0.0, 1.0, 1.0)

# Set the roughness of the material to 0.1
principled_bsdf_node.inputs["Roughness"].default_value = 0.1

# Assign the water material to the object
obj.data.materials.append(water_material)

# Create a new world
world = bpy.data.worlds.new("world")

# Set the background color of the world to white
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")
bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)

# Assign the world to the scene
bpy.context.scene.world = world

bpy.context.scene.frame_end = 0
#bpy.context.scene.render.file_extension = "PNG"
bpy.context.scene.render.filepath = f"//Data//{sigma}"
bpy.ops.render.render(write_still = True)

