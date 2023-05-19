### Import packages and libraries
import bpy
import sys
import os
import scipy
import matplotlib
from mathutils import Vector
import mathutils
from math import pi
import bmesh
from bmesh.ops import spin

# Add function directories
TEST_DIR = os.getcwd()
FUNC_DIR = os.path.join(TEST_DIR, "codes_gendrops_py\\")
sys.path.insert(1, TEST_DIR) if not TEST_DIR in sys.path else print("Path exists")
sys.path.insert(1, FUNC_DIR) if not FUNC_DIR in sys.path else print("Path exists")

from fun_genSingleDrop import *

sigma_range = range(int(sys.argv[-6]), int(sys.argv[-5]))
volume_range = range(int(sys.argv[-4]), int(sys.argv[-3]))
rneedle_range = range(int(sys.argv[-2]), int(sys.argv[-1]))

for sigma in sigma_range:
    for volume in volume_range:
        for rneedle in rneedle_range:

            ### Clear Scene except for Light and Camera objects
            for ob in bpy.data.objects:
                if ob.name not in ["Camera","Light",""]:
                    bpy.data.objects[ob.name].select_set(True)
                    bpy.ops.object.delete()


            r_a, z_a = genSingleDrop(sigma, volume, rneedle, 1)

            edge_points = [0] * (len(r_a)+1)
            for i in range(len(r_a)):
                x = 0    
                r = r_a[i]    
                z = z_a[i] + 4
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
            #bpy.context.scene.collection.objects.link(new_object)


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
            angle = 2*pi
            steps = 60
            cent = obj.location

            lathe_geometry(bm, cent, axis, dvec, angle, steps, remove_doubles=True, dist=0.0001)

            bm.to_mesh(obj.data)
            # obj.data.update()   # if you want update to show immediately
            bm.free()
            bpy.context.scene.collection.objects.link(obj)

            # Set the background color to blue
            bpy.context.scene.world.node_tree.nodes['Background'].inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
            
            ### Should be removed if not using two step ==>

            for ob in bpy.data.objects:
                if ob.name in ["droplet_object"]:
                    bpy.data.objects[ob.name].select_set(True)
                    if ob.data.materials:
                        idx = ob.active_material_index
                        ob.material_slots[idx].material = bpy.data.materials.get("DropletEmission")
                    else:
                        ob.data.materials.append(bpy.data.materials.get("DropletEmission"))
                if ob.name in ["needle_object"]:
                    try:
                        bpy.data.objects[ob.name].select_set(True)
                        if ob.data.materials:
                            idx = ob.active_material_index
                            ob.material_slots[idx].material = bpy.data.materials.get("BackgroundEmission")
                        else:
                            ob.data.materials.append(bpy.data.materials.get("BackgroundEmission"))
                    except:
                        pass

            # Save file
            bpy.context.scene.frame_end = 0
            bpy.context.scene.render.filepath = f"//Data//{sigma}_{volume}_{rneedle}_edgedetection"
            bpy.ops.render.render(write_still = True)

            bpy.data.objects["droplet_object"].select_set(True)
            bpy.data.objects["droplet_object"].data.materials.clear()

            ### <== Should be removed if we are not using to step


            # Get a reference to the world node tree
            world_node_tree = bpy.context.scene.world.node_tree

            # Create a new Environment Texture node
            environment_texture_node = world_node_tree.nodes.new(type='ShaderNodeTexEnvironment')

            # Load the background image
            background_image = bpy.data.images.load(os.path.join(TEST_DIR, 'TUE_background.jpg'))

            # Set the image as the texture for the Environment Texture node
            environment_texture_node.image = background_image

            # Connect the Environment Texture node to the Background node
            background_node = world_node_tree.nodes['Background']
            world_node_tree.links.new(environment_texture_node.outputs['Color'], background_node.inputs['Color'])



            # Add a subdivision surface modifier
            subsurf = obj.modifiers.new(name='Subdivision Surface', type='SUBSURF')
            subsurf.levels = 2  # Increase the number of subdivision levels

            # Create a new water material
            water_material = bpy.data.materials.new(name="Water")
            water_material.use_nodes = True

            # Get a reference to the Principled BSDF node in the material's node tree
            principled_bsdf_node = water_material.node_tree.nodes["Principled BSDF"]

            # Set the roughness of the material to 0.05
            principled_bsdf_node.inputs["Roughness"].default_value = 0.2

            # Set the transmission to 0.5 (partial transparency)
            principled_bsdf_node.inputs['Transmission'].default_value = 1.0

            # Set the alpha to 0.5 (partial transparency)
            principled_bsdf_node.inputs['Alpha'].default_value = 1.0

            # Set the IOR to 1.33 (refractive index of water)
            principled_bsdf_node.inputs['IOR'].default_value = 1.33

            # Set the color of the material to white
            principled_bsdf_node.inputs["Base Color"].default_value = (0.8, 0.9, 1.0, 1.0)

            # Add a Glossy BSDF node to the material's node tree
            glossy_bsdf_node = water_material.node_tree.nodes.new(type='ShaderNodeBsdfGlossy')
            glossy_bsdf_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
            glossy_bsdf_node.inputs['Roughness'].default_value = 0.1

            # Add a Mix Shader node to the material's node tree
            mix_shader_node = water_material.node_tree.nodes.new(type='ShaderNodeMixShader')
            mix_shader_node.inputs['Fac'].default_value = 0.5

            # Connect the Principled BSDF node to the Mix Shader node
            water_material.node_tree.links.new(principled_bsdf_node.outputs['BSDF'], mix_shader_node.inputs[1])

            # Connect the Glossy BSDF node to the Mix Shader node
            water_material.node_tree.links.new(glossy_bsdf_node.outputs['BSDF'], mix_shader_node.inputs[2])

            # Set the output of the material to the Mix Shader node
            water_material.node_tree.links.new(mix_shader_node.outputs['Shader'], water_material.node_tree.nodes['Material Output'].inputs['Surface'])

            # Set the blend mode to 'Alpha Blend'
            water_material.blend_method = 'BLEND'

            # Assign the water material to the object
            obj.data.materials.append(water_material)

            ########## Add a new point light to the scene
            light_data = bpy.data.lights.new(name="New Light", type='POINT')
            light_obj = bpy.data.objects.new(name="New Light", object_data=light_data)
            bpy.context.scene.collection.objects.link(light_obj)

            # Set the location of the light
            light_obj.location = (-2, -3, 4.0)

            # Set the intensity of the light
            light_data.energy = 100.0

            ##########Get the active camera object
            camera = bpy.context.scene.camera

            # Print the location and rotation of the camera
            # print("Camera location:", camera.location) # Camera location: begin (7.3589, -6.9258, 4.9583)>
            # print("Camera rotation (Euler angles):", camera.rotation_euler) #Camera rotation (Euler angles):begin (x=1.1093, y=0.0000, z=0.8149), order='XYZ'>

            # Set the camera location and rotation
            camera.location = (7.3589*1.5,-6.9258*1.5, 4.9583)         # x, y, z are the desired location values
            camera.rotation_euler = (1.1093*1.25, 0, 0.8149)           # rx, ry, rz are the rotation values in radians



            bpy.context.scene.frame_end = 0
            #bpy.context.scene.render.file_extension = "PNG"
            bpy.context.scene.render.filepath = f"//Data//{sigma}_{volume}_{rneedle}"
            bpy.ops.render.render(write_still = True)

