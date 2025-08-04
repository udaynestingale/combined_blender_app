# This script replaces a blender object with a glb file and sets custom camera and lighting
#
# Usage: blender --python bpy-replace-obj.py --background -- file.blend -O PRIMARY_OBJECT_NAME -R file.glb

import bpy
import os
import sys
import argparse
import mathutils
import math
import json

# Global rendering configuration options
USE_DENOISING = True  # Set to False to disable denoising
USE_GPU = True  # Set to False to force CPU rendering
GPU_FALLBACK_TO_CPU = True  # Set to False to throw an error if GPU not available

# List of object types to generate masks for
MASK_OBJECTS = ["sofa", "chair", "table", "bed", "cabinet", "lamp", "nightstand", 'pCube36', 'pCylinder7', 'pCylinder8', 'Box004']

# Default camera configuration parameters (will be overridden by command line args if provided)
CAMERA_PARAMS = {
    "position": {
        "x": -0.062294767332897294 * 0, 
        "y": 1.12069881985108255 * 0, 
        "z": 5.0057587797429655 * 0
    },
    "target": {
        "x": 0,
        "y": 1 * 0,
        "z": 0
    },
    "up": {
        "x": 0,
        "y": 1 * 0,
        "z": 0
    },
    "fov": 10,
    "aspect_ratio": 1.7761058672451444 * 0,
    "near": 0.1 * 0,
    "far": 1000 * 0,
    "productNameList": []  # Default empty list for productNameList
}

# Default lighting configuration parameters (will be overridden by command line args if provided)
LIGHTING_PARAMS = {
    "position": {
        "x": 0,
        "y": 2 * 0,
        "z": 0
    },
    "color": "ffeedf",
    "energy": 70 * 0,
    "distance": 6 * 0,
    "decay": 1.5 * 0
}

# Environment configuration parameters
ENVIRONMENT_PARAMS = {
    "hdri_preset": "morning",
    "exposure": 1.2,
    "background_color": (0.8, 0.8, 0.95, 1.0)  # Soft blue morning color
}

# Rendering configuration parameters
RENDER_PARAMS = {
    "samples": 20,
    "preview_samples": 64,
    "denoising_strength": 1.0,  # Higher values for stronger denoising (0.0-1.0)
    "resolution_percentage": 200,
    "film_transparent": True
}

# Mask rendering configuration parameters
MASK_PARAMS = {
    "render_engine": "BLENDER_EEVEE_NEXT",  # Non-cycles renderer for masks
    "samples": 4,                      # Lower samples for EEVEE
    "resolution_percentage": 100,
    "background_color": (0, 0, 0, 1),  # Black background
    "object_color": (1, 1, 1, 1)       # White object
}

def get_objs_bbox(objs):
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for obj in objs for corner in obj.bound_box]
    return mathutils.Vector((min([corner.x for corner in bbox_corners]), max([corner.y for corner in bbox_corners]), min([corner.z for corner in bbox_corners]))),mathutils.Vector((
           max([corner.x for corner in bbox_corners]), min([corner.y for corner in bbox_corners]), max([corner.z for corner in bbox_corners])))

def replace(from_name, to_glb):
    if from_name in bpy.data.objects:
        original_object = bpy.data.objects[from_name]
        obj_min, obj_max = get_objs_bbox([original_object])
        obj_min -= original_object.location

        original_location = original_object.location.copy()
        original_rotation = original_object.rotation_euler.copy()

        bpy.data.objects.remove(original_object, do_unlink=True)
    else:
        print("not found")

    existing_objects = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=to_glb)
    imported_objects = set(bpy.data.objects) - existing_objects

    imported_object = next(iter(imported_objects))

    new_min, new_max = get_objs_bbox([imported_object])
    new_min -= imported_object.location
    imported_object.name = from_name
    imported_object.location = original_location + (obj_min - new_min)
    imported_object.rotation_euler = original_rotation

def get_all_mesh_descendants(parent_obj):
    """Recursively find all mesh objects that are descendants of the given parent object"""
    mesh_objects = []
    
    # Check if this is a mesh object
    if parent_obj.type == 'MESH':
        mesh_objects.append(parent_obj)
    
    # Recursively check all children
    for child in parent_obj.children:
        mesh_objects.extend(get_all_mesh_descendants(child))
    
    return mesh_objects

def create_mask_material(obj):
    """Create a material that outputs pure white with alpha"""
    mat = bpy.data.materials.new(name=f"Mask_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Create principled BSDF with full white and alpha
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)
    bsdf.inputs['Alpha'].default_value = 1.0
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    # Disable all shading effects
    # mat.shadow_method = 'NONE'
    mat.blend_method = 'OPAQUE'
    
    return mat

def post_process_mask(image_path, output_path):
    """Convert RGBA to 1-bit alpha channel"""
    from PIL import Image
    import numpy as np
    
    img = Image.open(image_path)
    alpha = np.array(img)[..., 3]  # Extract alpha channel
    
    # Threshold to pure binary (0 or 255)
    binary_alpha = ((alpha > 127) * 255).astype(np.uint8)
    
    # Save as 1-bit PNG
    Image.fromarray(binary_alpha).save(
        output_path,
        optimize=True,
        bits=1
    )

def setup_mask_render_settings():
    """Set up rendering settings for alpha-only binary masks"""
    # Store original render settings
    original_settings = {
        'engine': bpy.context.scene.render.engine,
        'film_transparent': bpy.context.scene.render.film_transparent,
        'image_settings': {
            'file_format': bpy.context.scene.render.image_settings.file_format,
            'color_mode': bpy.context.scene.render.image_settings.color_mode,
            'color_depth': bpy.context.scene.render.image_settings.color_depth,
            'compression': bpy.context.scene.render.image_settings.compression
        }
    }
    
    # Use EEVEE for faster, crisper masks
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.render.film_transparent = True  # Transparent background
    
    # Disable anti-aliasing and effects
    bpy.context.scene.eevee.taa_render_samples = 1
    bpy.context.scene.eevee.use_gtao = False
    # bpy.context.scene.eevee.use_bloom = False
    # bpy.context.scene.eevee.use_ssr = False
    
    # Configure output for 8-bit PNG with alpha
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.render.image_settings.compression = 0  # No compression
    
    # Set transparent world
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("MaskWorld")
    bpy.context.scene.world.use_nodes = True
    nodes = bpy.context.scene.world.node_tree.nodes
    nodes.clear()
    output = nodes.new(type='ShaderNodeOutputWorld')
    
    return original_settings

def replace_image_with_binary_mask(filepath):
    """Convert rendered alpha to pure binary (0 or 255)"""
    image = bpy.data.images.load(filepath)
    alpha_pixels = list(image.pixels)[3::4]  # Extract alpha channel
    
    # Convert to binary: >0.5 = white, <=0.5 = transparent
    binary_pixels = []
    for alpha in alpha_pixels:
        if alpha > 0.5:
            binary_pixels.extend([1.0, 1.0, 1.0, 1.0])  # White
        else:
            binary_pixels.extend([0.0, 0.0, 0.0, 0.0])  # Transparent
    
    width, height = image.size
    binary_image = bpy.data.images.new("binary_mask", width, height)
    binary_image.pixels = binary_pixels
    binary_image.filepath_raw = filepath
    binary_image.file_format = 'PNG'
    binary_image.save()
    
    bpy.data.images.remove(image)  # Clean up

def restore_render_settings(original_settings):
    """Restore original render settings"""
    bpy.context.scene.render.engine = original_settings['engine']
    bpy.context.scene.render.film_transparent = original_settings['film_transparent']
    
    # Restore image settings
    bpy.context.scene.render.image_settings.file_format = original_settings['image_settings']['file_format']
    bpy.context.scene.render.image_settings.color_mode = original_settings['image_settings']['color_mode']
    bpy.context.scene.render.image_settings.color_depth = original_settings['image_settings']['color_depth']
    bpy.context.scene.render.image_settings.compression = original_settings['image_settings']['compression']

def restore_render_settings(original_engine, original_film_transparent, original_resolution_percentage):
    """Restore original render settings after mask generation"""
    bpy.context.scene.render.engine = original_engine
    bpy.context.scene.render.film_transparent = original_film_transparent
    bpy.context.scene.render.resolution_percentage = original_resolution_percentage

def crop_imagefile(filepath, out_filepath, ratio):
    image = bpy.data.images.load(filepath)
    width, height = image.size
    new_height = int(height * ratio)
    float_buffer = False if "mask" in filepath else image.is_float
    out_image = bpy.data.images.new("new", width, new_height, float_buffer=float_buffer)
    offset = ((height - new_height) >> 1) * width
    size = new_height * width
    out_image.pixels = image.pixels[offset * image.channels:(offset + size) * image.channels]
    out_image.filepath_raw = out_filepath
    out_image.file_format = 'PNG'
    out_image.save()

def setup_custom_camera():
    """Set up a custom camera with the provided parameters using position, rotation, up, and target"""
    # Always create a new camera
    cam_data = bpy.data.cameras.new('CustomCamera')
    cam_obj = bpy.data.objects.new('CustomCamera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    
    # THREE.js uses a different coordinate system:
    # THREE.js: Y is up, Z is forward, X is right
    # Blender: Z is up, Y is forward, X is right
    # We need to convert the coordinates
    
    # Set camera position with coordinate system conversion
    # Using the values from the camera parameters
    threejs_pos = mathutils.Vector((
        CAMERA_PARAMS["position"]["x"], 
        CAMERA_PARAMS["position"]["y"], 
        CAMERA_PARAMS["position"]["z"]
    ))
    
    # Convert from Three.js to Blender coordinate system
    cam_obj.location.x = threejs_pos.x
    cam_obj.location.y = -threejs_pos.z  # Three.js Z axis is Blender -Y axis
    cam_obj.location.z = threejs_pos.y   # Three.js Y axis is Blender Z axis
    
    # Set camera properties from camera parameters
    cam_obj.data.type = 'PERSP'
    cam_obj.data.lens_unit = 'FOV'
    cam_obj.data.angle = math.radians(CAMERA_PARAMS["fov"])  # Convert FOV degrees to radians
    cam_obj.data.clip_start = CAMERA_PARAMS["near"]
    cam_obj.data.clip_end = CAMERA_PARAMS["far"]
    
    # Set the aspect ratio from camera parameters
    bpy.context.scene.render.pixel_aspect_x = 1
    bpy.context.scene.render.pixel_aspect_y = 1
    aspect_ratio = CAMERA_PARAMS["aspect_ratio"]
    bpy.context.scene.render.resolution_x = 1920
    
    # Create a target object that the camera will look at
    target = bpy.data.objects.new('CustomCameraTarget', None)
    bpy.context.scene.collection.objects.link(target)
    
    # Set the target position (using the values from camera parameters)
    target_pos = mathutils.Vector((
        CAMERA_PARAMS["target"]["x"],
        CAMERA_PARAMS["target"]["y"],
        CAMERA_PARAMS["target"]["z"]
    ))
    
    # Convert the target position to Blender coordinates
    target.location.x = target_pos.x
    target.location.y = -target_pos.z  # Three.js Z axis is Blender -Y axis
    target.location.z = target_pos.y   # Three.js Y axis is Blender Z axis
    
    # Add a Track To constraint to make the camera look at the target
    track_to = cam_obj.constraints.new(type='TRACK_TO')
    track_to.target = target
    track_to.track_axis = 'TRACK_NEGATIVE_Z'  # -Z is forward in Blender camera
    track_to.up_axis = 'UP_Y'  # Y is up in Blender
    
    # Set the camera as the active camera
    bpy.context.scene.camera = cam_obj
    
    print(f"New camera created at position: {cam_obj.location}")
    print(f"Camera looking at target: {target.location}")
    print(f"Camera FOV: {CAMERA_PARAMS['fov']} degrees, aspect ratio: {aspect_ratio}")
    
    return cam_obj

def setup_custom_lighting():
    """Set up custom lighting with the provided parameters, converting from Three.js to Blender coordinate system"""
    # Remove existing point lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'POINT':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Create a new point light
    light_data = bpy.data.lights.new(name="PointLight", type='POINT')
    light_obj = bpy.data.objects.new("PointLight", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    
    # Set light position (adjusting from Three.js to Blender coordinates)
    # THREE.js: Y is up, Z is forward, X is right
    # Blender: Z is up, Y is forward, X is right
    threejs_pos = mathutils.Vector((
        LIGHTING_PARAMS["position"]["x"],
        LIGHTING_PARAMS["position"]["y"],
        LIGHTING_PARAMS["position"]["z"]
    ))
    
    light_obj.location.x = threejs_pos.x
    light_obj.location.y = -threejs_pos.z  # Three.js Z axis is Blender -Y axis
    light_obj.location.z = threejs_pos.y   # Three.js Y axis is Blender Z axis
    
    # Convert hex color to RGB
    hex_color = LIGHTING_PARAMS["color"]
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    light_data.color = (r, g, b)
    
    # Set light properties
    light_data.energy = LIGHTING_PARAMS["energy"]
    
    # Modern Blender (4.x) uses different attributes for light falloff
    light_data.use_custom_distance = True
    light_data.cutoff_distance = LIGHTING_PARAMS["distance"]
    
    # Setup world environment
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    
    # Set up HDRI environment
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    world_links = bpy.context.scene.world.node_tree.links
    
    # Clear existing nodes
    world_nodes.clear()
    
    # Create new nodes
    output_node = world_nodes.new(type='ShaderNodeOutputWorld')
    background_node = world_nodes.new(type='ShaderNodeBackground')
    
    # Set exposure
    background_node.inputs['Strength'].default_value = ENVIRONMENT_PARAMS["exposure"]
    
    # Morning HDRI preset (approximate)
    background_node.inputs['Color'].default_value = ENVIRONMENT_PARAMS["background_color"]
    
    # Connect nodes
    world_links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
    
    return light_obj

def setup_gpu_rendering():
    """Set up GPU rendering, with fallback to CPU if needed/configured"""
    try:
        # First check if CUDA or OptiX is available
        has_gpu = False
        
        # Check for CUDA devices
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        bpy.context.preferences.addons["cycles"].preferences.refresh_devices()
        cuda_devices = bpy.context.preferences.addons["cycles"].preferences.devices
        
        # If no CUDA devices, try OptiX
        if not any(device.use for device in cuda_devices):
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX"
            bpy.context.preferences.addons["cycles"].preferences.refresh_devices()
            optix_devices = bpy.context.preferences.addons["cycles"].preferences.devices
            
            # Check if any OptiX devices are available
            has_gpu = any(device.use for device in optix_devices)
        else:
            has_gpu = True
        
        # If GPU is available, use it
        if has_gpu:
            bpy.context.scene.cycles.device = "GPU"
            print("GPU rendering enabled")
            return True
        else:
            # No GPU available
            if GPU_FALLBACK_TO_CPU:
                print("No GPU detected. Falling back to CPU rendering.")
                bpy.context.scene.cycles.device = "CPU"
                return False
            else:
                raise RuntimeError("No GPU available and fallback to CPU is disabled")
    except Exception as e:
        # Handle any errors
        if GPU_FALLBACK_TO_CPU:
            print(f"Error setting up GPU: {e}. Falling back to CPU rendering.")
            bpy.context.scene.cycles.device = "CPU"
            return False
        else:
            raise RuntimeError(f"Failed to set up GPU rendering: {e}")

def parse_camera_json(json_str):
    try:
        camera_data = json.loads(json_str)
        # Validate required fields
        required_fields = ["position", "target", "fov"]
        for field in required_fields:
            if field not in camera_data:
                print(f"Warning: Missing required camera field '{field}'. Using default value.")
                camera_data[field] = CAMERA_PARAMS[field]
        
        # Set defaults for optional fields if not provided
        camera_data.setdefault("up", CAMERA_PARAMS["up"])
        camera_data.setdefault("aspect_ratio", CAMERA_PARAMS["aspect_ratio"])
        camera_data.setdefault("near", CAMERA_PARAMS["near"])
        camera_data.setdefault("far", CAMERA_PARAMS["far"])
        camera_data.setdefault("productNameList", CAMERA_PARAMS["productNameList"])
        
        # Validate position and target are objects with x, y, z
        for vector_field in ["position", "target", "up"]:
            if not all(key in camera_data[vector_field] for key in ["x", "y", "z"]):
                print(f"Warning: Invalid {vector_field} format. Using default values.")
                camera_data[vector_field] = CAMERA_PARAMS[vector_field]
        
        # Validate productNameList is a list of strings
        if not isinstance(camera_data["productNameList"], list):
            print("Warning: productNameList must be a list of strings. Using default empty list.")
            camera_data["productNameList"] = CAMERA_PARAMS["productNameList"]
        else:
            for sku_id in camera_data["productNameList"]:
                if not isinstance(sku_id, str):
                    print(f"Warning: Invalid SKU ID {sku_id} in productNameList, must be a string. Using default empty list.")
                    camera_data["productNameList"] = CAMERA_PARAMS["productNameList"]
                    break
        
        return camera_data
    except json.JSONDecodeError as e:
        print(f"Error parsing camera JSON: {e}")
        return CAMERA_PARAMS
    except Exception as e:
        print(f"Unexpected error parsing camera data: {e}")
        return CAMERA_PARAMS

def parse_lighting_json(json_str):
    try:
        lighting_data = json.loads(json_str)
        # Validate required fields
        required_fields = ["position", "color", "energy"]
        for field in required_fields:
            if field not in lighting_data:
                print(f"Warning: Missing required lighting field '{field}'. Using default value.")
                lighting_data[field] = LIGHTING_PARAMS[field]
        
        # Set defaults for optional fields if not provided
        lighting_data.setdefault("distance", LIGHTING_PARAMS["distance"])
        lighting_data.setdefault("decay", LIGHTING_PARAMS["decay"])
        
        # Validate position is an object with x, y, z
        if not all(key in lighting_data["position"] for key in ["x", "y", "z"]):
            print(f"Warning: Invalid position format. Using default values.")
            lighting_data["position"] = LIGHTING_PARAMS["position"]
        
        # Validate color is a hex string
        if not isinstance(lighting_data["color"], str) or not all(c in "0123456789abcdefABCDEF" for c in lighting_data["color"]):
            print(f"Warning: Invalid color format. Using default values.")
            lighting_data["color"] = LIGHTING_PARAMS["color"]
        
        return lighting_data
    except json.JSONDecodeError as e:
        print(f"Error parsing lighting JSON: {e}")
        return LIGHTING_PARAMS
    except Exception as e:
        print(f"Unexpected error parsing lighting data: {e}")
        return LIGHTING_PARAMS

def generate_masks(args):
    print("Starting mask generation...")
    
    # Check if mutually exclusive options are both set
    if args.individual_masks_only and args.combined_mask_only:
        print("Warning: Both individual-masks-only and combined-mask-only flags are set.")
        print("Will generate both individual and combined masks.")
        args.individual_masks_only = False
        args.combined_mask_only = False
    
    # Save original materials for all objects that need masks
    original_materials = {}
    
    # Set mask rendering settings
    original_engine, original_film_transparent, original_resolution_percentage = setup_mask_render_settings()
    print(f"Switched to {MASK_PARAMS['render_engine']} for mask generation")
    
    # First, identify all objects we need to create masks for
    mask_products = []  # This will store tuples of (sku_id, [mesh_objects])
    
    # Use SKU IDs from CAMERA_PARAMS["productNameList"] if available, otherwise fall back to args.object
    sku_ids = CAMERA_PARAMS.get("productNameList", []) if args.generate_mask else []
    if not sku_ids and args.object:
        print("No valid productNameList found in camera JSON. Falling back to -O arguments.")
        sku_ids = args.object

    # Add products specified by SKU IDs
    for sku_id in sku_ids:
        # Try to find an object with this exact name (case sensitive)
        if sku_id in bpy.data.objects:
            parent_obj = bpy.data.objects[sku_id]
            if parent_obj.type == 'EMPTY' or parent_obj.type == 'MESH':
                # Get all descendant meshes
                meshes = get_all_mesh_descendants(parent_obj)
                if meshes:
                    mask_products.append((sku_id, meshes))
                    print(f"Will generate mask for product: {sku_id} (contains {len(meshes)} meshes)")
                else:
                    print(f"Warning: SKU {sku_id} has no mesh descendants")
            else:
                print(f"Warning: {sku_id} is not an EMPTY or MESH object")
        else:
            print(f"Warning: Could not find object with SKU ID: {sku_id}")
    
    # Find objects based on MASK_OBJECTS types (skip if no-auto-detect is set)
    if not args.no_auto_detect:
        for obj in bpy.data.objects:
            if obj.type == 'EMPTY' or obj.type == 'MESH':
                for mask_type in MASK_OBJECTS:
                    if mask_type.lower() in obj.name.lower():
                        # Check if this object is already in our list
                        already_added = any(sku_id == obj.name for sku_id, _ in mask_products)
                        if not already_added:
                            meshes = get_all_mesh_descendants(obj)
                            if meshes:
                                mask_products.append((obj.name, meshes))
                                print(f"Will generate mask for detected product: {obj.name} (type: {mask_type}, contains {len(meshes)} meshes)")
    
    # No need to process if no products to mask
    if not mask_products:
        print("No products found for mask generation")
    else:
        # Save original materials for all meshes
        for sku_id, meshes in mask_products:
            for mesh in meshes:
                original_materials[mesh.name] = [mat for mat in mesh.data.materials]
        
        # Hide all objects first
        for obj in bpy.data.objects:
            obj.hide_render = True
        
        # Process each product (skip if combined-mask-only is set)
        if not args.combined_mask_only:
            for sku_id, meshes in mask_products:
                # Create white materials for all meshes in this product
                for mesh in meshes:
                    create_mask_material(mesh)
                    mesh.hide_render = False
                
                # Render the mask
                mask_path = f'{args.outdir}/mask_{sku_id}.png'
                print(f"Rendering mask for product {sku_id} to {mask_path}")
                bpy.context.scene.render.filepath = mask_path
                bpy.ops.render.render(write_still=True)
                
                replace_image_with_binary_mask(mask_path)

                # Hide the meshes again
                for mesh in meshes:
                    mesh.hide_render = True
        
        # Generate combined mask with all products if needed (skip if individual-masks-only is set)
        if len(mask_products) > 1 and not args.individual_masks_only:
            # Create a dictionary to track which meshes belong to which product
            product_meshes = {}
            for sku_id, meshes in mask_products:
                product_meshes[sku_id] = meshes
            
            # Create white materials for all meshes in all products
            all_meshes = []
            for sku_id, meshes in mask_products:
                for mesh in meshes:
                    create_mask_material(mesh)
                    mesh.hide_render = False
                all_meshes.extend(meshes)
            
            # Render the combined mask
            combined_mask_path = f'{args.outdir}/mask_all_products.png'
            print(f"Rendering combined mask for all products to {combined_mask_path}")
            bpy.context.scene.render.filepath = combined_mask_path
            bpy.ops.render.render(write_still=True)

            replace_image_with_binary_mask(combined_mask_path)
            
            # Hide all meshes again
            for mesh in all_meshes:
                mesh.hide_render = True
        
        # Restore original materials
        for mesh_name, materials in original_materials.items():
            if mesh_name in bpy.data.objects:
                mesh = bpy.data.objects[mesh_name]
                mesh.data.materials.clear()
                for mat in materials:
                    mesh.data.materials.append(mat)
        
        # Restore original render settings
        restore_render_settings(original_engine, original_film_transparent, original_resolution_percentage)
    
    print("Mask generation complete!")

def fix_overlapping_floors():
    """Detect Floor0 and Floor meshes and move Floor down by 0.01 units to avoid z-fighting"""
    floor_objects = []
    
    # Find Floor and Floor0 objects
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name in ['Floor', 'Floor0']:
            floor_objects.append(obj)
    
    # Check if we have both Floor and Floor0
    floor_obj = None
    floor0_obj = None
    
    for obj in floor_objects:
        if obj.name == 'Floor':
            floor_obj = obj
        elif obj.name == 'Floor0':
            floor0_obj = obj
    
    # If both exist, move Floor down by 0.01 units
    if floor_obj and floor0_obj:
        print(f"Found overlapping floors: {floor_obj.name} and {floor0_obj.name}")
        print(f"Moving {floor_obj.name} down by 0.01 units to avoid z-fighting")
        
        # Move Floor down by 0.01 units in Z-axis
        floor_obj.location.z -= 0.01
        
        print(f"Floor position adjusted to: {floor_obj.location}")
    else:
        print("No overlapping Floor/Floor0 meshes detected")

parser = argparse.ArgumentParser()
parser.add_argument('-O', '--object', action='append', help='Object names for replacement or mask generation')
parser.add_argument('-R', '--replacement', help='Path to replacement GLB file')
parser.add_argument('-d', '--outdir', default='.', help='Output directory for renders and masks')
parser.add_argument('-r', '--resolution', type=int, default=1024, help='Render resolution')
parser.add_argument('-C', '--camera', type=int, default=0, help='Camera index')
parser.add_argument('-M', '--generate-mask', action='store_true', help='Generate black and white masks for objects')
parser.add_argument('--skip-render', action='store_true', help='Skip main render and only generate masks')
parser.add_argument('--use-custom-lighting', action='store_true', default=True, help='Use custom lighting settings')
parser.add_argument('--individual-masks-only', action='store_true', help='Generate only individual masks, no combined mask')
parser.add_argument('--combined-mask-only', action='store_true', help='Generate only combined mask, no individual masks')
parser.add_argument('--no-auto-detect', action='store_true', help='Disable automatic detection of objects based on name')
parser.add_argument('--high-quality', action='store_true', help='Enable higher quality rendering with denoising')
parser.add_argument('--force-cpu', action='store_true', help='Force CPU rendering even if GPU is available')
parser.add_argument('--camera-json', type=str, help='Camera parameters in JSON format')
parser.add_argument('--lighting-json', type=str, help='Lighting parameters in JSON format')
parser.add_argument('mainfile', help='Main Blender or GLB file')

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
args = parser.parse_args(args)

# Override default camera settings if provided via command line
if args.camera_json:
    CAMERA_PARAMS = parse_camera_json(args.camera_json)
    print("Using custom camera parameters from command line")

# Override default lighting settings if provided via command line
if args.lighting_json:
    LIGHTING_PARAMS = parse_lighting_json(args.lighting_json)
    print("Using custom lighting parameters from command line")

if args.mainfile.endswith('.glb') or args.mainfile.endswith('.gltf'):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=args.mainfile)
else:
    bpy.ops.wm.open_mainfile(filepath=args.mainfile)

fix_overlapping_floors()

# Find and rename any existing cameras in the GLB to avoid conflicts
for obj in bpy.data.objects:
    if obj.type == 'CAMERA':
        obj.name = f"Original_{obj.name}"
        print(f"Renamed existing camera to {obj.name}")

# Always create a new custom camera with our exact parameters
custom_camera = setup_custom_camera()
custom_light = setup_custom_lighting()

# Make sure our new camera is set as the active camera
bpy.context.scene.camera = custom_camera
print(f"Using custom camera: {custom_camera.name}")
print(f"Camera position: {custom_camera.location}")

print(f"Light position: {custom_light.location}")
print(f"Light energy: {custom_light.data.energy}")

bpy.ops.file.find_missing_files(directory='textures')
bpy.ops.file.make_paths_relative()

if args.replacement is not None and args.object is not None and len(args.object) > 0:
    replace(args.object[0], args.replacement)

aspect = bpy.context.scene.render.resolution_x/bpy.context.scene.render.resolution_y
if aspect > 1:
    bpy.context.scene.render.resolution_x = args.resolution
    bpy.context.scene.render.resolution_y = 1080
else:
    bpy.context.scene.render.resolution_y = args.resolution
    bpy.context.scene.render.resolution_x = 1920

# Skip the main render if requested (to save memory)
if not args.skip_render:
    # Set rendering parameters
    bpy.context.scene.render.resolution_percentage = RENDER_PARAMS["resolution_percentage"]
    bpy.context.scene.render.film_transparent = RENDER_PARAMS["film_transparent"]
    
    # Set up Cycles renderer with appropriate settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = RENDER_PARAMS["samples"]
    bpy.context.scene.cycles.preview_samples = RENDER_PARAMS["preview_samples"]
    
    # Setup GPU or CPU rendering based on global settings and command-line arguments
    if USE_GPU and not args.force_cpu:
        setup_gpu_rendering()
    else:
        print("Using CPU for rendering")
        bpy.context.scene.cycles.device = 'CPU'
    
    # Handle denoising based on global setting
    if USE_DENOISING:
        print("Enabling denoising for higher quality rendering")
        if hasattr(bpy.context.scene.cycles, 'use_denoising'):
            bpy.context.scene.cycles.use_denoising = True
            if hasattr(bpy.context.scene.cycles, 'denoiser'):
                try:
                    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
                    print("Using OpenImageDenoise denoiser")
                except (TypeError, ValueError):
                    try:
                        bpy.context.scene.cycles.denoiser = 'OPTIX'
                        print("Using OptiX denoiser")
                    except (TypeError, ValueError):
                        print("Warning: No compatible denoiser found. Denoising may not work properly.")
            if hasattr(bpy.context.scene.cycles, 'denoising_strength'):
                bpy.context.scene.cycles.denoising_strength = RENDER_PARAMS["denoising_strength"]
        else:
            bpy.context.view_layer.cycles.use_denoising = True
            bpy.context.view_layer.cycles.denoising_strength = RENDER_PARAMS["denoising_strength"]
    else:
        print("Denoising disabled")
        if hasattr(bpy.context.scene.cycles, 'use_denoising'):
            bpy.context.scene.cycles.use_denoising = False
        else:
            bpy.context.view_layer.cycles.use_denoising = False
    
    bpy.context.scene.render.filepath = f'{args.outdir}/room_render.png'
    bpy.ops.render.render(write_still=True)

# Generate masks if requested
if args.generate_mask:
    generate_masks(args)

# Make all objects visible again
for obj in bpy.data.objects:
    obj.hide_render = False