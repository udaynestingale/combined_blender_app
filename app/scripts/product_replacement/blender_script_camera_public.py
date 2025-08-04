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
import urllib.request
import tempfile
import glob

# Global rendering configuration options
USE_DENOISING = True  # Set to False to disable denoising
USE_GPU = True  # Set to False to force CPU rendering
GPU_FALLBACK_TO_CPU = True  # Set to False to throw an error if GPU not available

# List of object types to generate masks for
MASK_OBJECTS = ["abcdefg"]

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
    "productNameList": [],  # Default empty list for productNameList
    "sensor_fit": "AUTO",
    "sensor_width": 36,
    "lens": 57,
    "clip_start": 2.51,
    "clip_end": 1000,
    "shift_x": 0,
    "shift_y": -0.170,
    "type": "PERSP",
    "resolution_x": 1920,
    "resolution_y": 1664,
    "resolution_percentage": 200,
    "pixel_aspect_x": 1,
    "pixel_aspect_y": 1
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
    "exposure": 150,
    "background_color": (0.8, 0.8, 0.95, 1.0)  # Soft blue morning color
}

# Rendering configuration parameters
RENDER_PARAMS = {
    "samples": 20,
    "preview_samples": 64,
    "denoising_strength": 1.0,  # Higher values for stronger denoising (0.0-1.0)
    "resolution_percentage": 100,
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

def download_glb(url, temp_dir):
    """Download a GLB file from a URL to a temporary location and return the path."""
    try:
        # Extract filename from URL or use a default name
        filename = os.path.basename(url) or "temp.glb"
        temp_path = os.path.join(temp_dir, filename)
        
        # Download the file
        print(f"Downloading GLB from {url} to {temp_path}")
        urllib.request.urlretrieve(url, temp_path)
        
        if not os.path.exists(temp_path):
            print(f"Failed to download GLB from {url}")
            return None
        
        return temp_path
    except Exception as e:
        print(f"Error downloading GLB from {url}: {e}")
        return None

def replace(from_name, to_glb, new_name=None):
    if from_name not in bpy.data.objects:
        print(f"Object {from_name} not found")
        return False

    original_object = bpy.data.objects[from_name]
    if original_object.type != 'EMPTY':
        print(f"Object {from_name} is not an EMPTY object, cannot proceed with replacement")
        return False

    # Find the mesh child of the original EMPTY object
    original_mesh = None
    for child in original_object.children:
        if child.type == 'MESH':
            original_mesh = child
            break
    if not original_mesh:
        print(f"No mesh child found for EMPTY object {from_name}")
        return False

    print(f"original_mesh and original_object {original_mesh, original_object}")
    print(f"original_mesh.rotation_quaternion {original_mesh.rotation_quaternion}")

    # Store the original EMPTY object's location and rotation
    original_location = original_object.location.copy()
    original_rotation = original_mesh.rotation_quaternion.copy()

    # Get the mesh's WORLD position (not local position relative to parent)
    original_mesh_world_pos = original_mesh.matrix_world.translation.copy()
    
    print(f"Original EMPTY location: {original_location}")
    print(f"Original mesh world location: {original_mesh_world_pos}")

    # Handle GLB source (URL or local file)
    temp_file = None
    if to_glb.startswith(('http://', 'https://')):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = download_glb(to_glb, temp_dir)
            if not temp_file:
                print(f"Failed to download GLB from {to_glb}")
                return False
            to_glb = temp_file
            # Proceed with replacement
            return perform_replace(from_name, to_glb, new_name, original_object, original_mesh, original_location, original_rotation, original_mesh_world_pos)
    else:
        # Local file, proceed directly
        return perform_replace(from_name, to_glb, new_name, original_object, original_mesh, original_location, original_rotation, original_mesh_world_pos)

def perform_replace(from_name, to_glb, new_name, original_object, original_mesh, original_location, original_rotation, original_mesh_world_pos):
    """Perform the actual replacement of the object with the GLB file."""
    # Remove the original object and its children from the scene
    children_to_delete = [child for child in original_object.children]
    
    for child in children_to_delete:
        print(f"Removing child object: {child.name}")
        bpy.data.objects.remove(child, do_unlink=True)
    
    print(f"Removing parent object: {original_object.name}")
    bpy.data.objects.remove(original_object, do_unlink=True)

    # Import the new GLB
    existing_objects = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=to_glb)
    imported_objects = set(bpy.data.objects) - existing_objects

    if not imported_objects:
        print(f"No objects imported from {to_glb}")
        return False

    # Create an EMPTY object to parent the imported object
    empty_name = new_name if new_name else from_name
    empty_obj = bpy.data.objects.new(empty_name, None)
    empty_obj.empty_display_type = 'PLAIN_AXES'
    bpy.context.scene.collection.objects.link(empty_obj)

    # Set the EMPTY object's location and rotation to match the original
    empty_obj.location = original_location

    # Parent the imported object to the EMPTY object
    imported_object = next(iter(imported_objects))
    imported_object.rotation_quaternion = original_rotation
    imported_object.parent = empty_obj
    print(f"original_rotation {original_rotation}")

    # Position the imported object to match the original mesh's world position
    imported_object.location = original_mesh_world_pos - original_location
    
    print(f"New imported mesh location: {imported_object.location}")
    print(f"New imported mesh world location: {imported_object.matrix_world.translation}")

    # Name the imported object to distinguish it from the EMPTY parent
    imported_object.name = f"{empty_name}_mesh" if new_name else from_name

    return True

def get_all_mesh_descendants(parent_obj):
    """Recursively find all mesh objects that are descendants of the given parent object"""
    mesh_objects = []
    
    if parent_obj.type == 'MESH':
        mesh_objects.append(parent_obj)
    
    for child in parent_obj.children:
        mesh_objects.extend(get_all_mesh_descendants(child))
    
    return mesh_objects

def create_mask_material(obj):
    """Create a material that outputs pure white with alpha"""
    mat = bpy.data.materials.new(name=f"Mask_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)
    bsdf.inputs['Alpha'].default_value = 1.0
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    mat.blend_method = 'OPAQUE'
    
    return mat

def post_process_mask(image_path, output_path):
    """Convert RGBA to 1-bit alpha channel"""
    from PIL import Image
    import numpy as np
    
    img = Image.open(image_path)
    alpha = np.array(img)[..., 3]
    
    binary_alpha = ((alpha > 127) * 255).astype(np.uint8)
    
    Image.fromarray(binary_alpha).save(
        output_path,
        optimize=True,
        bits=1
    )

def setup_mask_render_settings():
    """Set up rendering settings for alpha-only binary masks"""
    original_engine = bpy.context.scene.render.engine
    original_film_transparent = bpy.context.scene.render.film_transparent
    original_resolution_percentage = bpy.context.scene.render.resolution_percentage
    
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.eevee.taa_render_samples = 1
    bpy.context.scene.eevee.use_gtao = False
    
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.render.image_settings.compression = 0
    
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("MaskWorld")
    bpy.context.scene.world.use_nodes = True
    nodes = bpy.context.scene.world.node_tree.nodes
    nodes.clear()
    output = nodes.new(type='ShaderNodeOutputWorld')
    
    return original_engine, original_film_transparent, original_resolution_percentage

def replace_image_with_binary_mask(filepath):
    """Convert rendered alpha to pure binary (0 or 255)"""
    image = bpy.data.images.load(filepath)
    alpha_pixels = list(image.pixels)[3::4]
    
    binary_pixels = []
    for alpha in alpha_pixels:
        if alpha > 0.5:
            binary_pixels.extend([1.0, 1.0, 1.0, 1.0])
        else:
            binary_pixels.extend([0.0, 0.0, 0.0, 0.0])
    
    width, height = image.size
    binary_image = bpy.data.images.new("binary_mask", width, height)
    binary_image.pixels = binary_pixels
    binary_image.filepath_raw = filepath
    binary_image.file_format = 'PNG'
    binary_image.save()
    
    bpy.data.images.remove(image)

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
    cam_data = bpy.data.cameras.new('CustomCamera')
    cam_obj = bpy.data.objects.new('CustomCamera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    
    threejs_pos = mathutils.Vector((
        CAMERA_PARAMS["position"]["x"], 
        CAMERA_PARAMS["position"]["y"], 
        CAMERA_PARAMS["position"]["z"]
    ))
    
    cam_obj.location.x = threejs_pos.x
    cam_obj.location.y = -threejs_pos.z
    cam_obj.location.z = threejs_pos.y
    
    cam_obj.data.type = CAMERA_PARAMS["type"]
    cam_obj.data.sensor_fit = CAMERA_PARAMS["sensor_fit"]
    cam_obj.data.sensor_width = CAMERA_PARAMS["sensor_width"]
    cam_obj.data.lens = CAMERA_PARAMS["lens"]
    cam_obj.data.clip_start = CAMERA_PARAMS["clip_start"]
    cam_obj.data.clip_end = CAMERA_PARAMS["clip_end"]
    cam_obj.data.shift_x = CAMERA_PARAMS["shift_x"]
    cam_obj.data.shift_y = CAMERA_PARAMS["shift_y"]
    cam_obj.data.angle = math.radians(CAMERA_PARAMS["fov"])
    
    bpy.context.scene.render.pixel_aspect_x = CAMERA_PARAMS["pixel_aspect_x"]
    bpy.context.scene.render.pixel_aspect_y = CAMERA_PARAMS["pixel_aspect_y"]
    bpy.context.scene.render.resolution_x = CAMERA_PARAMS["resolution_x"]
    bpy.context.scene.render.resolution_y = CAMERA_PARAMS["resolution_y"]
    bpy.context.scene.render.resolution_percentage = CAMERA_PARAMS["resolution_percentage"]
    
    target = bpy.data.objects.new('CustomCameraTarget', None)
    bpy.context.scene.collection.objects.link(target)
    
    target_pos = mathutils.Vector((
        CAMERA_PARAMS["target"]["x"],
        CAMERA_PARAMS["target"]["y"],
        CAMERA_PARAMS["target"]["z"]
    ))
    
    target.location.x = target_pos.x
    target.location.y = -target_pos.z
    target.location.z = target_pos.y
    
    track_to = cam_obj.constraints.new(type='TRACK_TO')
    track_to.target = target
    track_to.track_axis = 'TRACK_NEGATIVE_Z'
    track_to.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = cam_obj
    
    print(f"New camera created at position: {cam_obj.location}")
    print(f"Camera looking at target: {target.location}")
    print(f"Camera FOV: {CAMERA_PARAMS['fov']} degrees, resolution: {CAMERA_PARAMS['resolution_x']}x{CAMERA_PARAMS['resolution_y']}")
    
    return cam_obj

def setup_custom_lighting():
    """Set up custom lighting or environment map based on command-line arguments"""
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'POINT':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    world_links = bpy.context.scene.world.node_tree.links
    
    world_nodes.clear()
    
    output_node = world_nodes.new(type='ShaderNodeOutputWorld')
    background_node = world_nodes.new(type='ShaderNodeBackground')
    
    if args.use_environment_map:
        env_path = args.use_environment_map
        if os.path.exists(env_path):
            print(f"Loading environment texture from: {env_path}")
            env_tex_node = world_nodes.new(type='ShaderNodeTexEnvironment')
            env_tex_node.image = bpy.data.images.load(env_path)
            world_links.new(env_tex_node.outputs['Color'], background_node.inputs['Color'])
            background_node.inputs['Strength'].default_value = ENVIRONMENT_PARAMS["exposure"]
        else:
            print(f"Error: Environment texture path does not exist: {env_path}")
            background_node.inputs['Color'].default_value = ENVIRONMENT_PARAMS["background_color"]
            background_node.inputs['Strength'].default_value = ENVIRONMENT_PARAMS["exposure"]
    else:
        light_data = bpy.data.lights.new(name="PointLight", type='POINT')
        light_obj = bpy.data.objects.new("PointLight", light_data)
        bpy.context.scene.collection.objects.link(light_obj)
        
        threejs_pos = mathutils.Vector((
            LIGHTING_PARAMS["position"]["x"],
            LIGHTING_PARAMS["position"]["y"],
            LIGHTING_PARAMS["position"]["z"]
        ))
        
        light_obj.location.x = threejs_pos.x
        light_obj.location.y = -threejs_pos.z
        light_obj.location.z = threejs_pos.y
        
        hex_color = LIGHTING_PARAMS["color"]
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        light_data.color = (r, g, b)
        
        light_data.energy = LIGHTING_PARAMS["energy"]
        
        light_data.use_custom_distance = True
        light_data.cutoff_distance = LIGHTING_PARAMS["distance"]
        
        background_node.inputs['Color'].default_value = ENVIRONMENT_PARAMS["background_color"]
        background_node.inputs['Strength'].default_value = ENVIRONMENT_PARAMS["exposure"]
    
    world_links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
    
    return light_obj if not args.use_environment_map else None

def find_environment_texture():
    """Find environment texture files in the current directory"""
    current_dir = os.getcwd()
    
    env_extensions = ['*.hdr', '*.exr', '*.hdri', '*.tiff', '*.tif', '*.jpg', '*.jpeg', '*.png']
    
    env_patterns = [
        '*environment*', '*env*', '*hdri*', '*world*', '*sky*', 
        '*studio*', '*lighting*', '*background*', '*morning*', 
        '*sunset*', '*dawn*', '*outdoor*', '*indoor*'
    ]
    
    found_textures = []
    
    for pattern in env_patterns:
        for ext in env_extensions:
            search_pattern = os.path.join(current_dir, f"{pattern}{ext}")
            matches = glob.glob(search_pattern, recursive=False)
            found_textures.extend(matches)
    
    for ext in ['*.hdr', '*.exr', '*.hdri']:
        search_pattern = os.path.join(current_dir, ext)
        matches = glob.glob(search_pattern, recursive=False)
        found_textures.extend(matches)
    
    found_textures = list(set(found_textures))
    
    if not found_textures:
        return None
    
    priority_files = []
    other_files = []
    
    for texture in found_textures:
        ext = os.path.splitext(texture)[1].lower()
        if ext in ['.hdr', '.exr', '.hdri']:
            priority_files.append(texture)
        else:
            other_files.append(texture)
    
    if priority_files:
        selected = priority_files[0]
    else:
        selected = other_files[0]
    
    print(f"Found environment texture: {selected}")
    return selected

def setup_gpu_rendering():
    """Set up GPU rendering, with fallback to CPU if needed/configured"""
    try:
        has_gpu = False
        
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        bpy.context.preferences.addons["cycles"].preferences.refresh_devices()
        cuda_devices = bpy.context.preferences.addons["cycles"].preferences.devices
        
        if not any(device.use for device in cuda_devices):
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX"
            bpy.context.preferences.addons["cycles"].preferences.refresh_devices()
            optix_devices = bpy.context.preferences.addons["cycles"].preferences.devices
            
            has_gpu = any(device.use for device in optix_devices)
        else:
            has_gpu = True
        
        if has_gpu:
            bpy.context.scene.cycles.device = "GPU"
            print("GPU rendering enabled")
            return True
        else:
            if GPU_FALLBACK_TO_CPU:
                print("No GPU detected. Falling back to CPU rendering.")
                bpy.context.scene.cycles.device = "CPU"
                return False
            else:
                raise RuntimeError("No GPU available and fallback to CPU is disabled")
    except Exception as e:
        if GPU_FALLBACK_TO_CPU:
            print(f"Error setting up GPU: {e}. Falling back to CPU rendering.")
            bpy.context.scene.cycles.device = "CPU"
            return False
        else:
            raise RuntimeError(f"Failed to set up GPU rendering: {e}")

def parse_camera_json(json_str):
    try:
        camera_data = json.loads(json_str)
        required_fields = ["position", "target", "fov"]
        for field in required_fields:
            if field not in camera_data:
                print(f"Warning: Missing required camera field '{field}'. Using default value.")
                camera_data[field] = CAMERA_PARAMS[field]
        
        camera_data.setdefault("up", CAMERA_PARAMS["up"])
        camera_data.setdefault("aspect_ratio", CAMERA_PARAMS["aspect_ratio"])
        camera_data.setdefault("near", CAMERA_PARAMS["near"])
        camera_data.setdefault("far", CAMERA_PARAMS["far"])
        camera_data.setdefault("productNameList", CAMERA_PARAMS["productNameList"])
        camera_data.setdefault("sensor_fit", CAMERA_PARAMS["sensor_fit"])
        camera_data.setdefault("sensor_width", CAMERA_PARAMS["sensor_width"])
        camera_data.setdefault("lens", CAMERA_PARAMS["lens"])
        camera_data.setdefault("clip_start", CAMERA_PARAMS["clip_start"])
        camera_data.setdefault("clip_end", CAMERA_PARAMS["clip_end"])
        camera_data.setdefault("shift_x", CAMERA_PARAMS["shift_x"])
        camera_data.setdefault("shift_y", CAMERA_PARAMS["shift_y"])
        camera_data.setdefault("type", CAMERA_PARAMS["type"])
        camera_data.setdefault("resolution_x", CAMERA_PARAMS["resolution_x"])
        camera_data.setdefault("resolution_y", CAMERA_PARAMS["resolution_y"])
        camera_data.setdefault("resolution_percentage", CAMERA_PARAMS["resolution_percentage"])
        camera_data.setdefault("pixel_aspect_x", CAMERA_PARAMS["pixel_aspect_x"])
        camera_data.setdefault("pixel_aspect_y", CAMERA_PARAMS["pixel_aspect_y"])
        
        for vector_field in ["position", "target", "up"]:
            if not all(key in camera_data[vector_field] for key in ["x", "y", "z"]):
                print(f"Warning: Invalid {vector_field} format. Using default values.")
                camera_data[vector_field] = CAMERA_PARAMS[vector_field]
        
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
        required_fields = ["position", "color", "energy"]
        for field in required_fields:
            if field not in lighting_data:
                print(f"Warning: Missing required lighting field '{field}'. Using default value.")
                lighting_data[field] = LIGHTING_PARAMS[field]
        
        lighting_data.setdefault("distance", LIGHTING_PARAMS["distance"])
        lighting_data.setdefault("decay", LIGHTING_PARAMS["decay"])
        
        if not all(key in lighting_data["position"] for key in ["x", "y", "z"]):
            print(f"Warning: Invalid position format. Using default values.")
            lighting_data["position"] = LIGHTING_PARAMS["position"]
        
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

def parse_replace_product_json(json_str):
    """Parse and validate the --replace-product JSON array."""
    try:
        replace_data = json.loads(json_str)
        if not isinstance(replace_data, list) or len(replace_data) != 3:
            print(f"Error: --replace-product must be a JSON array with exactly 3 elements, got: {replace_data}")
            return None, None, ""
        new_product = replace_data[0]
        existing_product = replace_data[1]
        if not isinstance(existing_product, str):
            print(f"Error: Second element must be a string, got: {existing_product}")
            return None, None, ""
        if not isinstance(new_product, str):
            print(f"Error: First element must be a string (URL or file path), got: {new_product}")
            return None, None, ""
        # Extract name from URL or file path for naming purposes
        if new_product.startswith(('http://', 'https://')):
            # Use the basename of the URL for the name, removing .glb extension
            new_product_name = os.path.splitext(os.path.basename(new_product))[0]
        else:
            # Use the basename without extension for local files
            new_product_name = os.path.splitext(os.path.basename(new_product))[0]
        
        return new_product, existing_product, new_product_name
    except json.JSONDecodeError as e:
        print(f"Error parsing --replace-product JSON: {e}")
        return None, None, ""
    except Exception as e:
        print(f"Unexpected error parsing --replace-product data: {e}")
        return None, None, ""

def generate_masks(args):
    print("Starting mask generation...") #

    # Store the replace_product arguments if they exist
    # This needs to be accessed from outside generate_masks as args.replace_product is a global command-line arg.
    # It's better to pass it in or access the globally available `args` if that's how it's handled.
    # Assuming `args.replace_product` is parsed and available as done in the main script.
    
    if args.individual_masks_only and args.combined_mask_only: #
        print("Warning: Both individual-masks-only and combined-mask-only flags are set.") #
        print("Will generate both individual and combined masks.") #
        args.individual_masks_only = False #
        args.combined_mask_only = False #
    
    original_materials = {} #
    
    original_engine, original_film_transparent, original_resolution_percentage = setup_mask_render_settings() #
    print(f"Switched to {MASK_PARAMS['render_engine']} for mask generation") #
    
    mask_products = [] #
    
    sku_ids = CAMERA_PARAMS.get("productNameList", []) if args.generate_mask else [] #
    if not sku_ids and args.object: #
        print("No valid productNameList found in camera JSON. Falling back to -O arguments.") #
        sku_ids = args.object #

    for sku_id in sku_ids: #
        if sku_id in bpy.data.objects: #
            parent_obj = bpy.data.objects[sku_id] #
            if parent_obj.type == 'EMPTY' or parent_obj.type == 'MESH': #
                meshes = get_all_mesh_descendants(parent_obj) #
                if meshes: #
                    mask_products.append((sku_id, meshes)) #
                    print(f"Will generate mask for product: {sku_id} (contains {len(meshes)} meshes)") #
                else:
                    print(f"Warning: SKU {sku_id} has no mesh descendants") #
            else:
                print(f"Warning: {sku_id} is not an EMPTY or MESH object") #
        else:
            print(f"Warning: Could not find object with SKU ID: {sku_id}") #
    
    if not args.no_auto_detect: #
        for obj in bpy.data.objects: #
            if obj.type == 'EMPTY' or obj.type == 'MESH': #
                for mask_type in MASK_OBJECTS: #
                    if mask_type.lower() in obj.name.lower(): #
                        already_added = any(sku_id == obj.name for sku_id, _ in mask_products) #
                        if not already_added: #
                            meshes = get_all_mesh_descendants(obj) #
                            if meshes: #
                                mask_products.append((obj.name, meshes)) #
                                print(f"Will generate mask for detected product: {obj.name} (type: {mask_type}, contains {len(meshes)} meshes)") #
    
    if not mask_products: #
        print("No products found for mask generation") #
    else:
        # Determine the override name from replace_product if applicable
        override_mask_name = None
        if args.replace_product: #
            try:
                replace_info = json.loads(args.replace_product) #
                if len(replace_info) == 3: #
                    # The name of the object that was replaced (existing_product)
                    new_product = replace_info[0] #
                    replaced_newproduct_name = os.path.splitext(os.path.basename(new_product))[0]
                    # The desired new name for the mask (explicit_new_name)
                    potential_override_name = replace_info[2] #
                    
                    # Check if the product that was replaced is among the mask_products
                    # and if its current sku_id in the list matches the intended new name.
                    # This check is important to ensure we only apply the override
                    # when a replacement actually happened and is relevant to the mask.
                    for i, (mp_sku_id, _) in enumerate(mask_products): #
                        if mp_sku_id == potential_override_name and replaced_newproduct_name in CAMERA_PARAMS.get("productNameList", []): # This logic might be slightly off. If CAMERA_PARAMS is updated, it'll contain the new name.
                            # A more direct check: if the product that was just replaced is now known by its new name
                            # in mask_products, and its new name matches the explicitly provided name, use it.
                            # The most robust way is to just use the new_product_name directly if it's found in mask_products.
                            override_mask_name = potential_override_name #
                            break

                    print(f"---------------------- Override {replaced_newproduct_name} {potential_override_name}")
            except json.JSONDecodeError: #
                print("Warning: Could not parse --replace-product JSON for mask naming override.") #
            except IndexError: #
                print("Warning: --replace-product JSON does not have 3 elements for mask naming override.") #


        for sku_id, meshes in mask_products: #
            for mesh in meshes: #
                original_materials[mesh.name] = [mat for mat in mesh.data.materials] #
        
        for obj in bpy.data.objects: #
            obj.hide_render = True #
        
        if not args.combined_mask_only: #
            for sku_id, meshes in mask_products: #
                for mesh in meshes: #
                    create_mask_material(mesh) #
                    mesh.hide_render = False #
                
                # Apply the override if applicable for the current sku_id
                if replaced_newproduct_name and sku_id == replaced_newproduct_name: #
                    safe_filename = potential_override_name #
                    print(f"Overriding mask filename for {sku_id} to {safe_filename}.png") #
                else:
                    safe_filename = os.path.basename(sku_id) #

                mask_path = f'{args.outdir}/mask_{safe_filename}.png' #
                print(f"Rendering mask for product {sku_id} to {mask_path}") #
                bpy.context.scene.render.filepath = mask_path #
                bpy.ops.render.render(write_still=True) #
                
                replace_image_with_binary_mask(mask_path) #

                for mesh in meshes: #
                    mesh.hide_render = True #
        
        if len(mask_products) > 1 and not args.individual_masks_only: #
            product_meshes = {} #
            for sku_id, meshes in mask_products: #
                product_meshes[sku_id] = meshes #
            
            all_meshes = [] #
            for sku_id, meshes in mask_products: #
                for mesh in meshes: #
                    create_mask_material(obj=mesh) #
                    mesh.hide_render = False #
                all_meshes.extend(meshes) #
            
            combined_mask_path = f'{args.outdir}/mask_all_products.png' #
            print(f"Rendering combined mask for all products to {combined_mask_path}") #
            bpy.context.scene.render.filepath = combined_mask_path #
            bpy.ops.render.render(write_still=True) #

            replace_image_with_binary_mask(combined_mask_path) #
            
            for mesh in all_meshes: #
                mesh.hide_render = True #
        
        for mesh_name, materials in original_materials.items(): #
            if mesh_name in bpy.data.objects: #
                mesh = bpy.data.objects[mesh_name] #
                mesh.data.materials.clear() #
                for mat in materials: #
                    mesh.data.materials.append(mat) #
        
        restore_render_settings(original_engine, original_film_transparent, original_resolution_percentage) #
    
    print("Mask generation complete!") #

def fix_overlapping_floors():
    """Detect Floor0 and Floor meshes and move Floor down by 0.01 units to avoid z-fighting"""
    floor_objects = []
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name in ['Floor', 'Floor0']:
            floor_objects.append(obj)
    
    floor_obj = None
    floor0_obj = None
    
    for obj in floor_objects:
        if obj.name == 'Floor':
            floor_obj = obj
        elif obj.name == 'Floor0':
            floor0_obj = obj
    
    if floor_obj and floor0_obj:
        print(f"Found overlapping floors: {floor_obj.name} and {floor0_obj.name}")
        print(f"Moving {floor_obj.name} down by 0.01 units to avoid z-fighting")
        
        floor_obj.location.z -= 0.01
        
        print(f"Floor position adjusted to: {floor_obj.location}")
    else:
        print("No overlapping Floor/Floor0 meshes detected")

parser = argparse.ArgumentParser(description="Blender script to replace objects and render with custom camera and lighting")
parser.add_argument('--use-environment-map', type=str, help='Path to environment texture (HDR/EXR) to use instead of lighting JSON')
parser.add_argument('--replace-product', type=str, help='JSON array with [new_product_path_or_url, existing_product_name] for product replacement')
parser.add_argument('--use-existing-camera', action='store_true', help='Use existing camera from GLB file instead of custom camera')
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

existing_camera = None
for obj in bpy.data.objects:
    if obj.type == 'CAMERA':
        if args.use_existing_camera and not existing_camera:
            existing_camera = obj
            existing_camera.data.type = CAMERA_PARAMS["type"]
            existing_camera.data.sensor_fit = CAMERA_PARAMS["sensor_fit"]
            existing_camera.data.sensor_width = CAMERA_PARAMS["sensor_width"]
            existing_camera.data.lens = CAMERA_PARAMS["lens"]
            existing_camera.data.clip_start = CAMERA_PARAMS["clip_start"]
            existing_camera.data.clip_end = CAMERA_PARAMS["clip_end"]
            existing_camera.data.shift_x = CAMERA_PARAMS["shift_x"]
            existing_camera.data.shift_y = CAMERA_PARAMS["shift_y"]
            print(f"Applied camera properties to existing camera: {existing_camera.name}")
        else:
            obj.name = f"Original_{obj.name}"
            print(f"Renamed existing camera to {obj.name}")

if args.use_existing_camera and existing_camera:
    custom_camera = existing_camera
    bpy.context.scene.render.resolution_x = CAMERA_PARAMS["resolution_x"]
    bpy.context.scene.render.resolution_y = CAMERA_PARAMS["resolution_y"]
    bpy.context.scene.render.resolution_percentage = CAMERA_PARAMS["resolution_percentage"]
    bpy.context.scene.render.pixel_aspect_x = CAMERA_PARAMS["pixel_aspect_x"]
    bpy.context.scene.render.pixel_aspect_y = CAMERA_PARAMS["pixel_aspect_y"]
    print(f"Using existing camera from GLB: {custom_camera.name}")
    print(f"Render resolution set to {CAMERA_PARAMS['resolution_x']}x{CAMERA_PARAMS['resolution_y']} at {CAMERA_PARAMS['resolution_percentage']}%")
else:
    custom_camera = setup_custom_camera()
    print(f"Created new custom camera: {custom_camera.name}")
    print(f"Render resolution set to {CAMERA_PARAMS['resolution_x']}x{CAMERA_PARAMS['resolution_y']} at {CAMERA_PARAMS['resolution_percentage']}%")

bpy.context.scene.camera = custom_camera
print(f"Using camera: {custom_camera.name}")
print(f"Camera position: {custom_camera.location}")

custom_light = setup_custom_lighting()
if custom_light:
    print(f"Light position: {custom_light.location}")
    print(f"Light energy: {custom_light.data.energy}")
else:
    print("Using environment map for lighting")

if args.replace_product:
    new_product, existing_product, new_product_name = parse_replace_product_json(args.replace_product)
    if new_product and existing_product:
        print(f"Replacing product {existing_product} with {new_product} (name: {new_product_name})")
        if replace(existing_product, new_product, new_name=new_product_name):
            if "productNameList" in CAMERA_PARAMS and existing_product in CAMERA_PARAMS["productNameList"]:
                CAMERA_PARAMS["productNameList"] = [new_product_name if x == existing_product else x for x in CAMERA_PARAMS["productNameList"]]
                print(f"Updated productNameList to: {CAMERA_PARAMS['productNameList']}")
            else:
                print(f"Warning: {existing_product} not found in productNameList or productNameList not defined")
        else:
            print(f"Failed to replace {existing_product} with {new_product}")

bpy.ops.file.find_missing_files(directory='textures')
if bpy.data.filepath:
    bpy.ops.file.make_paths_relative()

if args.replacement is not None and args.object is not None and len(args.object) > 0:
    replace(args.object[0], args.replacement)

if not args.skip_render:
    bpy.context.scene.render.resolution_percentage = RENDER_PARAMS["resolution_percentage"]
    bpy.context.scene.render.film_transparent = RENDER_PARAMS["film_transparent"]
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = RENDER_PARAMS["samples"]
    bpy.context.scene.cycles.preview_samples = RENDER_PARAMS["preview_samples"]
    
    if USE_GPU and not args.force_cpu:
        setup_gpu_rendering()
    else:
        print("Using CPU for rendering")
        bpy.context.scene.cycles.device = 'CPU'
    
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
                        bpy.context.scene.cycles = 'OPTIX'
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

if args.generate_mask:
    generate_masks(args)

for obj in bpy.data.objects:
    obj.hide_render = False