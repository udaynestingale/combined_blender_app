import bpy
import os
import sys
sys.path.append("/home/ubuntu/blender_addons/python_libs") ## <--- Import folder of extra libraries

import tempfile
from PIL import Image, ImageFilter
import numpy as np
import urllib.request
import urllib.parse
import math
import mathutils

# --- Utility Functions ---

def download_image_from_url(url):
    """Download an image from URL to temporary file."""
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(temp_path, 'wb') as out_file:
            out_file.write(response.read())
        return temp_path
    except Exception as e:
        print(f"Error downloading image: {e}")
        raise

def validate_image_file(image_path):
    """Validate if file is a supported image."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Invalid image: {e}")
        return False

def get_image_file(image_path_or_url):
    """Get local file path, downloading if URL."""
    if image_path_or_url.startswith(('http://', 'https://')):
        return download_image_from_url(image_path_or_url), True
    return image_path_or_url, False

def clear_scene():
    """Clear Blender scene."""
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images]:
        for item in block:
            block.remove(item)

def parse_arguments():
    """Parse command line arguments."""
    try:
        separator_index = sys.argv.index('--')
        script_args = sys.argv[separator_index + 1:]
    except ValueError:
        script_args = []
    
    if not script_args:
        print("Usage: blender -b -P script.py -- --rug|--pillow <image> [image2]")
        sys.exit(1)

    if script_args[0] == '--rug':
        return 'rug', script_args[1:]
    elif script_args[0] == '--pillow':
        return 'pillow', script_args[1:]
    else:
        print("Invalid mode. Use --rug or --pillow")
        sys.exit(1)

# --- Rug Functions ---

def remove_white_edges_precise_rug(image_path, threshold=240):
    """Process rug image."""
    try:
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        non_white_mask = np.any(img_array < threshold, axis=2)
        
        if not np.any(non_white_mask):
            print("No non-white content found")
            cropped = image
        else:
            rows = np.any(non_white_mask, axis=1)
            cols = np.any(non_white_mask, axis=0)
            top, bottom = np.where(rows)[0][[0, -1]]
            left, right = np.where(cols)[0][[0, -1]]
            cropped = image.crop((left, top, right + 1, bottom + 1))

        resized = cropped.resize((646, 1024), Image.Resampling.LANCZOS)
        final = Image.new('RGB', (1024, 1024), (0, 0, 0))
        final.paste(resized, (0, 0))
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        final.save(temp_file, format='JPEG')
        return final, temp_file.name, cropped
        
    except Exception as e:
        print(f"Error processing rug image: {e}")
        raise

def replace_textures_in_glb_rug(glb_path, diffuse_path, output_path, base_name):
    """Replace textures in rug GLB and renames parent."""
    try:
        clear_scene()
        result = bpy.ops.import_scene.gltf(filepath=glb_path)
        print("Import result:", result)

        # Check if there are any objects in the scene
        if len(bpy.context.scene.objects) > 0:
            parent_node = bpy.context.scene.objects[0]
            parent_node.name = base_name
            
            # Define the expected child rotation (90 degrees around Z-axis)
            z_rotation_quat = mathutils.Quaternion((0, 0, 1), math.radians(90))  # (0, 0, 0.7071, 0.7071)
            
            # Process immediate children
            if parent_node.children:
                for child in parent_node.children:
                    child.rotation_mode = 'QUATERNION'
                    # Store child's current rotation
                    child_quat = child.rotation_quaternion.copy()
                    # Set child rotation to identity
                    child.rotation_quaternion = (1, 0, 0, 0)
                    
                    # If child's rotation matches expected Z rotation, adjust parent
                    if child_quat.to_euler().z == math.radians(90):  # Compare Z rotation
                        # Apply inverse rotation to parent to maintain orientation
                        parent_node.rotation_mode = 'QUATERNION'
                        parent_node.rotation_quaternion = z_rotation_quat
            else:
                print("Warning: No immediate children found for the parent node")
        else:
            raise ValueError("No objects found in the scene after GLB import")
        
        material = bpy.data.materials[0]
        material.use_nodes = True
        nodes = material.node_tree.nodes
        
        # Find or create texture node
        tex_node = next((n for n in nodes if n.type == 'TEX_IMAGE'), None)
        if not tex_node:
            tex_node = nodes.new('ShaderNodeTexImage')
        
        tex_node.image = bpy.data.images.load(diffuse_path)
        tex_node.image.colorspace_settings.name = 'sRGB'
        
        # Connect to Principled BSDF
        principled = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
        material.node_tree.links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
        
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB'
        )
        
    except Exception as e:
        print(f"Error processing GLB: {e}")
        raise

# --- Pillow Functions ---

def create_normal_map_pillow(unstretched_object, target_width, target_height, target_x, target_y, strength=1.5):
    """
    Create a normal map from an unstretched PIL Image for the pillow, and place it
    into a 1024x1024 texture at the specified target dimensions and position.
    
    Args:
        unstretched_object (PIL.Image): The object image with transparent background (RGBA), before stretching.
        target_width (int): The target width for the normal map on the 1024x1024 texture.
        target_height (int): The target height for the normal map on the 1024x1024 texture.
        target_x (int): The X coordinate for pasting the normal map on the 1024x1024 texture.
        target_y (int): The Y coordinate for pasting the normal map on the 1024x1024 texture.
        strength (float): Normal map strength (0.0 to 5.0)
    
    Returns:
        tuple: (PIL.Image, str) - 1024x1024 normal map, temporary file path
    """
    # Convert unstretched_object to grayscale for height map
    gray = unstretched_object.convert('L')
    height_map = np.array(gray, dtype=np.float32) / 255.0
    
    # Using Sobel filter to calculate gradients for the normal map
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    grad_x = np.zeros_like(height_map)
    grad_y = np.zeros_like(height_map)
    
    # Manual convolution; for larger images, scipy.signal.convolve2d would be faster
    padded = np.pad(height_map, 1, mode='edge')
    for i in range(height_map.shape[0]):
        for j in range(height_map.shape[1]):
            region = padded[i:i+3, j:j+3]
            grad_x[i, j] = np.sum(region * sobel_x)
            grad_y[i, j] = np.sum(region * sobel_y)

    # Normalize gradients and calculate the Z component
    normal_x = -grad_x * strength
    normal_y = -grad_y * strength
    normal_z = np.sqrt(np.maximum(0, 1 - normal_x**2 - normal_y**2))
    
    # Convert to 8-bit RGB format
    normal_map_array = np.stack([
        (normal_x + 1) * 127.5,
        (normal_y + 1) * 127.5,
        normal_z * 255
    ], axis=2).astype(np.uint8)
    
    normal_img = Image.fromarray(normal_map_array, 'RGB')
    
    # Resize normal map to fit the target area (including spill-over)
    resized_normal = normal_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Create the final 1024x1024 normal map with a neutral background
    final_normal_map = Image.new('RGB', (1024, 1024), (128, 128, 255))
    final_normal_map.paste(resized_normal, (target_x, target_y))
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        final_normal_map.save(temp_file, format='JPEG')
        temp_path = temp_file.name
    
    print(f"Created normal map (1024x1024) and placed at pillow 1 location.")
    return final_normal_map, temp_path

def remove_white_background_pillow(image_path, image_path2=None, threshold=240, horizontal_spill_pixels=0, vertical_spill_pixels=0):
    """
    Removes white background using an alpha mask and creates a 1024x1024 diffuse map
    with the image(s) placed in the specified pillow rectangles, spilling over by a given amount.
    
    This function correctly handles irregular shapes by making the background transparent.
    
    Args:
        image_path (str): Path to the first input image.
        image_path2 (str, optional): Path to the second input image.
        threshold (int): White threshold (0-255).
        horizontal_spill_pixels (int): Number of pixels to extend the image horizontally beyond the rect boundaries.
        vertical_spill_pixels (int): Number of pixels to extend the image vertically beyond the rect boundaries.
        
    Returns:
        tuple: (PIL.Image, str, PIL.Image, int, int, int, int) - diffuse map, temp path, first processed (unstretched) image with alpha, pillow1_width (with spill), pillow1_height (with spill), pillow1_x (with spill), pillow1_y (with spill).
    """
    def _process_and_mask_image(img_path):
        """
        Helper function to open an image, find the object, and return it as an
        RGBA image with a transparent background (unstretched).
        """
        image = Image.open(img_path)
        print(f"Processing '{os.path.basename(img_path)}' with original size: {image.size}")
        
        # Ensure image is in a consistent format for processing
        if image.mode == 'RGBA':
            # Flatten against a white background to correctly identify the object mask
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a boolean mask of the object
        img_array = np.array(image)
        non_white_mask = np.any(img_array < threshold, axis=2)

        if not np.any(non_white_mask):
            print(f"Warning: No non-white content found in {os.path.basename(img_path)}. Using original image.")
            return image.convert('RGBA')

        # Find the tightest bounding box around the object
        rows = np.any(non_white_mask, axis=1)
        cols = np.any(non_white_mask, axis=0)
        top, bottom = np.where(rows)[0][[0, -1]]
        left, right = np.where(cols)[0][[0, -1]]

        # Crop the image and mask to the bounding box
        cropped_img_pil = image.crop((left, top, right + 1, bottom + 1))
        cropped_mask_np = non_white_mask[top:bottom + 1, left:right + 1]

        # Create an RGBA image with a transparent background using the mask
        object_with_alpha = cropped_img_pil.convert('RGBA')
        alpha_data = np.array(object_with_alpha)
        alpha_data[:, :, 3] = cropped_mask_np * 255  # True=255 (opaque), False=0 (transparent)
        
        final_object = Image.fromarray(alpha_data)
        print(f"Isolated object from '{os.path.basename(img_path)}', new size: {final_object.size}")
        # Return the isolated object as is (unstretched)
        return final_object

    try:
        # Define base pillow placement rectangles
        base_pillow1_x, base_pillow1_y, base_pillow1_width, base_pillow1_height = 7, 528, 483, 490
        base_pillow2_x, base_pillow2_y, base_pillow2_width, base_pillow2_height = 512, 528, 506, 488

        # Calculate new dimensions and paste coordinates for spill-over
        pillow1_x = base_pillow1_x - horizontal_spill_pixels
        pillow1_y = base_pillow1_y - vertical_spill_pixels
        pillow1_width = base_pillow1_width + (2 * horizontal_spill_pixels)
        pillow1_height = base_pillow1_height + (2 * vertical_spill_pixels)

        pillow2_x = base_pillow2_x - horizontal_spill_pixels
        pillow2_y = base_pillow2_y - vertical_spill_pixels
        pillow2_width = base_pillow2_width + (2 * horizontal_spill_pixels)
        pillow2_height = base_pillow2_height + (2 * vertical_spill_pixels)

        # Get the unstretched, background-removed object for normal map generation
        unstretched_img1 = _process_and_mask_image(image_path)
        # Prepare the image for the diffuse map by stretching it to the target dimensions (with spill-over)
        stretched_diffuse_img1 = unstretched_img1.resize((pillow1_width, pillow1_height), Image.Resampling.LANCZOS)
        
        # Process the second image if provided
        unstretched_img2 = None
        stretched_diffuse_img2 = None
        if image_path2 and os.path.exists(image_path2):
            # Get the unstretched, background-removed object for the second pillow
            unstretched_img2 = _process_and_mask_image(image_path2)
            # Invert the back image vertically as requested
            unstretched_img2 = unstretched_img2.transpose(Image.FLIP_TOP_BOTTOM)
            # Prepare the image for the diffuse map by stretching it to the target dimensions (with spill-over)
            stretched_diffuse_img2 = unstretched_img2.resize((pillow2_width, pillow2_height), Image.Resampling.LANCZOS)
        else: # If only one image for pillow, use it for both sides
            unstretched_img2 = unstretched_img1.transpose(Image.FLIP_TOP_BOTTOM)
            stretched_diffuse_img2 = unstretched_img2.resize((pillow2_width, pillow2_height), Image.Resampling.LANCZOS)


        # Create the 1024x1024 black background canvas
        diffuse_map = Image.new('RGB', (1024, 1024), (0, 0, 0))
        
        # Paste the second image if it exists (already stretched)
        if unstretched_img2:
            diffuse_map.paste(stretched_diffuse_img2, (pillow2_x, pillow2_y), stretched_diffuse_img2)
            print(f"Placed second image at ({pillow2_x}, {pillow2_y}) with size {pillow2_width}x{pillow2_height} (spill-over)")

        # Paste the first image (already stretched)
        diffuse_map.paste(stretched_diffuse_img1, (pillow1_x, pillow1_y), stretched_diffuse_img1) # Use RGBA image as its own mask
        print(f"Placed first image at ({pillow1_x}, {pillow1_y}) with size {pillow1_width}x{pillow1_height} (spill-over)")

        
        # Save the final diffuse map to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            diffuse_map.save(temp_file, format='JPEG')
            temp_path = temp_file.name
            
        print(f"Created pillow diffuse map (1024x1024) at: {temp_path}")
        
        # Return the diffuse map, its path, the first processed unstretched image for normal map generation,
        # and the calculated dimensions/position for its placement
        return diffuse_map, temp_path, unstretched_img1, pillow1_width, pillow1_height, pillow1_x, pillow1_y
    
    except Exception as e:
        print(f"Error processing image(s) for diffuse map: {e}")
        raise

def replace_textures_in_glb_pillow(glb_path, diffuse_map_path, normal_map_path, output_glb_path, base_name):
    """
    Replace the diffuse and normal map textures of the only material in a GLB file for the pillow.
    
    Args:
        glb_path (str): Path to the input GLB file
        diffuse_map_path (str): Path to the temporary Diffuse map
        normal_map_path (str): Path to the normal map (can be None if not used)
        output_glb_path (str): Path to save the modified GLB file
    """
    try:
        clear_scene()
        bpy.ops.import_scene.gltf(filepath=glb_path)
        print(f"Imported GLB: {glb_path}")

        # Check if there are any objects in the scene
        if len(bpy.context.scene.objects) > 0:
            mesh_node = bpy.context.scene.objects[0]
            
            # Set mesh position to (0, 0, 0)
            mesh_node.location = (0, 0, 0)
            
            # Check if the mesh has a parent
            if mesh_node.parent:
                parent_node = mesh_node.parent
                parent_node.name = base_name
            else:
                # Create a new parent (empty object) if none exists
                parent_node = bpy.data.objects.new(base_name, None)  # Empty object
                bpy.context.scene.collection.objects.link(parent_node)
                mesh_node.parent = parent_node
        else:
            raise ValueError("No objects found in the scene after GLB import")


        if not bpy.data.materials:
            raise ValueError("No materials found in the GLB file.")
        
        # Assume the first material is the one to modify
        material = bpy.data.materials[0]
        print(f"Found material: {material.name}")

        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        # Find the Principled BSDF shader node
        principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not principled:
            raise ValueError("Principled BSDF node not found in the material.")

        # Remove existing image texture and normal map nodes to ensure a clean slate
        for node in nodes:
            if node.type in ('TEX_IMAGE', 'NORMAL_MAP'):
                nodes.remove(node)

        # Create and link Diffuse (Base Color) map
        if not os.path.exists(diffuse_map_path):
            raise FileNotFoundError(f"Diffuse map not found: {diffuse_map_path}")
            
        diffuse_node = nodes.new(type='ShaderNodeTexImage')
        diffuse_node.location = (-350, 300)
        diffuse_node.image = bpy.data.images.load(filepath=diffuse_map_path)
        diffuse_node.image.colorspace_settings.name = 'sRGB'
        links.new(diffuse_node.outputs['Color'], principled.inputs['Base Color'])
        print(f"Applied diffuse map: {os.path.basename(diffuse_map_path)}")

        # Create and link Normal map (optional)
        if normal_map_path and os.path.exists(normal_map_path):
            normal_node = nodes.new(type='ShaderNodeTexImage')
            normal_node.location = (-350, 0)
            normal_node.image = bpy.data.images.load(filepath=normal_map_path)
            normal_node.image.colorspace_settings.name = 'Non-Color'
            
            normal_map_node = nodes.new(type='ShaderNodeNormalMap')
            normal_map_node.location = (-150, 0)
            links.new(normal_node.outputs['Color'], normal_map_node.inputs['Color'])
            links.new(normal_map_node.outputs['Normal'], principled.inputs['Normal'])
            print(f"Applied normal map: {os.path.basename(normal_map_path)}")
        else:
            print("Skipping normal map application (normal map creation was disabled or file not found).")

        # Export the final GLB
        bpy.ops.export_scene.gltf(
            filepath=output_glb_path,
            export_format='GLB',
            export_image_format='AUTO', # Let Blender decide best format
            export_materials='EXPORT'
        )
        print(f"Successfully exported modified GLB to: {output_glb_path}")

    except Exception as e:
        print(f"Error during GLB processing: {e}")
        raise

# --- Main Function ---

def main():
    mode, image_names = parse_arguments()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Initialize variables
    diffuse_temp = None
    downloaded_files = []
    output_path = os.path.join(script_dir, "output.glb")  # Default fallback
    
    try:
        if mode == 'rug':
            if not image_names:
                print("Error: No image provided for rug")
                sys.exit(1)
                
            # Get the base name from URL or local path
            input_name = image_names[0]
            if input_name.startswith(('http://', 'https://')):
                # Extract from URL like https://rizzyhome.in/images/msr/EPS213.jpg
                base_name = os.path.splitext(os.path.basename(urllib.parse.urlparse(input_name).path))[0]
            else:
                # Extract from local path
                base_name = os.path.splitext(os.path.basename(input_name))[0]
            
            # Create output path with the base name
            output_path = os.path.join(script_dir, "output.glb")  #f"{base_name}_rug.glb"
            
            image_path, is_temp = get_image_file(input_name)
            if is_temp:
                downloaded_files.append(image_path)
                
            if not validate_image_file(image_path):
                print("Invalid image file")
                sys.exit(1)
                
            glb_path = os.path.join(script_dir, "Rug_Rectangular_Base_10x13.glb")
            if not os.path.exists(glb_path):
                print("Rug Base GLB - 'Rug_Rectangular_Base_10x13.glb' file not found")
                sys.exit(1)
                
            diffuse_img, diffuse_temp, _ = remove_white_edges_precise_rug(image_path)
            replace_textures_in_glb_rug(glb_path, diffuse_temp, output_path, base_name)
            
        elif mode == 'pillow':
            normal_map_creation_enabled = False # You can change this to False if you want to disable normal map for pillow
            
            if not image_names:
                print("Error: No image filenames provided for pillow mode.")
                sys.exit(1)
            
            # Process first image (download if URL)
            image_path1, is_temp1 = get_image_file(image_names[0])
            if is_temp1:
                downloaded_files.append(image_path1)
            
            # Get base name for output file
            if image_names[0].startswith(('http://', 'https://')):
                base_name1 = os.path.splitext(os.path.basename(urllib.parse.urlparse(image_names[0]).path))[0]
            else:
                base_name1 = os.path.splitext(os.path.basename(image_names[0]))[0]
            
            # Process second image if provided
            image_path2 = None
            if len(image_names) > 1:
                image_path2, is_temp2 = get_image_file(image_names[1])
                if is_temp2:
                    downloaded_files.append(image_path2)
            else:
                # Use first image for both sides
                image_path2 = image_path1
            
            # Set output path
            output_glb_path = os.path.join(script_dir, "output.glb")  #f"{base_name1}_pillow.glb"
            
            # Validate files
            if not validate_image_file(image_path1):
                print(f"Error: First image is invalid")
                sys.exit(1)
            if image_path2 != image_path1 and not validate_image_file(image_path2):
                print(f"Error: Second image is invalid")
                sys.exit(1)
            
            glb_path = os.path.join(script_dir, "Pillow_Square_Base_40x40.glb")
            if not os.path.exists(glb_path):
                print(f"Error: Pillow Base GLB file 'Pillow_Square_Base_40x40.glb' not found")
                sys.exit(1)

            print("\nStarting pillow creation process...")
            

            # 1. Create Diffuse Map
            print("\n[STEP 1] Creating Diffuse Map Texture...")
            print("-" * 55)
            diffuse_image, diffuse_temp_path, unstretched_img1_for_normal, pillow1_target_width, pillow1_target_height, pillow1_target_x, pillow1_target_y = remove_white_background_pillow(image_path1, image_path2, horizontal_spill_pixels=25, vertical_spill_pixels=15)
            
            # 2. Create Normal Map (Optional)
            if normal_map_creation_enabled:
                print("\n[STEP 2] Creating Normal Map Texture...")
                print("-" * 40)
                normal_image, normal_temp_path = create_normal_map_pillow(unstretched_img1_for_normal, pillow1_target_width, pillow1_target_height, pillow1_target_x, pillow1_target_y)
            else:
                print("\n[STEP 2] Skipping Normal Map creation as it's disabled.")
                normal_temp_path = None

            # 3. Replace textures in GLB
            print("\n[STEP 3] Applying Textures to Pillow GLB...")
            print("-" * 45)
            replace_textures_in_glb_pillow(glb_path, diffuse_temp_path, normal_temp_path, output_glb_path, base_name1)
            
    except Exception as e:
        print(f"Error in main processing: {e}")
        sys.exit(1)
        
    finally:
        # Cleanup
        if diffuse_temp and os.path.exists(diffuse_temp):
            os.remove(diffuse_temp)
        for temp_file in downloaded_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    

if __name__ == "__main__":
    main()