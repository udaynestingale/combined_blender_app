import bpy
import os
import sys
#sys.path.append("/home/ubuntu/blender_addons/python_libs") ## <--- Import folder of extra libraries

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
        print("Usage: blender -b -P script.py -- --rug|--pillow <image>")
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

        resizeFactor = 2
        resized = cropped.resize((646 * resizeFactor, 1024 * resizeFactor), Image.Resampling.LANCZOS)
        final = Image.new('RGB', (1024 * resizeFactor, 1024 * resizeFactor), (0, 0, 0))
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
        print(f"Successfully exported modified GLB to: {output_path}")
        
    except Exception as e:
        print(f"Error processing GLB: {e}")
        raise

# --- Pillow Functions ---

def remove_white_background_pillow(image_path, threshold=240):
    """
    Removes white background and creates a 1024x1024 diffuse map with the image.
    
    Args:
        image_path (str): Path to the input image.
        threshold (int): White threshold (0-255).
        
    Returns:
        tuple: (PIL.Image, str) - diffuse map, temp path
    """
    try:
        # Process image
        image = Image.open(image_path)
        print(f"Processing '{os.path.basename(image_path)}' with original size: {image.size}")
        
        # Ensure image is in a consistent format
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create mask
        img_array = np.array(image)
        non_white_mask = np.any(img_array < threshold, axis=2)

        if not np.any(non_white_mask):
            print(f"Warning: No non-white content found in {os.path.basename(image_path)}. Using original image.")
            cropped = image
        else:
            rows = np.any(non_white_mask, axis=1)
            cols = np.any(non_white_mask, axis=0)
            top, bottom = np.where(rows)[0][[0, -1]]
            left, right = np.where(cols)[0][[0, -1]]
            cropped = image.crop((left, top, right + 1, bottom + 1))

        # Resize to 1024x1024
        diffuse_map = cropped.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        # Horizontally flip the image
        diffuse_map = diffuse_map.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            diffuse_map.save(temp_file, format='JPEG')
            temp_path = temp_file.name
            
        print(f"Created pillow diffuse map (1024x1024, horizontally flipped) at: {temp_path}")
        return diffuse_map, temp_path
    
    except Exception as e:
        print(f"Error processing image for diffuse map: {e}")
        raise

def replace_textures_in_glb_pillow(glb_path, front_diffuse_path, back_diffuse_path, output_glb_path, base_name):
    """
    Replace the diffuse map textures for both front and back pillow meshes in a GLB file.
    
    Args:
        glb_path (str): Path to the input GLB file
        front_diffuse_path (str): Path to the front diffuse map
        back_diffuse_path (str): Path to the back diffuse map
        output_glb_path (str): Path to save the modified GLB file
        base_name (str): Base name for the output file
    """
    try:
        clear_scene()
        bpy.ops.import_scene.gltf(filepath=glb_path)
        print(f"Imported GLB: {glb_path}")

        # Check if there are any objects in the scene
        if len(bpy.context.scene.objects) == 0:
            raise ValueError("No objects found in the scene after GLB import")

        # Find the meshes - look for objects with "Front" and "Back" in their names
        front_mesh = None
        back_mesh = None
        parent_node = None
        
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                if 'Front' in obj.name or 'front' in obj.name.lower():
                    front_mesh = obj
                elif 'Back' in obj.name or 'back' in obj.name.lower():
                    back_mesh = obj
                    
                # Set mesh position to (0, 0, 0)
                obj.location = (0, 0, 0)
                
                # Handle parent relationship
                if obj.parent and not parent_node:
                    parent_node = obj.parent
                elif not obj.parent and not parent_node:
                    # Create a new parent (empty object) if none exists
                    parent_node = bpy.data.objects.new(base_name, None)
                    bpy.context.scene.collection.objects.link(parent_node)
                    obj.parent = parent_node

        # If we couldn't find meshes by name, try to identify them by order or material
        if not front_mesh or not back_mesh:
            mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
            if len(mesh_objects) >= 2:
                front_mesh = mesh_objects[0]  # First mesh as front
                back_mesh = mesh_objects[1]   # Second mesh as back
                print("Warning: Could not identify Front/Back meshes by name. Using first two mesh objects.")
            else:
                raise ValueError("Could not find two mesh objects for front and back of pillow")

        if parent_node:
            parent_node.name = base_name

        print(f"Front mesh: {front_mesh.name}")
        print(f"Back mesh: {back_mesh.name}")

        # Get materials for each mesh
        front_material = None
        back_material = None
        
        if front_mesh.data.materials:
            front_material = front_mesh.data.materials[0]
        if back_mesh.data.materials:
            back_material = back_mesh.data.materials[0]

        if not front_material or not back_material:
            raise ValueError("Could not find materials for both front and back meshes")

        # Process front material
        print(f"Processing front material: {front_material.name}")
        front_material.use_nodes = True
        front_nodes = front_material.node_tree.nodes
        front_links = front_material.node_tree.links

        # Find the Principled BSDF shader node for front
        front_principled = next((n for n in front_nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not front_principled:
            raise ValueError("Principled BSDF node not found in the front material.")

        # Remove existing image texture nodes from front material
        for node in front_nodes:
            if node.type == 'TEX_IMAGE':
                front_nodes.remove(node)

        # Create and link front diffuse map
        if not os.path.exists(front_diffuse_path):
            raise FileNotFoundError(f"Front diffuse map not found: {front_diffuse_path}")
            
        front_diffuse_node = front_nodes.new(type='ShaderNodeTexImage')
        front_diffuse_node.image = bpy.data.images.load(filepath=front_diffuse_path)
        front_diffuse_node.image.colorspace_settings.name = 'sRGB'
        front_links.new(front_diffuse_node.outputs['Color'], front_principled.inputs['Base Color'])
        print(f"Applied front diffuse map (horizontally flipped): {os.path.basename(front_diffuse_path)}")

        # Process back material
        print(f"Processing back material: {back_material.name}")
        back_material.use_nodes = True
        back_nodes = back_material.node_tree.nodes
        back_links = back_material.node_tree.links

        # Find the Principled BSDF shader node for back
        back_principled = next((n for n in back_nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not back_principled:
            raise ValueError("Principled BSDF node not found in the back material.")

        # Remove existing image texture nodes from back material
        for node in back_nodes:
            if node.type == 'TEX_IMAGE':
                back_nodes.remove(node)

        # Create and link back diffuse map
        if not os.path.exists(back_diffuse_path):
            raise FileNotFoundError(f"Back diffuse map not found: {back_diffuse_path}")
            
        back_diffuse_node = back_nodes.new(type='ShaderNodeTexImage')
        back_diffuse_node.image = bpy.data.images.load(filepath=back_diffuse_path)
        back_diffuse_node.image.colorspace_settings.name = 'sRGB'
        back_links.new(back_diffuse_node.outputs['Color'], back_principled.inputs['Base Color'])
        print(f"Applied back diffuse map (horizontally flipped): {os.path.basename(back_diffuse_path)}")

        # Export the final GLB
        bpy.ops.export_scene.gltf(
            filepath=output_glb_path,
            export_format='GLB',
            export_image_format='AUTO',
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
    output_path = os.path.join(script_dir, "output.glb")
    
    try:
        if mode == 'rug':
            if not image_names:
                print("Error: No image provided for rug")
                sys.exit(1)
                
            # Get the base name from URL or local path
            input_name = image_names[0]
            if input_name.startswith(('http://', 'https://')):
                base_name = os.path.splitext(os.path.basename(urllib.parse.urlparse(input_name).path))[0]
            else:
                base_name = os.path.splitext(os.path.basename(input_name))[0]
            
            # Create output path with the base name
            output_path = os.path.join(script_dir, "output.glb") #os.path.join(script_dir, f"{base_name}_rug.glb")
            
            image_path, is_temp = get_image_file(input_name)
            if is_temp:
                downloaded_files.append(image_path)
                
            if not validate_image_file(image_path):
                print("Invalid image file")
                sys.exit(1)
                
            glb_path = os.path.join(script_dir, "Rug_New_Basefile.glb")
            if not os.path.exists(glb_path):
                print("Rug Base GLB - 'Rug_New_Basefile.glb' file not found")
                sys.exit(1)
                
            diffuse_img, diffuse_temp, _ = remove_white_edges_precise_rug(image_path)
            replace_textures_in_glb_rug(glb_path, diffuse_temp, output_path, base_name)
            
        elif mode == 'pillow':
            if len(image_names) < 2:
                print("Error: Pillow mode requires two images (front and back).")
                sys.exit(1)
            
            front_image_name = image_names[0]
            back_image_name = image_names[1]
            
            # Process front image (download if URL)
            front_image_path, front_is_temp = get_image_file(front_image_name)
            if front_is_temp:
                downloaded_files.append(front_image_path)
            
            # Process back image (download if URL)
            back_image_path, back_is_temp = get_image_file(back_image_name)
            if back_is_temp:
                downloaded_files.append(back_image_path)
            
            # Get base name for output file (from front image)
            if front_image_name.startswith(('http://', 'https://')):
                base_name = os.path.splitext(os.path.basename(urllib.parse.urlparse(front_image_name).path))[0]
            else:
                base_name = os.path.splitext(os.path.basename(front_image_name))[0]
            
            # Set output path
            output_glb_path = os.path.join(script_dir, "output.glb") #os.path.join(script_dir, f"{base_name}_pillow.glb")
            
            # Validate files
            if not validate_image_file(front_image_path):
                print(f"Error: Front image is invalid")
                sys.exit(1)
            
            if not validate_image_file(back_image_path):
                print(f"Error: Back image is invalid")
                sys.exit(1)
            
            glb_path = os.path.join(script_dir, "Pillow_75.glb")
            if not os.path.exists(glb_path):
                print(f"Error: Pillow Base GLB file 'Pillow_75.glb' not found")
                sys.exit(1)

            print("\nStarting pillow creation process...")
            
            # Create Front Diffuse Map
            print("\n[STEP 1] Creating Front Diffuse Map Texture...")
            print("-" * 55)
            front_diffuse_image, front_diffuse_temp = remove_white_background_pillow(front_image_path)
            
            # Create Back Diffuse Map
            print("\n[STEP 2] Creating Back Diffuse Map Texture...")
            print("-" * 55)
            back_diffuse_image, back_diffuse_temp = remove_white_background_pillow(back_image_path)
            
            # Replace textures in GLB
            print("\n[STEP 3] Applying Textures to Pillow GLB...")
            print("-" * 45)
            replace_textures_in_glb_pillow(glb_path, front_diffuse_temp, back_diffuse_temp, output_glb_path, base_name)
            
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