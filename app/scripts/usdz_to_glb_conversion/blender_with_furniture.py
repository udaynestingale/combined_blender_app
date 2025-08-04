import bpy
import sys
import os

def convert_usdz_to_glb(input_file, output_file):
    # Clear existing scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import USDZ file
    bpy.ops.wm.usd_import(filepath=input_file)

    # Select all objects
    bpy.ops.object.select_all(action='SELECT')

    # Rotate 90 degrees around X axis to match coordinate systems
    bpy.ops.transform.rotate(value=1.5708, orient_axis='X')  # 90 degrees in radians

    # Export as GLB
    bpy.ops.export_scene.gltf(filepath=output_file, export_yup=False, export_format='GLB')

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: blender --background --python usdz_to_glb.py -- <input_file> <output_file>")
        sys.exit(1)

    # Parse the command line arguments
    input_file = sys.argv[-2]
    output_file = sys.argv[-1]

    # Perform the conversion
    convert_usdz_to_glb(input_file, output_file)