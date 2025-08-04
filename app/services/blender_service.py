import os
import json
from threading import Thread
from app.services.s3_service import S3Service


def run_blender_script(script_path: str, input_file: str, output_dir: str, **kwargs):
    command = f'blender -b -P {script_path} -- {input_file} -d {output_dir}'
    for key, value in kwargs.items():
        command += f' --{key} {json.dumps(value)}'
    os.system(command)
    print(f"Blender processing complete for {input_file}")

def process_blender_request(script_path: str, input_file: str, output_dir: str, **kwargs):
    process_thread = Thread(target=run_blender_script, args=(script_path, input_file, output_dir), kwargs=kwargs)
    process_thread.start()