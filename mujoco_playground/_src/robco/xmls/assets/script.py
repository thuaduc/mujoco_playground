import os
from stl import mesh

# Path to the directory containing the STL files
folder_path = '/home/sfrl_lab/ros2_ws/urdf'

def validate_and_process_stl(file_path):
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return False

    try:
        # Load the ASCII STL file
        stl_mesh = mesh.Mesh.from_file(file_path)

        # Save it back in binary mode (overwrite the original file)
        stl_mesh.save(file_path)  # By default, this saves in binary format

        print(f"Successfully validated and converted {file_path} to binary.")
        return True
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return False

def process_all_stls_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.stl'):
            file_path = os.path.join(directory_path, filename)
            if validate_and_process_stl(file_path):
                print(f"Successfully processed {file_path}")
            else:
                print(f"Failed to process {file_path}")

# Start processing STL files in the folder
process_all_stls_in_directory(folder_path)
