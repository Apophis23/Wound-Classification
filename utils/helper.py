import os
import re
import shutil

def copy_files_with_incremented_number(src_dir, dest_dir, start_num):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    for file in files:
        src_file_path = os.path.join(src_dir, file)
        filename, ext = os.path.splitext(file)

        match = re.search(r'\((\d+)\)$', filename)
        if match:
            base_name = filename[:match.start()].strip()
        new_filename = f"{base_name} ({start_num}){ext}"
        new_file_path = os.path.join(dest_dir, new_filename)

        if not os.path.exists(new_file_path):
            shutil.copy(src_file_path, new_file_path)
            print(f"Copied: {src_file_path} -> {new_file_path}")

        start_num += 1

def count_files_in_directory(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return 0

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)

def decode_class(label, mapping_dict):
    return mapping_dict[label]