import os        # For interacting with the file system (directories, paths, etc.)
import shutil    # For copying files

def prepare_classification_dataset(source_dir, target_dir):
    # Create the target directory if it doesn't already exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Loop through all files in the source directory
    for filename in os.listdir(source_dir):
        # Process only the YOLO annotation files (.txt)
        if filename.endswith('.txt'):
            # Full path to the annotation file
            txt_path = os.path.join(source_dir, filename)

            # Corresponding image filename (YOLO uses same name with .jpg extension)
            img_filename = filename.replace('.txt', '.jpg')
            img_path = os.path.join(source_dir, img_filename)

            # If corresponding image file doesn't exist, skip this annotation
            if not os.path.exists(img_path):
                continue  # Move to the next file

            # Open the annotation file and read the first line
            with open(txt_path, 'r') as file:
                first_line = file.readline().strip()
                if not first_line:
                    continue  # Skip empty annotation files

                # The first token in YOLO annotation is the class ID (e.g., '0', '1', etc.)
                class_id = first_line.split()[0]

            # Path for the class-specific subfolder in the target directory
            class_dir = os.path.join(target_dir, class_id)

            # Create the class subfolder if it doesn't already exist
            os.makedirs(class_dir, exist_ok=True)

            # Copy the image into the corresponding class subfolder
            shutil.copy(img_path, os.path.join(class_dir, img_filename))

    # Final message after processing is done
    print("✅ Classification dataset prepared at:", target_dir)


# This part runs only if the script is executed directly (not imported)
if __name__ == "__main__":
    source = "Dataset"  # Replace with your YOLO-format dataset folder
    target = "classification_dataset"  # Folder where classification-format data will be saved
    prepare_classification_dataset(source, target)
