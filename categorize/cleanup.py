import os
import shutil
import xml.etree.ElementTree as ET
import argparse

def categorize_images(xml_path, img_folder):
    print(f"Parsing XML: {xml_path}...")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error reading XML file: {e}")
        return

    # Map the bare filename to its annotation count
    image_annotation_counts = {}
    
    for img in root.findall('image'):
        # Get the 'name' attribute from XML
        name_attr = img.get('name')
        # os.path.basename strips 'Compile/' or any other folder path
        basename = os.path.basename(name_attr)
        
        annotation_count = len(list(img))
        image_annotation_counts[basename] = annotation_count

    print(f"Loaded {len(image_annotation_counts)} image records.")

    complete_dir = os.path.join(img_folder, 'complete')
    incomplete_dir = os.path.join(img_folder, 'incomplete')
    unannotated_dir = os.path.join(img_folder, 'unannotated')

    for directory in [complete_dir, incomplete_dir, unannotated_dir]:
        os.makedirs(directory, exist_ok=True)

    print(f"\nProcessing files in: {img_folder}...")
    
    stats = {'complete': 0, 'incomplete': 0, 'unannotated': 0, 'skipped': 0}

    for filename in os.listdir(img_folder):
        file_path = os.path.join(img_folder, filename)
        
        # 1. Skip if it's a directory (avoids moving the folders we just made)
        if not os.path.isfile(file_path):
            continue
            
        # 2. Check if this file exists in our XML map
        if filename not in image_annotation_counts:
            # If not in XML, move to unannotated
            dest_dir = unannotated_dir
            stats['unannotated'] += 1
        else:
            count = image_annotation_counts[filename]
            if count >= 3:
                dest_dir = complete_dir
                stats['complete'] += 1
            elif count > 0:
                dest_dir = incomplete_dir
                stats['incomplete'] += 1
            else:
                dest_dir = unannotated_dir
                stats['unannotated'] += 1
                
        # Perform the move
        shutil.move(file_path, os.path.join(dest_dir, filename))

    print("\n--- Categorization Complete ---")
    print(f"Complete (>= 3): {stats['complete']}")
    print(f"Incomplete (1-2): {stats['incomplete']}")
    print(f"Unannotated/Not in XML: {stats['unannotated']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--xml", required=True)
    parser.add_argument("-i", "--img_dir", required=True)
    args = parser.parse_args()
    
    if not os.path.isdir(args.img_dir):
        print(f"Error: The directory '{args.img_dir}' does not exist.")
    else:
        categorize_images(args.xml, args.img_dir)