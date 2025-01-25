import xml.etree.ElementTree as ET
import os

def annotation_parser(xml_path):

    # Analyze the xml files as a tree and get their root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    yolo_annotation = []

    # Find the image dimension
    width = int(root.find(".//width").text)
    height = int(root.find(".//height").text) 

    for obj in root.findall(".//object"): # Iterates among all the nodes that have an object in them

        # Find all useful information for each object and elaborate them
        class_name = obj.find(".//name").text
        class_number = {'person':0 ,'person?' : 0, 'people' : 1, 'cyclist': 2}[class_name] # Mapping between class names and class id
        x_bb = int(obj.find(".//x").text)
        y_bb = int(obj.find(".//y").text)
        w_bb = int(obj.find(".//w").text)
        h_bb = int(obj.find(".//h").text)
        x_bb_norm = (x_bb + w_bb / 2) / width
        y_bb_norm = (y_bb + h_bb / 2) / height
        width_bb_norm = w_bb / width
        height_bb_norm = h_bb / height

        # Save all this into the annotation list
        yolo_annotation.append(f"{class_number} {x_bb_norm} {y_bb_norm} {width_bb_norm} {height_bb_norm}")

    return yolo_annotation

def save_new_annotation(annotation_to_save, path_to_save, original_name):

    # Create the directory if it doesn't exists
    os.makedirs(path_to_save, exist_ok=True)

    # Use the original file name to create the new one
    output_file_path = os.path.join(path_to_save, f'{original_name}.txt')

    # Open a file in write mode
    with open(output_file_path, 'w') as F:

        # Write all the lines obtained from the annotation file into the output file
        for annotation in annotation_to_save:
            F.write(annotation + '\n')

def process_all_xml_files(input_folder, output_folder):

    # Get a list of all the files in the input folder
    xml_files = [f for f in os.listdir(input_folder) if f.endswith('.xml')]

    for xml_file in xml_files:

        # Build the path for the xml files
        xml_path = os.path.join(input_folder, xml_file)

        # Get the file name without the extension
        xml_name = os.path.splitext(xml_file)[0]

        # Do the elaboration
        annotation = annotation_parser(xml_path)
        save_new_annotation(annotation, output_folder, xml_name)


input_folder = '/hpc/home/federico.rovighi/3Dperception/Kaistdataset/kaist-cvpr15/annotations-xml-new/set11/V000/'
output_folder = '/hpc/home/federico.rovighi/3Dperception/Kaistdataset/kaist-cvpr15/Labels/set11/V000/'

process_all_xml_files(input_folder, output_folder)
print("conversion done")