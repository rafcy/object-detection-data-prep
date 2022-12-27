import os
import random
import pathlib
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image
import json

def create_test_train_txt(directory, percentage = 20):
    # Percentage of images to be used for the test set
    percentage_test = percentage

    # Create and/or truncate train.txt and test.txt
    file_train = open(directory+'/train.txt', 'w')
    file_test = open(directory+'/test.txt', 'w')
    file_valid = open(directory+'/valid.txt', 'w')

    # Populate train.txt and test.txt
    counter = 1
    valid_c = 0
    index_test = round(100 / percentage_test)
    filenames = []
    for path, subdirs, files in os.walk(directory):
        for filename in files:
            infilename = os.path.join(path, filename)
            if not os.path.isfile(infilename): continue
            if infilename.endswith('.jpg') or infilename.endswith('.JPG') or infilename.endswith('.PNG') or infilename.endswith('.png'):  # check if ifle is txt format
                filenames.append(path+'/'+filename)
    random.shuffle(filenames)
    train, test, valid  = [],[],[]
    for filename in filenames:
        if counter == index_test:
            counter = 1
            if valid_c:
                file_test.write(filename + "\n")
                test.append(filename)
                valid_c = 0
            else:
                file_valid.write(filename + "\n")
                valid.append(filename)
                valid_c = 1
        else:
            file_train.write(filename  + "\n")
            train.append(filename)
            counter = counter + 1
        index_test = round(100 / percentage_test)
    return train,test,valid



def move_txt(folders,destination_txt, destination_img):
    # folders => {folder type: images}
    img_dict = dict()
    labels_dict = dict()
    for i, (folder ,paths) in enumerate(folders.items()):
        img_files = []
        lbl_files = []
        for counter,image in enumerate(tqdm(paths, desc=f"Copying files to \'{folder}\' folder")): # reading images paths from the train file line by line
            # print("Converting annotations in file \"{}\" for folder \"{}\".".format({image,folder))
            img_path = image.strip('\n')#.split('\n')[0]
            filepath_txt = pathlib.Path(image).with_suffix('.txt')

            if not os.path.isfile(filepath_txt):
                continue
            try:
                # Copy the txt file to the destination folder
                dst1= "{}/{}/{}_{}".format(destination_txt, folder, counter,
                                           os.path.basename(filepath_txt))
                shutil.copyfile(filepath_txt, dst1)
                lbl_files.append(dst1)
            except PermissionError:
                print('Permission denied.')
            except:
                print(f"Error occured for file {filepath_txt}")

            try:
                # Copy the image file to the destination folder
                dst2 =  "{}/{}/{}_{}".format(destination_img, folder, counter,
                                          os.path.basename(img_path))
                shutil.copyfile(img_path,  dst2)
                img_files.append(dst2)
            except PermissionError:
                print('Permission denied.')
            except:
                print(f"Error occured for file {img_path}")

        img_dict[folder] = img_files
        labels_dict[folder] = lbl_files
    return img_dict,labels_dict

def yolo_to_voc(folders, lbl_dict, destination, classes):
    # folders => {folder type: images}
    voc_paths = {key:[] for key in folders}
    for i, (folder, paths) in enumerate(lbl_dict.items()):
        for counter, text in enumerate(tqdm(paths,
                                           desc=f"Converting files to VOC \'{folder}\' folder")):
            image_file_name = folders[folder][i].strip('\n')  # .split('\n')[0]
            filepath_txt = pathlib.Path(text).with_suffix('.txt')
            if not os.path.isfile(filepath_txt):
                continue

            # create the root element for the XML tree
            annotation = ET.Element("annotation")

            # add the folder and filename elements
            ET.SubElement(annotation, "folder").text = "images"
            ET.SubElement(annotation, "filename").text = os.path.basename(image_file_name)

            img = Image.open(image_file_name)
            img_w, img_h = img.size

            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, "width").text = str(img_w)
            ET.SubElement(size, "height").text = str(img_h)
            ET.SubElement(size, "depth").text = "3"

            # open the YOLO annotation file and read the annotations
            with open(filepath_txt, "r") as f:
                for line in f:
                    # split the line into components
                    data = line.strip().split()
                    class_name = classes[int(data[0])]
                    w = float(data[3]) * img_w
                    h = float(data[4]) * img_h
                    x = float(data[1]) * img_w
                    y = float(data[2]) * img_h
                    # create an object element for each annotation
                    obj = ET.SubElement(annotation, "object")
                    ET.SubElement(obj, "name").text = class_name
                    ET.SubElement(obj, "pose").text = "Unspecified"
                    ET.SubElement(obj, "truncated").text = "0"
                    ET.SubElement(obj, "difficult").text = "0"

                    # create the bounding box element and add the xmin, ymin, xmax, and ymax values
                    bndbox = ET.SubElement(obj, "bndbox")
                    ET.SubElement(bndbox, "xmin").text = str(int(float(x) - float(w) / 2))
                    ET.SubElement(bndbox, "ymin").text = str(int(float(y) - float(h) / 2))
                    ET.SubElement(bndbox, "xmax").text = str(int(float(x) + float(w) / 2))
                    ET.SubElement(bndbox, "ymax").text = str(int(float(y) + float(h) / 2))

            # create an ElementTree object and write the annotation to an XML file
            tree = ET.ElementTree(annotation)
            xml_path = os.path.join(destination, folder, os.path.basename(filepath_txt.with_suffix('.xml')))
            voc_paths[folder].append(xml_path)
            tree.write(xml_path)
    return voc_paths


def voc_to_coco(folder_path, classes, destination):
    # create a mapping from VOC class names to COCO class IDs
    class_mapping = {lbl:i for i,lbl in enumerate(classes)}

    # folders => {folder type: images}
    for i, (folder, paths) in enumerate(folder_path.items()):
        # initialize COCO format data
        coco_data = {
            "info": {
                "description": "Annotations for images in COCO format",
                "url": "",
                "version": "1.0",
                "year": 2022,
                "contributor": "Rafael Makrigiorgis",
                "date_created": ""
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        # loop through all files in the given folder
        for counter, file_name in enumerate(tqdm(paths,
                                           desc=f"Converting files from VOC to COCO \'{folder}\' folder")):
            # parse the XML file
            tree = ET.parse(file_name)
            root = tree.getroot()

            # get the image file name from the XML file
            image_file_name = root.find("filename").text

            # add the image to the COCO data
            coco_data["images"].append({
                "file_name": image_file_name,
                "height": int(root.find("size").find("height").text),
                "width": int(root.find("size").find("width").text),
                "id": len(coco_data["images"])
            })

            # loop through all object elements in the XML file
            for obj in root.findall("object"):
                # map the VOC class name to a COCO class ID, if necessary
                class_name = obj.find("name").text

                # add the annotation to the COCO annotations list
                bndbox = obj.find("bndbox")
                coco_data["annotations"].append({
                  "image_id": len(coco_data["images"]) - 1,
                  "category_id": class_mapping[class_name],
                  "bbox": [
                      int(bndbox.find("xmin").text),
                      int(bndbox.find("ymin").text),
                      int(bndbox.find("xmax").text) - int(bndbox.find("xmin").text),
                      int(bndbox.find("ymax").text) - int(bndbox.find("ymin").text)
                  ],
                  "area": (int(bndbox.find("xmax").text) - int(bndbox.find("xmin").text)) * (
                              int(bndbox.find("ymax").text) - int(bndbox.find("ymin").text)),
                  "iscrowd": 0
                })
        for label, label_id in class_mapping.items():
            category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
            coco_data['categories'].append(category_info)

        # write the COCO data to a JSON file
        with open(os.path.join(destination, folder,f"{folder}.json"), "w") as f:
          json.dump(coco_data, f)


