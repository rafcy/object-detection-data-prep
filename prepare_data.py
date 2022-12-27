'''
Author      : Rafael Makrigiorgis
Date        : 27/12/2022
Repository  : https://github.com/rafcy/object-detection-data-prep
Description : Split images with YOLO annotations into Train/Valid/Test sets
              and convert annotations into VOC and COCO formats.
'''

import os
from pathlib import Path
import argparse
from utilities import create_test_train_txt, move_txt, yolo_to_voc, voc_to_coco


def main():
    parser = argparse.ArgumentParser(
        description='This script support converting voc form at xmls to coco format json')
    parser.add_argument('--data', type=str, default=str(Path.cwd()),
                        help='path to dataset files directory. It is not need when use --data')
    parser.add_argument('--labels', type=str, default=os.path.join(str(Path.cwd()),"labels.txt"),
                        help='path to label list.')
    parser.add_argument('--split', type=int, default=40, help='split percentage')
    args = parser.parse_args()

    # check if labels files exist
    assert os.path.exists(args.labels), "Please provide a text file with the labels."

    # split train/test/valid images
    train_fp, test_fp, valid_fp = create_test_train_txt(args.data, args.split)
    folders = {"Train":train_fp, "Test":test_fp, "Valid":valid_fp}

    # check if images are found
    assert len(train_fp)!= 0 or len(test_fp) != 0 or len(valid_fp) != 0, "Please provide a folder path with images."

    # create the initial folders [ Images / Annotations ]
    images_path = os.path.join(args.data,"Images")
    annot_path = os.path.join(args.data,"Annotations")
    yolo_path = os.path.join(annot_path,"Yolo")
    voc_path = os.path.join(annot_path,"VOC")
    coco_path = os.path.join(annot_path,"COCO")
    Path(images_path).mkdir(parents=True, exist_ok=True)
    Path(annot_path).mkdir(parents=True, exist_ok=True)

    for folder in folders:
        Path(os.path.join(images_path,folder)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(yolo_path,folder)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(voc_path,folder)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(coco_path,folder)).mkdir(parents=True, exist_ok=True)

    # get the classes
    with open(args.labels) as f:
        classes = [line.rstrip('\n') for line in f]

    # copy images files to folders
    img_dict, lbl_dict = move_txt(folders, yolo_path, images_path)

    # convert files to VOC format
    voc_paths = yolo_to_voc(img_dict, lbl_dict, voc_path, classes)

    # convert files to COCO format
    voc_to_coco(voc_paths, classes, coco_path)



if __name__ == '__main__':
    main()
