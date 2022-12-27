
# Object Detection Images Preparation

This repository contains helps indivituals for preparing object detection image data for use in machine learning models. Given a path to a directory containing images and YOLO annotations, the script in this repository can be used to split the data into train, validation, and test sets, and also convert the annotations into VOC and COCO formats, all nice and tidy.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.6+

### Installing

Clone the repository and install the required packages.

```
git clone https://github.com/rafcy/object-detection-data-prep.git
cd object-detection-data-prep
```

## Usage

The main script is called `prepare_data.py`. To run it, provide the path containing the images and annotations, a text file containing the classes and the percentage of the split (eg. 40 means 60% for training and 20/20 for the validation and testing) as arguments.

```
python prepare_data.py --data /path/to/images --labels /path/to/labels.txt --split 40
```

The script will split the annotations and the images into `Annotations` and `Images` folders. The `Annotations` folder contains annotations in (`YOLO`,`VOC`,`COCO`) formats and every folder is split into  three new folders (`Train`, `Valid`, and `Test`) inside the `/path/to/images`.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request if you have any ideas or bug fixes.

## License

This project is licensed under the MIT License.