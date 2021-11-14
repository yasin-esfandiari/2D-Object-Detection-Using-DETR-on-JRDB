"""
@ author: yasin as m.sain

STRUCTURE OF IDs:
image_id: annotations_map.image_name                            2 char + 6 char --> 8 digits
annotation_id: annotations_map.image_name.ann_id_counter        2 char + 6 char + 2 char    --> 10 digits
"""

# should be rewrite as a class
import re
import os
import json
import itertools
import numpy as np
from glob import glob
import scipy.io as sio
from pycocotools import mask as cocomask
from PIL import Image


# def JRDB2COCO(rootaddress=""):

# pass a list of train, list of val

categories = [
    {
        "supercategory": "none",
        "name": "pedestrian",
        "id": 0
    }
]

# modes = ['train', 'val']

# path = "clark-center-2019-02-28_1.json"
# path = "simplified.json"
# clark-center-2019-02-28_1_image0.json
# clark-center-intersection-2019-02-28_0_image0_test.json
# path = "../jrdb_train/cvgl/group/jrdb/data/train_dataset/labels/labels_2d/clark-center-2019-02-28_1_image0.json"
# json_file = "clark-center-2019-02-28_1_image0_{}.json".format('train')
# save_dir = "clark-center-2019-02-28_1_image0_{}.json".format('train')
save_dir_test = "../jrdb_test/cvgl/group/jrdb/data/test_dataset/test(wo_FO).json"

annotations_path = "../jrdb_test/cvgl/group/jrdb/data/test_dataset/images/image_stitched/"
# annotations_path = "labels_2d_stitched/"
# images_path = "../jrdb_train/cvgl/group/jrdb/data/train_dataset/images/image_stitched/"   # address as avale root? No
max_anns_per_image = 0
max_anns_class_and_img = ""
# image_limit = 500

test_annotations = [
    'cubberly-auditorium-2019-04-22_1',
    'discovery-walk-2019-02-28_0',
    'discovery-walk-2019-02-28_1',
    'food-trucks-2019-02-12_0',
    'gates-ai-lab-2019-04-17_0',
    'gates-basement-elevators-2019-01-17_0',
    'gates-foyer-2019-01-17_0',
    'gates-to-clark-2019-02-28_0',
    'hewlett-class-2019-01-23_0',
    'hewlett-class-2019-01-23_1',
    'huang-2-2019-01-25_1',
    'huang-intersection-2019-01-22_0',
    'indoor-coupa-cafe-2019-02-06_0',
    'lomita-serra-intersection-2019-01-30_0',
    'meyer-green-2019-03-16_1',
    'nvidia-aud-2019-01-25_0',
    'nvidia-aud-2019-04-18_1',
    'nvidia-aud-2019-04-18_2',
    'outdoor-coupa-cafe-2019-02-06_0',
    'quarry-road-2019-02-28_0',
    'serra-street-2019-01-30_0',
    'stlc-111-2019-04-19_1',
    'stlc-111-2019-04-19_2',
    'tressider-2019-03-16_2',
    'tressider-2019-04-26_0',
    'tressider-2019-04-26_1',
    'tressider-2019-04-26_3'
]

annotations_map = {
    'cubberly-auditorium-2019-04-22_1': 11,
    'discovery-walk-2019-02-28_0': 12,
    'discovery-walk-2019-02-28_1': 13,
    'food-trucks-2019-02-12_0': 14,
    'gates-ai-lab-2019-04-17_0': 15,
    'gates-basement-elevators-2019-01-17_0': 16,
    'gates-foyer-2019-01-17_0': 17,
    'gates-to-clark-2019-02-28_0': 18,
    'hewlett-class-2019-01-23_0': 19,
    'hewlett-class-2019-01-23_1': 20,
    'huang-2-2019-01-25_1': 21,
    'huang-intersection-2019-01-22_0': 22,
    'indoor-coupa-cafe-2019-02-06_0': 23,
    'lomita-serra-intersection-2019-01-30_0': 24,
    'meyer-green-2019-03-16_1': 25,
    'nvidia-aud-2019-01-25_0': 26,
    'nvidia-aud-2019-04-18_1': 27,
    'nvidia-aud-2019-04-18_2': 28,
    'outdoor-coupa-cafe-2019-02-06_0': 29,
    'quarry-road-2019-02-28_0': 30,
    'serra-street-2019-01-30_0': 31,
    'stlc-111-2019-04-19_1': 32,
    'stlc-111-2019-04-19_2': 33,
    'tressider-2019-03-16_2': 34,
    'tressider-2019-04-26_0': 35,
    'tressider-2019-04-26_1': 36,
    'tressider-2019-04-26_3': 37,
}
# for output of COCO format
test_file = {
    "categories": categories,
    "images": [],
    "annotations": []
}

folders = test_annotations

for folder_name in folders:
    folder = os.path.join(annotations_path, folder_name)
    # files = [os.path.join(annotations_path, f) for f in os.listdir(folder)]
    files = os.listdir(folder)
    print(folder, '/', files)
    for file in files:
        # print(file, ": ", annotations_map[file])

        ann_id_counter = 1

        image_id = file.split('.')[0]  # 000479.jpg    --> 000479
        image_id_int = int(str(annotations_map[folder_name]) + image_id)  # 11000479

        sub_folder = file.split('.')[0]
        img_elem = {"file_name": folder_name + '/' + file,
                    "height": 480,
                    "width": 3760,
                    "id": image_id_int
                    }

        test_file["images"].append(img_elem)

        annot_elem = {
            "id": int(str(annotations_map[folder_name]) + image_id + str(ann_id_counter)),  # 110004790
            "bbox": [
                float(0),
                float(0),
                float(50),
                float(60)
            ],
            "segmentation": list(),  # list([poly])
            "image_id": image_id_int,
            "ignore": 0,
            "category_id": 0,
            "iscrowd": 0,
            "area": float(100)
            # "attributes": {
            #     "no_eval": attributes["no_eval"],
            #     "truncated": attributes["truncated"],
            #     "interpolated": attributes["interpolated"],
            #     "occlusion": attributes["occlusion"]
            # },
            # "prevJRDB": {
            #     "file_id": ann["file_id"],
            #     "label_id": ann["label_id"]
            # }
        }

        ann_id_counter += 1
        test_file["annotations"].append(annot_elem)

        # if idx == image_limit:
        #     break

with open(save_dir_test, 'w') as json_file:
    json.dump(test_file, json_file, indent=4, sort_keys=True)

# print("max_anns_per_image: ", max_anns_per_image)
# print("max_anns_class_and_img: ", max_anns_class_and_img)