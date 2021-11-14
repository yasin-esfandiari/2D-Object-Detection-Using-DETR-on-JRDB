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

modes = ['train', 'val']

# path = "clark-center-2019-02-28_1.json"
# path = "simplified.json"
# clark-center-2019-02-28_1_image0.json
# clark-center-intersection-2019-02-28_0_image0_test.json
# path = "../jrdb_train/cvgl/group/jrdb/data/train_dataset/labels/labels_2d/clark-center-2019-02-28_1_image0.json"
# json_file = "clark-center-2019-02-28_1_image0_{}.json".format('train')
# save_dir = "clark-center-2019-02-28_1_image0_{}.json".format('train')
save_dir_train = "../jrdb_train/cvgl/group/jrdb/data/train_dataset/train(wo_FO).json"
save_dir_val = "../jrdb_train/cvgl/group/jrdb/data/train_dataset/val(wo_FO).json"
save_dir_subtrain = "../jrdb_train/cvgl/group/jrdb/data/train_dataset/subtrain(wo_FO).json"  # 7 train, 2 val, 500 image
save_dir_subval = "../jrdb_train/cvgl/group/jrdb/data/train_dataset/subval(wo_FO).json"

annotations_path = "../jrdb_train/cvgl/group/jrdb/data/train_dataset/labels/labels_2d_stitched/"
# annotations_path = "labels_2d_stitched/"
# images_path = "../jrdb_train/cvgl/group/jrdb/data/train_dataset/images/image_stitched/"   # address as avale root? No
max_anns_per_image = 0
max_anns_class_and_img = ""
# image_limit = 500

train_annotations = [
    'bytes-cafe-2019-02-07_0.json',
    'clark-center-2019-02-28_0.json',
    'clark-center-intersection-2019-02-28_0.json',
    'cubberly-auditorium-2019-04-22_0.json',
    'forbes-cafe-2019-01-22_0.json',
    'gates-159-group-meeting-2019-04-03_0.json',
    'gates-basement-elevators-2019-01-17_1.json',
    'gates-to-clark-2019-02-28_1.json',
    'hewlett-packard-intersection-2019-01-24_0.json',
    'huang-basement-2019-01-25_0.json',
    'huang-lane-2019-02-12_0.json',
    'jordan-hall-2019-04-22_0.json',
    'memorial-court-2019-03-16_0.json',
    'packard-poster-session-2019-03-20_0.json',
    'packard-poster-session-2019-03-20_1.json',
    'packard-poster-session-2019-03-20_2.json',
    'stlc-111-2019-04-19_0.json',
    'svl-meeting-gates-2-2019-04-08_0.json',
    'svl-meeting-gates-2-2019-04-08_1.json',
    'tressider-2019-03-16_0.json'
]

val_annotations = [
    'clark-center-2019-02-28_1.json',
    'gates-ai-lab-2019-02-08_0.json',
    'huang-2-2019-01-25_0.json',
    'meyer-green-2019-03-16_0.json',
    'nvidia-aud-2019-04-18_0.json',
    'tressider-2019-03-16_1.json',
    'tressider-2019-04-26_2.json'
]

annotations_map = {
    'bytes-cafe-2019-02-07_0.json': 11,
    'clark-center-2019-02-28_0.json': 12,
    'clark-center-2019-02-28_1.json': 13,
    'clark-center-intersection-2019-02-28_0.json': 14,
    'cubberly-auditorium-2019-04-22_0.json': 15,
    'forbes-cafe-2019-01-22_0.json': 16,
    'gates-159-group-meeting-2019-04-03_0.json': 17,
    'gates-ai-lab-2019-02-08_0.json': 18,
    'gates-basement-elevators-2019-01-17_1.json': 19,
    'gates-to-clark-2019-02-28_1.json': 20,
    'hewlett-packard-intersection-2019-01-24_0.json': 21,
    'huang-2-2019-01-25_0.json': 22,
    'huang-basement-2019-01-25_0.json': 23,
    'huang-lane-2019-02-12_0.json': 24,
    'jordan-hall-2019-04-22_0.json': 25,
    'memorial-court-2019-03-16_0.json': 26,
    'meyer-green-2019-03-16_0.json': 27,
    'nvidia-aud-2019-04-18_0.json': 28,
    'packard-poster-session-2019-03-20_0.json': 29,
    'packard-poster-session-2019-03-20_1.json': 30,
    'packard-poster-session-2019-03-20_2.json': 31,
    'stlc-111-2019-04-19_0.json': 32,
    'svl-meeting-gates-2-2019-04-08_0.json': 33,
    'svl-meeting-gates-2-2019-04-08_1.json': 34,
    'tressider-2019-03-16_0.json': 35,
    'tressider-2019-03-16_1.json': 36,
    'tressider-2019-04-26_2.json': 37,
}
# for output of COCO format
train_file = {
    "categories": categories,
    "images": [],
    "annotations": []
}
val_file = {
    "categories": categories,
    "images": [],
    "annotations": []
}

for mode in modes:
    if mode == 'train':
        files = train_annotations
    else:
        files = val_annotations

    for file in files:
        # print(file, ": ", annotations_map[file])
        with open(os.path.join(annotations_path, file)) as f:
            file_dict = json.load(f)
            labels = file_dict['labels']

            for idx, image_name in enumerate(labels):
                ann_id_counter = 1
                anns = labels[image_name]

                # checking the maximum number of anns for image --> can be used for num_query in DETR
                # if len(anns) > max_anns_per_image:
                #     max_anns_per_image = len(anns)
                #     max_anns_class_and_img = annotations_path + "/" + file + "/" + image_name

                image_id = image_name.split('.')[0]     # 000479.jpg    --> 000479
                image_id_int = int(str(annotations_map[file]) + image_id)    # 11000479

                sub_folder = file.split('.')[0]
                img_elem = {"file_name": sub_folder + '/' + image_name,
                            "height": 480,
                            "width": 3760,
                            "id": image_id_int
                            }

                if mode == 'train':
                    train_file["images"].append(img_elem)
                else:
                    val_file["images"].append(img_elem)

                for ann in anns:
                    attributes = ann["attributes"]
                    if attributes["occlusion"] == "Fully_occluded":
                        # print(ann)
                        continue

                    # for segmentation purpose:
                    # poly = [[ann["box"][0], ann["box"][1]],                                     # xmin, ymin
                    #         [ann["box"][0] + ann["box"][2], ann["box"][1]],                     # xmax, ymin
                    #         [ann["box"][0] + ann["box"][2], ann["box"][1] + ann["box"][3]],     # xmax, ymax
                    #         [ann["box"][0], ann["box"][1] + ann["box"][3]]                      # xmin, ymax
                    #         ]

                    annot_elem = {
                        "id": int(str(annotations_map[file]) + image_id + str(ann_id_counter).zfill(2)),    # 1100047901
                        "bbox": [
                            float(ann["box"][0]),
                            float(ann["box"][1]),
                            float(ann["box"][2]),
                            float(ann["box"][3])
                        ],
                        "segmentation": list(),  # list([poly])
                        "image_id": image_id_int,
                        "ignore": 0,
                        "category_id": 0,
                        "iscrowd": 0,
                        "area": float(attributes["area"])
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
                    if mode == 'train':
                        train_file["annotations"].append(annot_elem)
                    else:
                        val_file["annotations"].append(annot_elem)

                # if idx == image_limit:
                #     break

with open(save_dir_train, 'w') as json_file:
    json.dump(train_file, json_file, indent=4, sort_keys=True)

with open(save_dir_val, 'w') as json_file:
    json.dump(val_file, json_file, indent=4, sort_keys=True)

# print("max_anns_per_image: ", max_anns_per_image)
# print("max_anns_class_and_img: ", max_anns_class_and_img)