"""
DRDB dataset and benchmark which returns image_id for evaluation.

Mostly copy-paste from JRDB.py with some amendments.
"""

from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as JRDB_mask

import datasets.transforms as T

#  edited by m.sain  #
class JRDBDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(JRDBDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertJRDBPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(JRDBDetection, self).__getitem__(idx)   # (loaded img by PIL, corresponding anno file)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
        # return img, target, image_id


def convert_JRDB_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = JRDB_mask.frPyObjects(polygons, height, width)
        mask = JRDB_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertJRDBPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]                                            # FOR ONE IMAGE IMAGE !!! #

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0] # pick non-crowded anno json objects

        boxes = [obj["bbox"] for obj in anno]                                   # pick bboxes from the anno
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]                                            # first two element of all boxes: (x, y)
        boxes[:, 0::2].clamp_(min=0, max=w)                                     # clamp 0, 2 element of boxes: (x, w)
        boxes[:, 1::2].clamp_(min=0, max=h)                                     # clamp 1, 3 element of boxes: (y, h)

        classes = [obj["category_id"] for obj in anno]                          # pick the cat_id for each bbox
        classes = torch.tensor(classes, dtype=torch.int64)                      # json --> tensor

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]               # pick segmentation from each bbox
            masks = convert_JRDB_poly_to_mask(segmentations, h, w)              # find the masks

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]                      # finding the keypoints --> for skeleton
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])        # keep bboxes with h > y and w > x
        boxes = boxes[keep]                                                     # filter the bboxes
        classes = classes[keep]                                                 # filter the classes
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to JRDB api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_JRDB_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        T.Normalize([0.3691, 0.3965, 0.3752], [0.2716, 0.2871, 0.2799])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

#  edited by m.sain  #
def build(image_set, args):
    # root = Path(args.JRDB_path)
    root = Path(args.data_path)
    assert root.exists(), f'provided JRDB path {root} does not exist'
    mode = 'instances'
    # PATHS = {
    #     "train": (root / "train", root / f'clark-center-2019-02-28_1_image0_train.json'),
    #     "val": (root / "validation", root / f'clark-center-intersection-2019-02-28_0_image0_test.json'),
    # }

    # for validation output in training
    PATHS = {
        "train": (root / "images/image_stitched", root / f'train(wo_FO).json'),
        "val": (root / "images/image_stitched", root / f'val(wo_FO).json'),
        # "val": (root / "images/image_stitched", root / f'train(w_FO).json'),
        # "train": (root / "images/image_stitched", root / f'val(w_FO).json'),
    }

    # fot test set outputs
    # PATHS = {
    #     "val": (root / "images/image_stitched", root / f'test(wo_FO).json'),
    #     "train": (root / "images/image_stitched", root / f'test(wo_FO).json'),
    #     # "val": (root / "images/image_stitched", root / f'train(w_FO).json'),
    #     # "train": (root / "images/image_stitched", root / f'val(w_FO).json'),
    # }

    img_folder, ann_file = PATHS[image_set]
    dataset = JRDBDetection(img_folder, ann_file, transforms=make_JRDB_transforms(image_set), return_masks=args.masks)
    return dataset
