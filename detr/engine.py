# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    # for samples, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        img_id = targets[0]["image_id"].item()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]        # k: 'image_id', 'annotations'

        #   memory usage --> check for memory leak
        # print("CUDA GPU STATISTICS: ")
        # print(torch.cuda.memory_summary())
        # print(torch.cuda.memory_stats())

        outputs = model(samples)
        # print("outputs: ", outputs)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # <!----- commencing the 2kitti reformatting -----!>

        # annotations_map = {
        #     11: 'bytes-cafe-2019-02-07_0',
        #     12: 'clark-center-2019-02-28_0',
        #     13: 'clark-center-2019-02-28_1',
        #     14: 'clark-center-intersection-2019-02-28_0',
        #     15: 'cubberly-auditorium-2019-04-22_0',
        #     16: 'forbes-cafe-2019-01-22_0',
        #     17: 'gates-159-group-meeting-2019-04-03_0',
        #     18: 'gates-ai-lab-2019-02-08_0',
        #     19: 'gates-basement-elevators-2019-01-17_1',
        #     20: 'gates-to-clark-2019-02-28_1',
        #     21: 'hewlett-packard-intersection-2019-01-24_0',
        #     22: 'huang-2-2019-01-25_0',
        #     23: 'huang-basement-2019-01-25_0',
        #     24: 'huang-lane-2019-02-12_0',
        #     25: 'jordan-hall-2019-04-22_0',
        #     26: 'memorial-court-2019-03-16_0',
        #     27: 'meyer-green-2019-03-16_0',
        #     28: 'nvidia-aud-2019-04-18_0',
        #     29: 'packard-poster-session-2019-03-20_0',
        #     30: 'packard-poster-session-2019-03-20_1',
        #     31: 'packard-poster-session-2019-03-20_2',
        #     32: 'stlc-111-2019-04-19_0',
        #     33: 'svl-meeting-gates-2-2019-04-08_0',
        #     34: 'svl-meeting-gates-2-2019-04-08_1',
        #     35: 'tressider-2019-03-16_0',
        #     36: 'tressider-2019-03-16_1',
        #     37: 'tressider-2019-04-26_2'
        # }
        #
        # image_id_str = str(img_id)
        # seq_name, seq_idx = annotations_map[int(image_id_str[:2])], image_id_str[2:]
        # img_size = (3760, 480)
        # label_lines = []
        # # output_dir = "../outputs2kitti/"
        # output_dir = "../detection_eval/ExperimentSetup/wot+wofo/Train/image_stitched/"
        #
        # pred_logits = outputs["pred_logits"]
        # pred_boxes = outputs["pred_boxes"]
        #
        # probas = pred_logits.softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.9
        #
        # probas = probas[keep]  # .to('cpu').numpy()
        # pred_boxes = pred_boxes[0, keep]
        #
        # bboxes_scaled = rescale_bboxes(pred_boxes, img_size)  # .to('cpu').numpy()
        #
        # for i in range(probas.size()[0]):
        #     x1_2d = bboxes_scaled[i][0].item()
        #     y1_2d = bboxes_scaled[i][1].item()
        #     x2_2d = bboxes_scaled[i][2].item()
        #     y2_2d = bboxes_scaled[i][3].item()
        #     score = probas[i].item()
        #
        #     # print(x1_2d,
        #     #       y1_2d,
        #     #       x2_2d,
        #     #       y2_2d,
        #     #       score)
        #     label_lines.append(
        #         "Pedestrian %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % \
        #         # (truncated, occlusion, num_points_3d, alpha, x1_2d, y1_2d, x2_2d, y2_2d, height_3d, width_3d, length_3d,
        #         #  centerx_3d, centery_3d, centerz_3d, rotation_y)
        #         (0, 0, -1, -1, x1_2d, y1_2d, x2_2d, y2_2d, -1, -1, -1,
        #          -1, -1, -1, -1, score)
        #     )
        #
        # # print(probas, probas.size())
        # # print(bboxes_scaled, bboxes_scaled.size())
        #
        # # Write label text file to the output directory.
        # seq_dir = os.path.join(output_dir, seq_name)
        # os.makedirs(seq_dir, exist_ok=True)
        # with open(os.path.join(seq_dir, str(seq_idx) + '.txt'), 'w') as f:
        #     f.writelines(label_lines)

        # <!----- end of 2kitti reformatting -----!>

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox).to('cuda:0')
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to('cuda:0')
    return b


# edited by m.sain
# @torch.no_grad()
# def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, epoch_num):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # for samples, targets, image_id in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        img_id = targets[0]["image_id"].item()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # <!----- commencing the 2kitti reformatting -----!>

        annotations_map = {
            11: 'bytes-cafe-2019-02-07_0',
            12: 'clark-center-2019-02-28_0',
            13: 'clark-center-2019-02-28_1',
            14: 'clark-center-intersection-2019-02-28_0',
            15: 'cubberly-auditorium-2019-04-22_0',
            16: 'forbes-cafe-2019-01-22_0',
            17: 'gates-159-group-meeting-2019-04-03_0',
            18: 'gates-ai-lab-2019-02-08_0',
            19: 'gates-basement-elevators-2019-01-17_1',
            20: 'gates-to-clark-2019-02-28_1',
            21: 'hewlett-packard-intersection-2019-01-24_0',
            22: 'huang-2-2019-01-25_0',
            23: 'huang-basement-2019-01-25_0',
            24: 'huang-lane-2019-02-12_0',
            25: 'jordan-hall-2019-04-22_0',
            26: 'memorial-court-2019-03-16_0',
            27: 'meyer-green-2019-03-16_0',
            28: 'nvidia-aud-2019-04-18_0',
            29: 'packard-poster-session-2019-03-20_0',
            30: 'packard-poster-session-2019-03-20_1',
            31: 'packard-poster-session-2019-03-20_2',
            32: 'stlc-111-2019-04-19_0',
            33: 'svl-meeting-gates-2-2019-04-08_0',
            34: 'svl-meeting-gates-2-2019-04-08_1',
            35: 'tressider-2019-03-16_0',
            36: 'tressider-2019-03-16_1',
            37: 'tressider-2019-04-26_2'
        }
        # annotations_map = {
        #     11: 'cubberly-auditorium-2019-04-22_1',
        #     12: 'discovery-walk-2019-02-28_0',
        #     13: 'discovery-walk-2019-02-28_1',
        #     14: 'food-trucks-2019-02-12_0',
        #     15: 'gates-ai-lab-2019-04-17_0',
        #     16: 'gates-basement-elevators-2019-01-17_0',
        #     17: 'gates-foyer-2019-01-17_0',
        #     18: 'gates-to-clark-2019-02-28_0',
        #     19: 'hewlett-class-2019-01-23_0',
        #     20: 'hewlett-class-2019-01-23_1',
        #     21: 'huang-2-2019-01-25_1',
        #     22: 'huang-intersection-2019-01-22_0',
        #     23: 'indoor-coupa-cafe-2019-02-06_0',
        #     24: 'lomita-serra-intersection-2019-01-30_0',
        #     25: 'meyer-green-2019-03-16_1',
        #     26: 'nvidia-aud-2019-01-25_0',
        #     27: 'nvidia-aud-2019-04-18_1',
        #     28: 'nvidia-aud-2019-04-18_2',
        #     29: 'outdoor-coupa-cafe-2019-02-06_0',
        #     30: 'quarry-road-2019-02-28_0',
        #     31: 'serra-street-2019-01-30_0',
        #     32: 'stlc-111-2019-04-19_1',
        #     33: 'stlc-111-2019-04-19_2',
        #     34: 'tressider-2019-03-16_2',
        #     35: 'tressider-2019-04-26_0',
        #     36: 'tressider-2019-04-26_1',
        #     37: 'tressider-2019-04-26_3'
        # }

        image_id_str = str(img_id)
        seq_name, seq_idx = annotations_map[int(image_id_str[:2])], image_id_str[2:]
        img_size = (3760, 480)
        label_lines = []
        label_lines5 = []

        # output_dir = "../outputs2kitti/"
        # output_dir = "../detection_eval/ExperimentSetup/wot+wofo/Val/image_stitched/"
        # output_dir_0_val = f'../detection_eval/ExperimentSetup/wot+wofo/E{epoch_num}/th_0/Val/'
        output_dir_5_val = f'../detection_eval/ExperimentSetup/wot+wofo/E{epoch_num}/th_0.5/Val/'
        # output_dir_0_train = f'../detection_eval/ExperimentSetup/wot+wofo/E{epoch_num}/th_0/Train/'
        # output_dir_5_train = f'../detection_eval/ExperimentSetup/wot+wofo/E{epoch_num}/th_0.5/Train/'
        output_dir_5_test = f'../detection_eval/ExperimentSetup/wot+wofo/E{epoch_num}/th_0.5/Test/'

        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        probas = pred_logits.softmax(-1)[0, :, :-1]

        keep5 = probas.max(-1).values >= 0.5

        probas5 = probas[keep5]   # .to('cpu').numpy()
        pred_boxes5 = pred_boxes[0, keep5]

        # print(pred_boxes, pred_boxes.size())
        # bboxes_scaled = rescale_bboxes(pred_boxes[0, :], img_size)    # .to('cpu').numpy()
        bboxes_scaled5 = rescale_bboxes(pred_boxes5, img_size)    # .to('cpu').numpy()

        # for i in range(probas.size()[0]):
        #     x1_2d = bboxes_scaled[i][0].item()
        #     y1_2d = bboxes_scaled[i][1].item()
        #     x2_2d = bboxes_scaled[i][2].item()
        #     y2_2d = bboxes_scaled[i][3].item()
        #     score = probas[i].item()
        #
        #     # print(x1_2d,
        #     #       y1_2d,
        #     #       x2_2d,
        #     #       y2_2d,
        #     #       score)
        #     label_lines.append(
        #         "Pedestrian %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % \
        #         # (truncated, occlusion, num_points_3d, alpha, x1_2d, y1_2d, x2_2d, y2_2d, height_3d, width_3d, length_3d,
        #         #  centerx_3d, centery_3d, centerz_3d, rotation_y)
        #         (0, 0, -1, -1, x1_2d, y1_2d, x2_2d, y2_2d, -1, -1, -1,
        #          -1, -1, -1, -1, score)
        #     )


        for i in range(probas5.size()[0]):
            x1_2d = bboxes_scaled5[i][0].item()
            y1_2d = bboxes_scaled5[i][1].item()
            x2_2d = bboxes_scaled5[i][2].item()
            y2_2d = bboxes_scaled5[i][3].item()
            score = probas5[i].item()

            # print(x1_2d,
            #       y1_2d,
            #       x2_2d,
            #       y2_2d,
            #       score)
            label_lines5.append(
                "Pedestrian %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % \
                # (truncated, occlusion, num_points_3d, alpha, x1_2d, y1_2d, x2_2d, y2_2d, height_3d, width_3d, length_3d,
                #  centerx_3d, centery_3d, centerz_3d, rotation_y)
                (0, 0, -1, -1, x1_2d, y1_2d, x2_2d, y2_2d, -1, -1, -1,
                 -1, -1, -1, -1, score)
            )

        # # Write label text file to the output directory.
        # seq_dir = os.path.join(output_dir_0_train, seq_name + '/image_stitched/')
        # os.makedirs(seq_dir, exist_ok=True)
        # with open(os.path.join(seq_dir, str(seq_idx)+'.txt'), 'w') as f:
        #     f.writelines(label_lines)

        # Write label text file to the output directory.
        seq_dir = os.path.join(output_dir_5_val, seq_name + '/image_stitched/')
        os.makedirs(seq_dir, exist_ok=True)
        with open(os.path.join(seq_dir, str(seq_idx)+'.txt'), 'w') as f:
            f.writelines(label_lines5)

        # print(sequence, img_name)

        # print(keep, keep.size())
        # print(img_id)
        # print(outputs)
        # print(samples[0][0][0].size())

        # <!----- end of 2kitti reformatting -----!>

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
