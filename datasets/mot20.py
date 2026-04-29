# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
# MOT20 dataset loader for MOTRv2, adapted from datasets/dance.py.
#
# Expected on-disk layout (args.mot_path is the root, same as DanceTrack):
#
#   {mot_path}/MOT20/train/<SEQ>/img1/000001.jpg   # 6-digit frame names
#   {mot_path}/MOT20/train/<SEQ>/gt/gt.txt         # standard MOT gt format
#   {mot_path}/MOT20/test/<SEQ>/img1/...           # no gt
#   {mot_path}/crowdhuman/...                      # optional, shared with dance.py
#   {mot_path}/det_db_motrv2.json                  # or a MOT20-specific det_db
#
# Differences vs. dance.py:
#   * MOT20 image files use 6-digit zero padding (000001.jpg) instead of 8.
#   * MOT20 gt.txt has 9 columns (last is visibility); we take first 8 like
#     dance.py does for DanceTrack.
#   * Class filter: MOT20 label == 1 is pedestrian; drop all other labels.
#   * det_db lookup uses .get() so missing proposals degrade gracefully to
#     "no anchor queries" rather than crashing (useful before a dedicated
#     det_db_mot20.json has been generated).
# ------------------------------------------------------------------------

"""MOT20 dataset wrapper for MOTRv2 training / validation."""

from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image
import copy

import datasets.transforms as T
from models.structures import Instances

from random import randint


def is_crowd(ann):
    return 'extra' in ann and 'ignore' in ann['extra'] and ann['extra']['ignore'] == 1


class DetMOTDetection:
    """Video-clip sampler over MOT20 sequences (with optional CrowdHuman)."""

    def __init__(self, args, data_txt_path: str, seqs_folder, transform):
        self.args = args
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.video_dict = {}
        self.mot_path = args.mot_path

        self.labels_full = defaultdict(lambda: defaultdict(list))

        def add_mot20_folder(split_dir):
            print("[MOT20] Adding", split_dir)
            base = os.path.join(self.mot_path, split_dir)
            if not os.path.isdir(base):
                print(f"[MOT20]   skip (not a directory): {base}")
                return
            for seq in sorted(os.listdir(base)):
                if seq.startswith('.') or seq == 'seqmap':
                    continue
                vid = os.path.join(split_dir, seq)
                gt_path = os.path.join(self.mot_path, vid, 'gt', 'gt.txt')
                if not os.path.exists(gt_path):
                    print(f"[MOT20]   warning: no gt.txt for {vid}, skipped")
                    continue
                n_lines = 0
                for l in open(gt_path):
                    parts = l.strip().split(',')
                    if len(parts) < 8:
                        continue
                    t, i, *xywh, mark, label = parts[:8]
                    t, i, mark, label = map(int, (t, i, mark, label))
                    if mark == 0:
                        continue
                    if label != 1:
                        continue
                    x, y, w, h = map(float, xywh)
                    self.labels_full[vid][t].append([x, y, w, h, i, False])
                    n_lines += 1
                print(f"[MOT20]   {vid}: {len(self.labels_full[vid])} frames, "
                      f"{n_lines} boxes")

        add_mot20_folder("MOT20/train")
        if getattr(args, 'append_mot20_test', False):
            add_mot20_folder("MOT20/test")

        vid_files = list(self.labels_full.keys())

        self.indices = []
        self.vid_tmax = {}
        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys()) + 1
            self.vid_tmax[vid] = t_max - 1
            for t in range(t_min, t_max - self.num_frames_per_batch):
                self.indices.append((vid, t))
        print(f"[MOT20] Found {len(vid_files)} videos, {len(self.indices)} clip starts")

        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("[MOT20] sampler_steps={} lengths={}".format(self.sampler_steps, self.lengths))
        self.period_idx = 0

        # CrowdHuman (optional; reuses the same layout/det_db keys as dance.py).
        self.ch_dir = Path(args.mot_path) / 'crowdhuman'
        self.ch_indices = []
        if args.append_crowd:
            ch_ann = self.ch_dir / "annotation_trainval.odgt"
            if ch_ann.exists():
                for line in open(ch_ann):
                    datum = json.loads(line)
                    boxes = [ann['fbox'] for ann in datum['gtboxes'] if not is_crowd(ann)]
                    self.ch_indices.append((datum['ID'], boxes))
            else:
                print(f"[MOT20] --append_crowd set but {ch_ann} not found, skipping")
        print(f"[MOT20] CrowdHuman images: {len(self.ch_indices)}")

        # Detection proposal database (YOLOX-style anchors keyed by image path).
        if args.det_db:
            det_db_path = os.path.join(args.mot_path, args.det_db)
            print(f"[MOT20] Loading detection DB from {det_db_path}")
            with open(det_db_path) as f:
                self.det_db = json.load(f)
        else:
            self.det_db = {}

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            return
        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt_instances.boxes = targets['boxes'][:n_gt]
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        return gt_instances

    def load_crowd(self, index):
        ID, boxes = self.ch_indices[index]
        boxes = copy.deepcopy(boxes)
        img = Image.open(self.ch_dir / 'Images' / f'{ID}.jpg')

        w, h = img._size
        n_gts = len(boxes)
        scores = [0. for _ in range(len(boxes))]
        for line in self.det_db.get(f'crowdhuman/train_image/{ID}.txt', []):
            *box, s = map(float, line.split(','))
            boxes.append(box)
            scores.append(s)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        areas = boxes[..., 2:].prod(-1)
        boxes[:, 2:] += boxes[:, :2]

        target = {
            'boxes': boxes,
            'scores': torch.as_tensor(scores),
            'labels': torch.zeros((n_gts,), dtype=torch.long),
            'iscrowd': torch.zeros((n_gts,), dtype=torch.bool),
            'image_id': torch.tensor([0]),
            'area': areas,
            'obj_ids': torch.arange(n_gts),
            'size': torch.as_tensor([h, w]),
            'orig_size': torch.as_tensor([h, w]),
            'dataset': "CrowdHuman",
        }
        rs = T.FixedMotRandomShift(self.num_frames_per_batch)
        return rs([img], [target])

    def _pre_single_frame(self, vid, idx: int):
        # MOT20 uses 6-digit zero-padded frame filenames: 000001.jpg ...
        img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:06d}.jpg')
        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        obj_idx_offset = self.video_dict[vid] * 100000  # per-video unique id offset

        targets['dataset'] = 'MOT20'
        targets['boxes'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['scores'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])

        for *xywh, tid, crowd in self.labels_full[vid].get(idx, []):
            targets['boxes'].append(xywh)
            assert not crowd
            targets['iscrowd'].append(crowd)
            targets['labels'].append(0)
            targets['obj_ids'].append(tid + obj_idx_offset)
            targets['scores'].append(1.)

        # Detection proposals keyed by the image path (xywh format).
        txt_key = os.path.join(vid, 'img1', f'{idx:06d}.txt')
        for line in self.det_db.get(txt_key, []):
            *box, s = map(float, line.split(','))
            targets['boxes'].append(box)
            targets['scores'].append(s)

        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.float64)
        targets['scores'] = torch.as_tensor(targets['scores'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['boxes'][:, 2:] += targets['boxes'][:, :2]  # xywh -> xyxy
        return img, targets

    def _get_sample_range(self, start_idx):
        assert self.sample_mode in ['fixed_interval', 'random_interval'], \
            'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = (start_idx,
                         start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1,
                         sample_interval)
        return default_range

    def pre_continuous_frames(self, vid, indices):
        return zip(*[self._pre_single_frame(vid, i) for i in indices])

    def sample_indices(self, vid, f_index):
        assert self.sample_mode == 'random_interval'
        rate = randint(1, self.sample_interval + 1)
        tmax = self.vid_tmax[vid]
        ids = [f_index + rate * i for i in range(self.num_frames_per_batch)]
        return [min(i, tmax) for i in ids]

    def __getitem__(self, idx):
        if idx < len(self.indices):
            vid, f_index = self.indices[idx]
            indices = self.sample_indices(vid, f_index)
            images, targets = self.pre_continuous_frames(vid, indices)
        else:
            images, targets = self.load_crowd(idx - len(self.indices))
        if self.transform is not None:
            images, targets = self.transform(images, targets)
        gt_instances, proposals = [], []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
            n_gt = len(targets_i['labels'])
            proposals.append(torch.cat([
                targets_i['boxes'][n_gt:],
                targets_i['scores'][n_gt:, None],
            ], dim=1))
        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'proposals': proposals,
        }

    def __len__(self):
        return len(self.indices) + len(self.ch_indices)


class DetMOTDetectionValidation(DetMOTDetection):
    def __init__(self, args, seqs_folder, transform):
        args.data_txt_path = args.val_data_txt_path
        super().__init__(args, seqs_folder, transform)


def make_transforms_for_mot20(image_set, args=None):
    """Augmentation pipeline tuned for MOT20's dense, high-resolution scenes."""
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # MOT20 images are up to 1920x1080; use slightly higher scale range.
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([800, 1000, 1200]),
                    T.FixedMotRandomCrop(800, 1200),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            T.MOTHSV(),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_transform(args, image_set):
    mot20_train = make_transforms_for_mot20('train', args)
    mot20_val = make_transforms_for_mot20('val', args)
    if image_set == 'train':
        return mot20_train
    if image_set == 'val':
        return mot20_val
    raise NotImplementedError()


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    transform = build_transform(args, image_set)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path,
                                  seqs_folder=root, transform=transform)
    elif image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path,
                                  seqs_folder=root, transform=transform)
    else:
        raise NotImplementedError()
    return dataset
