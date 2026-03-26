# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
import torch

def get_keys_from_pth(path):
    try:
        ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict):
            # if saved like {"model": ..., ...}
            if 'model' in ckpt and isinstance(ckpt['model'], dict):
                return set(ckpt['model'].keys())
            # else may be just state_dict, or the full dict is the state dict
            return set(ckpt.keys())
        else:
            return set()
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return set()

path1 = "/home/sata/JoshuaWen/Models/MOTRv2/motrv2_dancetrack.pth"
path2 = "/home/sata/JoshuaWen/Models/MOTRv2/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth"

keys1 = get_keys_from_pth(path1)
keys2 = get_keys_from_pth(path2)

print(f"\nFile 1: {path1}")
print(f"Number of keys: {len(keys1)}")
print(f"\nFile 2: {path2}")
print(f"Number of keys: {len(keys2)}")

print("\nKeys only in File 1:")
for k in sorted(keys1 - keys2):
    print(k)

print("\nKeys only in File 2:")
for k in sorted(keys2 - keys1):
    print(k)

print("\nKeys in both files:")
for k in sorted(keys1 & keys2):
    print(k)