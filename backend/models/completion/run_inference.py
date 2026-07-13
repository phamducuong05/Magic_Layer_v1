#!/usr/bin/env python3
import os
import argparse
import yaml
import numpy as np
import cv2
import torch
from PIL import Image

import models
import inference as infer
import utils

def expand_bbox(bboxes, enlarge_box):
    new = []
    for b in bboxes:
        cx, cy = b[0] + b[2] / 2., b[1] + b[3] / 2.
        size = max(np.sqrt(b[2] * b[3] * enlarge_box), b[2] * 1.1, b[3] * 1.1)
        new.append([int(cx - size / 2.), int(cy - size / 2.), int(size), int(size)])
    return np.array(new)

def masks_to_inputs(modal_masks):
    inmodal, bboxes = [], []
    for m in modal_masks:
        m = (m > 0).astype(np.uint8)
        ys, xs = np.where(m)
        x0, y0 = int(xs.min()), int(ys.min())
        inmodal.append(m)
        bboxes.append([x0, y0, int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)])
    return np.stack(inmodal, 0), np.array(bboxes, dtype=np.int32)

def main(args):
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    image_fn = os.path.basename(args.image)
    
    # Đọc mask
    modal_masks = [np.array(Image.open(p).convert('L')) for p in args.masks]
    inmodal, bboxes = masks_to_inputs(modal_masks)
    H, W = inmodal.shape[1], inmodal.shape[2]
    category = np.ones((len(modal_masks),), dtype=np.int32)
    bboxes = expand_bbox(bboxes, cfg['data']['enlarge_box'])

    # Load model
    print("Đang tải mô hình SDAmodal...")
    model = models.__dict__[cfg['model']['algo']](cfg['model'], dist_model=False)
    model.load_state(args.ckpt)
    model.switch_to('eval')

    # Chạy suy luận (sẽ tự động đọc feature từ folder feature/pthX do bạn đã sửa file inference.py)
    print("Đang suy luận Amodal Completion...")
    patches = infer.infer_amodal_aw_sdm(
        model, image_fn, inmodal, category, bboxes,
        use_rgb=cfg['model']['use_rgb'], th=args.amodal_th,
        input_size=cfg['data']['input_size'],
        min_input_size=16, interp='nearest', args=None)
        
    amodal = infer.patch_to_fullimage(patches, bboxes, H, W, interp='linear')

    # Lưu kết quả
    os.makedirs(args.out, exist_ok=True)
    for i in range(amodal.shape[0]):
        amodal_img = (amodal[i] * 255).astype(np.uint8)
        Image.fromarray(amodal_img).save(os.path.join(args.out, f'amodal_{i}.png'))
        
        holes_img = ((amodal[i] > 0) & (inmodal[i] == 0)).astype(np.uint8) * 255
        Image.fromarray(holes_img).save(os.path.join(args.out, f'holes_{i}.png'))
        
    print(f"Hoàn tất! Đã lưu amodal và holes vào thư mục: {args.out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chạy SDAmodal Inference lấy đầu vào là inmodal mask (thường chạy sau extract_dift)")
    parser.add_argument('--image', required=True, help='Đường dẫn ảnh gốc')
    parser.add_argument('--masks', nargs='+', required=True, help='Đường dẫn các file modal mask (ảnh trắng đen)')
    parser.add_argument('--ckpt', required=True, help='Đường dẫn file trọng số ckpt_SDAmodal.pth')
    parser.add_argument('--config', default='experiments/COCOA/pcnet_m/config_SDAmodal.yaml')
    parser.add_argument('--amodal_th', type=float, default=0.5, help='Ngưỡng xác định mask')
    parser.add_argument('--out', default='amodal_out', help='Thư mục lưu kết quả')
    main(parser.parse_args())
