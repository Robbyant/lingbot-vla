import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import matplotlib
import einops
from PIL import Image, ImageDraw

try:
    from mdm.model.v2 import MDMModel as v2_morgbd

    from moge.model.v2 import MoGeModel as v2
    from moge.utils.vis import colorize_depth
except:
    print('Load MoGe Module Failed!!')

def make_grid(images, pil_images):
    # Assuming each image is the same size
    
    new_images = []
    new_captions = []
    for image, pil_image in zip(images, pil_images):
        new_images.append(image)
        pil_image = pil_image.resize((image.size[0], image.size[1]))
        new_images.append(pil_image)
        new_captions.append("Predicted")
        new_captions.append("GT")
    
    images = new_images
    captions = new_captions

    width, height = images[0].size
    font_size = 14
    caption_height = font_size + 10

    # Calculate the size of the final image
    images_per_row = min(len(images), 16)  # Round up for odd number of images
    row_count = (len(images) + 1) // images_per_row
    total_width = width * images_per_row
    total_height = (height + caption_height) * row_count

    # Create a new blank image
    new_image = Image.new("RGB", (total_width, total_height), "white")

    draw = ImageDraw.Draw(new_image)

    for i, (image, caption) in enumerate(zip(images, captions)):
        row = i // images_per_row
        col = i % images_per_row
        x_offset = col * width
        y_offset = row * (height + caption_height)
        
        new_image.paste(image, (x_offset, y_offset))
        text_position = (x_offset + 10, y_offset + height)
        draw.text(text_position, caption, fill="red", font_size=font_size)
    
    return new_image

def build_depth_model(config):

    moge_model = v2.from_pretrained(config['depth']['moge_path'])
    for p in moge_model.parameters():
        p.requires_grad = False
    moge_model.cuda()
    moge_model.eval()

    morgbd_model = v2_morgbd.from_pretrained(config['depth']['morgbd_path'])
    for p in morgbd_model.parameters():
        p.requires_grad = False
    morgbd_model.cuda()
    morgbd_model.eval()
    return moge_model, morgbd_model

def get_depth_target(model_type, depth_model, pil_images):
    device = pil_images.device
    B, _, C, H, W = pil_images.shape
    images = einops.rearrange(pil_images, 'b n c h w -> (b n) c h w', n=3).contiguous().float()

    input_images = images / 255.0
    moge_model, morgbd_model = depth_model
    output_moge = moge_model.infer(input_images, resolution_level=3, num_tokens=256, apply_mask=False)
    depth_pred = output_moge['depth'].squeeze().detach().clone() # moge2
    depth_pred = torch.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0)
    depth_pred *= 1
    depth_down_scale = 1
    depth_target, cls_token = morgbd_model.infer_feat(input_images, depth_pred, 
                                            depth_down_scale=depth_down_scale,
                                            resolution_level=3,
                                            num_tokens=256,
                                            enable_depth_mask=False)
    depth_target = depth_target.permute(0, 2, 3, 1)
    depth_target = depth_target.view(depth_target.shape[0], -1, depth_target.shape[-1])

    return depth_target.to(dtype=torch.bfloat16), cls_token

def log_depth(vis_head, depth_pred_feats, depth_target_feats=None, steps=0, config=None, cls_token=None):
    model_type = config['depth']['model_type']
    llm_image_token_size = config['llm']['image_token_size']
    depth_token_size = config['depth']['token_size']
    visual_dir = config['visual_dir']

    if config['mode'] == "direct":
        depth_pred_feats = depth_pred_feats.view(depth_pred_feats.shape[0], llm_image_token_size, llm_image_token_size, depth_pred_feats.shape[-1])
        depth_pred_feats = depth_pred_feats.permute(0, 3, 1, 2)
        depth_pred_feats = F.interpolate(depth_pred_feats, size=(depth_token_size, depth_token_size), mode="bilinear", align_corners=False)
    elif config['mode'] == "query":
        depth_pred_feats = depth_pred_feats.view(depth_pred_feats.shape[0], depth_token_size, depth_token_size, depth_pred_feats.shape[-1])
        depth_pred_feats = depth_pred_feats.permute(0, 3, 1, 2)

    import cv2
    morgbd_model = vis_head
    depth_target_feats = depth_target_feats.view(depth_target_feats.shape[0], depth_token_size, depth_token_size, depth_target_feats.shape[-1])
    depth_target_feats = depth_target_feats.permute(0, 3, 1, 2)
    
    output_morgbd_preds = morgbd_model.dec_depth(depth_pred_feats, cls_token, num_tokens=256, resolution_level=3, img_h=224, img_w=224)
    output_morgbd_targets = morgbd_model.dec_depth(depth_target_feats, cls_token, num_tokens=256, resolution_level=3, img_h=224, img_w=224)

    output_morgbd_preds = output_morgbd_preds['depth_reg'].squeeze().cpu().numpy()
    output_morgbd_targets = output_morgbd_targets['depth_reg'].squeeze().cpu().numpy()

    for idx, (output_morgbd_target, output_morgbd_pred) in enumerate(zip(output_morgbd_targets, output_morgbd_preds)):

        depth_list = [output_morgbd_target, output_morgbd_pred]
        depth_color_list = [cv2.cvtColor(colorize_depth(depth_raw), cv2.COLOR_RGB2BGR) for depth_raw in depth_list]

        depth_concat = np.concatenate(depth_color_list, axis=1)

        dst_path = os.path.join(visual_dir, f"depth_morgbd_{steps}_{idx}.png")
        cv2.imwrite(dst_path,depth_concat)


