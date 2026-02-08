# parts/detection.py
import cv2
import torch
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils import recursive_to
from hamer.utils.renderer import cam_crop_to_full
from global_config import HUMAN_HAND_COLOR, INTRINSICS_HAMER_RENDERER

# 导入初始化好的模型
from parts.models_loader import detector, cpm, model, model_cfg, device, renderer, args

#对手部模型进行建模，渲染成深度图
def get_hand_and_rendered_depth(rgb):
    # Get all demo images ends with .jpg or .png
    # Iterate over all images in folder
    img_cv2 = rgb
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # Detect humans in image
    det_out = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
    pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores=det_instances.scores[valid_idx].cpu().numpy()
    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(img, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],)
    bboxes = []
    is_right = []
    # Use hands based on hand keypoint detections
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]
        # Rejecting not confident detections
        keyp = left_hand_keyp
        valid = keyp[:,2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            bboxes.append(bbox)
            is_right.append(0)
        keyp = right_hand_keyp
        valid = keyp[:,2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            bboxes.append(bbox)
            is_right.append(1)

    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    # Run reconstruction on all detected hands
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    all_verts = []
    all_cam_t = []
    all_right = []
    
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        multiplier = (2*batch['right']-1)
        pred_cam = out['pred_cam']
        pred_cam[:,1] = multiplier*pred_cam[:,1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        multiplier = (2*batch['right']-1)
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        INTRINSICS_HAMER_RENDERER[0 ,0] = scaled_focal_length.item()
        INTRINSICS_HAMER_RENDERER[1, 1] = scaled_focal_length.item()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
        # Render the result
        batch_size = batch['img'].shape[0]
        print(f'Batch size: {batch_size}')
        for n in range(batch_size):
            # Get filename from path img_path
            input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
            input_patch = input_patch.permute(1,2,0).numpy()
            # Add all verts and cams to list
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            is_right = batch['right'][n].cpu().numpy()
            verts[:,0] = (2*is_right-1)*verts[:,0]
            cam_t = pred_cam_t_full[n]
            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right)
    all_verts_as_np = np.asarray(all_verts)
    all_verts_as_np = all_verts_as_np[0]
    all_verts_list = [all_verts_as_np]

    misc_args = dict(
        mesh_base_color=HUMAN_HAND_COLOR,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
    )
    cam_view, rend_depth_front_view, mesh_list = renderer.render_rgba_multiple(all_verts_list, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)
    # Overlay image
    input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel


    
    input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
    camera_translation = cam_t.copy()
    hand_mesh = renderer.vertices_to_trimesh(verts, camera_translation, HUMAN_HAND_COLOR, is_right=is_right)
    return input_img_overlay, cam_view, rend_depth_front_view, det_out, out, (hand_mesh, camera_translation, verts), (mesh_list, all_cam_t, all_verts)

#从Detectron2检测结果中提取手部掩码
def get_hand_mask_from_detectron(det_out, show_mask=False, rgb_im=None):
    if show_mask and rgb_im is not None:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        # Make prediction
        outputs = det_out
        v = Visualizer(rgb_im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    try:
        human_mask = det_out["instances"].pred_masks[np.argwhere(det_out['instances'].pred_classes == 0)[0,0].item()].cpu().numpy()
    except:
        human_mask = det_out["instances"].pred_masks[np.argwhere(det_out['instances'].pred_classes.cpu().numpy() == 0)[0,0].item()].cpu().numpy()

    return human_mask
