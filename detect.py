import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging
from yolov5.utils.plots import colors, plot_one_box
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized


def detect( source = None, weights = "", img_size = 640, no_save = False,
            projects = "runs/detect", device = '', name ='exp', conf_thres = 0.25, 
            augment = False, agnostic_nms = False, iou_thres = 0.45, 
            line_thickness = 3):
    """
    This is the detect method.
    Input:  source -> input image (numpy array/opencv format).
            weight -> Path to weight file (*.pt).
            img_size -> Image size, default = 640. Must be divisible by 32.
            devide -> Select device to running detect method 'cpu' or 'cuda'.
            augment -> Choose to using or not using augment when detect.
            conf_thres -> confident threshold.
            iou_thres -> IoU threshold. 
    Output: Return to a base64 image and a list of prediction and bbox. 
    """
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    im0 = source
    img = letterbox(im0, new_shape=imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Run inference
    labels_list = []
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    # for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=agnostic_nms)
    t2 = time_synchronized()
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0)
    
    # Process detections
    for det in pred:  # detections per image
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            # Write results
            for *xyxy, conf, cls in reversed(det):
                # if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) # if opt.save_conf else (cls, *xywh)  # label format
                labels_list.append(('%g ' * len(line)).rstrip() % line + '\n') # Store labels to a list
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}' # Alway show labels
                plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)

        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

    print(f'Done. ({time.time() - t0:.3f}s)')
    return im0, labels_list

# Test
# image = cv2.imread("C:/Users/Admin/Documents/GitHub/MiniProject/mini_project/projects/yolov5_results/testimg/37233691ba64ed88b4e05882d7e41d61.png")
# image, labels = detect(source = image, weights="best2.pt", img_size=512, device='')
# for i in labels:
#     print(i)
# cv2.imshow('ImageWindow', image)
# cv2.waitKey()
    