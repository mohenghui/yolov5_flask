import os
import cv2
from base_camera import BaseCamera
import torch
from pathlib import Path
import sys
from models.common import DetectMultiBackend
from utils.datasets import  LoadImages
from utils.general import (check_img_size,increment_path, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Camera(BaseCamera):
    video_source = None

    def __init__(self, source_name="", img_flag=True):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        Camera.video_source = source_name
        self.img_flag = img_flag
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        out, weights, imgsz = \
            ROOT / 'inference/output', ROOT / 'runs/train/exp/weights/best.pt', 640
        # device = select_device(0)  # 0 or 0,1,2,3 or cpu 没有gpu的记得改为cpu
        device = select_device("cpu")  # 0 or 0,1,2,3 or cpu 没有gpu的记得改为cpu
        # Directories
        save_dir = increment_path(Path(out) / 'exp', exist_ok=False)  # increment run
        (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        model = DetectMultiBackend(weights, device=device, dnn=False, data=ROOT / 'data/my_person.yaml')

        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        half = False
        half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()

        # Dataloader
        print(Camera.video_source)
        dataset = LoadImages(Camera.video_source, img_size=imgsz)
        vid_path, vid_writer = [None], [None]
        # save_result
        save_img = True
        # Run inference
        t0 = time_sync()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap, s in dataset:

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_sync()
            pred = model(img, augment=False, visualize=False)

            # nms
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                                       max_det=1000)
            t2 = time_sync()

            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)
                s += '%gx%g ' % img.shape[2:]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                annotator = Annotator(im0, line_width=3, example=str(names))
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, -1].detach().unique():
                        n = (det[:, -1] == c).sum()
                        s += '%g %s, ' % (n, names[int(c)])
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        c = int(cls)
                        annotator.box_label(xyxy, label, color=colors(c, True))

                print('%sDone. (%.3fs)' % (s, t2 - t1))
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            yield cv2.imencode('.jpg', im0)[1].tobytes()
