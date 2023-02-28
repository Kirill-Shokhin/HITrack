import yolov7
import os


class YOLOv7:
    def __init__(self, model_name='yolov7.pt', conf=0.5, iou=0.45, device='cuda'):
        if device == 'tensorRT':
            pass
        else:
            model = yolov7.load(os.path.join(os.path.dirname(__file__), 'models', model_name), device=device)

        model.conf, model.iou, model.classes = conf, iou, [0]  # (optional list) filter by class
        self.model = model

    def __call__(self, rgb):
        boxes = self.model(rgb).pred[0].cpu().detach().numpy()[:, :4]
        return boxes
