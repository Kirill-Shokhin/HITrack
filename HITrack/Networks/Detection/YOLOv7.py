from yolov7.helpers import load_model
import os


class YOLOv7:
    def __init__(self, model_name='yolov7', conf=0.5, iou=0.45, device='cuda'):
        self.model_path = os.path.join(os.path.dirname(__file__),
                                       'models', model_name + '.pt')

        model = load_model(self.model_path, device=device)
        model.conf, model.iou, model.classes = conf, iou, [0]  # (optional list) filter by class
        self.device = device
        self.model = model

    def __call__(self, rgb):
        boxes = self.model(rgb).pred[0].cpu().detach().numpy()[:, :4]
        return boxes