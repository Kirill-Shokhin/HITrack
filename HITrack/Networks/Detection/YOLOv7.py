from yolov7.helpers import load_model
import os


class YOLOv7:
    def __init__(self, model_name='yolov7', conf=0.5, iou=0.45, device='cuda'):
        self.model_path = os.path.join(os.path.dirname(__file__),
                                       'models', model_name + '.pt')
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        model = load_model(self.model_path, device=device)
        model.conf, model.iou, model.classes = conf, iou, [0]  # (optional list) filter by class
        self.device = device
        self.model = model

    def __call__(self, rgb):
        boxes = self.model(rgb, size=1280).pred[0].cpu().detach().numpy()[:, :4]
        return boxes