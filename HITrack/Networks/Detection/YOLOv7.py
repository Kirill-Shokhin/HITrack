from .yolov7 import autoShape, attempt_load, attempt_download_from_hub,\
    attempt_download, TracedModel
import os


class YOLOv7:
    def __init__(self, model_name='yolov7', conf=0.5, iou=0.45, device='cuda'):
        model = load_model(os.path.join(os.path.dirname(__file__), 'yolov7',
                                        'models', model_name + '.pt'), device)

        model.conf, model.iou, model.classes = conf, iou, [0]  # (optional list) filter by class
        self.model = model

    def __call__(self, rgb):
        boxes = self.model(rgb).pred[0].cpu().detach().numpy()[:, :4]
        return boxes


def load_model(model_path, device='cpu', autoshape=True, trace=False, size=640, half=False, hf_model=False):
    """
    Creates a specified YOLOv7 model
    Arguments:
        model_path (str): path of the model
        device (str): select device that model will be loaded (cpu, cuda)
        trace (bool): if True, model will be traced
        size (int): size of the input image
        half (bool): if True, model will be in half precision
        hf_model (bool): if True, model will be loaded from huggingface hub
    Returns:
        pytorch model
    (Adapted from yolov7.hubconf.create)
    """
    if hf_model:
        model_file = attempt_download_from_hub(model_path)
    else:
        model_file = attempt_download(model_path)

    model = attempt_load(model_file, map_location=device)
    if trace:
        model = TracedModel(model, device, size)

    if autoshape:
        model = autoShape(model)

    if half:
        model.half()

    return model
