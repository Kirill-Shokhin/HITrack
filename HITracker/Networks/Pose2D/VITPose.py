from .ViTPose_pytorch.vit_models.model import ViTPose
from .ViTPose_pytorch.utils.top_down_eval import keypoints_from_heatmaps
from .ViTPose_pytorch.configs import ViTPose_base_coco_256x192, ViTPose_large_coco_256x192, ViTPose_huge_coco_256x192

from torchvision.transforms import transforms
import cv2
import os
import torch
import numpy as np

model_map = {'b': (ViTPose_base_coco_256x192, 'vitpose-b-multi-coco.pth'),
             'l': (ViTPose_large_coco_256x192, 'vitpose-l-multi-coco.pth'),
             'h': (ViTPose_huge_coco_256x192, 'vitpose-h-multi-coco.pth')}


class VITPOSE:
    def __init__(self, model_version='b'):
        self.cfg, model_name = model_map[model_version]
        self.model_path = os.path.join(os.path.dirname(__file__), 'ViTPose_pytorch', model_name)
        self.model_img_size = self.cfg.data_cfg['image_size']
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.model_init()

    def predict(self, img):
        org_w, org_h = img.shape[:2][::-1]

        # preprocess
        resized = cv2.resize(img, self.model_img_size, interpolation=cv2.INTER_AREA)
        img_tensor = transforms.ToTensor()(resized).unsqueeze(0).to(self.device)
        #         img_tensor = transforms.Compose (
        #                         [transforms.ToPILImage(),
        #                         transforms.Resize(self.model_img_size[::-1]),
        #                          transforms.ToTensor()]
        #                             )(img).unsqueeze(0).to(self.device)

        # process
        with torch.no_grad():
            heatmaps = self.model(img_tensor).detach().cpu().numpy()

        # decoder
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w // 2, org_h // 2]]),
                                               scale=np.array([[org_w, org_h]]), unbiased=True, use_udp=True)
        keypoints = np.concatenate([points[:, :, ::-1], prob], axis=2)[0]
        return keypoints

    def __call__(self, rgb, boxes):
        keypoints_list = []
        for x1, y1, x2, y2 in boxes.astype(int):
            crop = rgb[y1:y2, x1:x2]
            keypoints = self.predict(crop)
            keypoints[:, :2] = keypoints[:, :2][:, ::-1] + [x1, y1]
            keypoints_list.append(keypoints)
        return np.float64(keypoints_list)

    def model_init(self):
        model = ViTPose(self.cfg.model)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.to(self.device).eval()
        return model
