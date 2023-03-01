import cv2
import numpy as np
from ..constants import PERSON_SKELETONS

# import matplotlib
# COLORS = np.int32(255*np.array([matplotlib.cm.get_cmap('tab20')((x + 0.05) / 20)[:3] for x in range(20)])).tolist()
COLORS = [[31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
          [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150],
          [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148], 
          [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199], 
          [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229]]

COLORS_CSS = ['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a','#d62728',
              '#ff9896','#9467bd','#c5b0d5','#8c564b','#c49c94','#e377c2','#f7b6d2',
              '#7f7f7f','#c7c7c7','#bcbd22','#dbdb8d','#17becf','#9edae5']


class HumanVisualizer:
    def __init__(self, label_scale=1, thickness=2, alpha=0.4, box_size=20, thickness_text=2, skeleton_format='COCO'):
        
        self.colors = COLORS
        self.label_scale = label_scale
        self.thickness = thickness
        self.thickness_text = thickness_text
        self.box_size = box_size
        self.alpha = alpha
        self.pad = label_scale*box_size
        self.point_size = round(1.5*thickness)
        self.skeleton_format = skeleton_format
        self.SKELETON = np.array(PERSON_SKELETONS[skeleton_format])-1

    def __call__(self, image, skeletons, ids=None):
        
        if ids is not None:
            assert len(skeletons) == len(ids)
            for id_, skeleton in zip(ids, skeletons):
                image = self.print_skeleton(image, skeleton, id_)
        else:
            for skeleton in skeletons:
                image = self.print_skeleton(image, skeleton)
                
        return image
    
    def print_skeleton(self, image, skeleton, id_=None):

        assert image.dtype == 'uint8'
        x, y = skeleton.T.astype(int)

        for ci, (j1i, j2i) in enumerate(self.SKELETON):
            p1, p2 = (x[j1i], y[j1i]), (x[j2i], y[j2i])
            cv2.circle(image, p1, self.point_size, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(image, p2, self.point_size, (255,255,255), -1, cv2.LINE_AA)
            cv2.line(image, p1, p2, self.colors[ci % 20], self.thickness, cv2.LINE_AA)
            
        if id_ is not None:    
            self.label(image, self.label_coord(x, y), id_)
            
        return image
        
    def label(self, img, point, id_):
        
        point = (int(point[0]+self.pad), int(point[1]-self.pad))
        (dx, dy) = cv2.getTextSize(str(id_), cv2.FONT_HERSHEY_SIMPLEX, self.label_scale, self.thickness)[0]
        box_coords = (point[0]-self.pad, point[1] - dy-self.pad,
                      point[0] + dx+self.pad, point[1]+self.pad)
        self.rectangle(img, box_coords, id_)
        cv2.putText(img, str(id_), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, self.label_scale,
                (255, 255, 255), self.thickness_text, cv2.LINE_AA)
        
        return img
            
    def rectangle(self, image, box, id_):
        
        x1, y1, x2, y2 = np.clip(np.int32(box), 0, None)
        crop = image[y1:y2, x1:x2]
        rect = np.uint8(np.ones(crop.shape)*self.colors[id_ % 20])
        image[y1:y2, x1:x2] = crop*(1-self.alpha)+rect*self.alpha
        
        return image

    def label_coord(self, x, y):
        coord_x = x.mean()
        if self.skeleton_format == 'COCO':
            coord_y = y[:5].mean()
        elif self.skeleton_format == 'H36':
            coord_y = y[10].mean()
        else:
            raise NotImplementedError('Use only COCO or H36 format')
        return coord_x, coord_y