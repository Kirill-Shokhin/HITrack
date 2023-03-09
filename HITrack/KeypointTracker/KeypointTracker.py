from collections import defaultdict
from filterpy.kalman import KalmanFilter
from catboost import CatBoostRegressor
import pandas as pd
import cv2
import numpy as np
import os
from ..constants import WEIGHTS_COCO


class KeypointTracker:
    def __init__(self, skeletons_init, var_smooth=(100, 100), var_pred=(100, 20), wait_recovery=10,
                 keypoint_weights=WEIGHTS_COCO):
        self.ids = defaultdict()
        self.var_smooth = var_smooth
        self.var_pred = var_pred
        self.frame = 0
        self.wait_recovery = wait_recovery
        self.model = CatBoostRegressor()
        self.model.load_model(os.path.join(os.path.dirname(__file__), 'reg_model_1.cbm'))
        self.cur_max_id = -1
        self.keypoint_weights = keypoint_weights

        for sk in skeletons_init:
            self.ids[self.cur_max_id] = self.create_id(sk)

    def step(self, skeletons):
        self.frame += 1
        self.match, self.map_ids, self.free_ids, self.free_new_ids = self.matching(skeletons)
        for id_, index in self.map_ids:
            new_sk = skeletons[index]
            self.ids[id_]['frame'] = self.frame
            self.ids[id_]['skeleton'] = new_sk
            prediction = self.ids[id_]['predictor'].predict(new_sk[:, :2])
            self.ids[id_]['prediction'] = prediction

        for id_ in self.free_ids:
            # id recovering
            if self.ids[id_]['prediction'] is not None:
                pred_sk = self.ids[id_]['skeleton'][:, :2] + self.ids[id_]['prediction']
                self.ids[id_]['skeleton'] = pred_sk
                prediction = self.ids[id_]['predictor'].predict(pred_sk)
                self.ids[id_]['prediction'] = prediction
            else:
                del self.ids[id_]

        for new_id in self.free_new_ids:
            sk = skeletons[new_id]
            self.ids[self.cur_max_id] = self.create_id(sk)

        self.index_to_del = [x for x in self.ids if self.frame - self.ids[x]['frame'] > self.wait_recovery]
        for index in self.index_to_del:
            del self.ids[index]

    def create_id(self, sk, prediction=None):
        self.cur_max_id += 1
        return defaultdict(None, [('id', self.cur_max_id), ('frame', self.frame), ('skeleton', sk),
                                  ('predictor', KF_skeleton_predictor(sk[:, :2], self.var_smooth, self.var_pred)),
                                  ('prediction', prediction)])

    def current(self):
        ids = [id_ for id_ in self.ids if self.ids[id_]['frame'] == self.frame]
        skeletons = np.float64([self.ids[id_]['skeleton'] for id_ in ids])
        return skeletons, ids

    def matching(self, skeletons, tresh=0.5):
        match = pd.DataFrame()
        for id_ in self.ids:
            skeleton1 = self.ids[id_]['skeleton']
            prediction = self.ids[id_]['prediction']
            for j, skeleton2 in enumerate(skeletons):
                X, ret = self.get_statistics(skeleton1, skeleton2, prediction)
                if ret:
                    if prediction is None:
                        match.loc[id_, j] = X[-1] ** (-1)
                    else:
                        match.loc[id_, j] = self.model.predict(X)

        match = match[match > tresh].sort_index(axis=1)
        map_ids, free_ids, free_new_ids = self.get_indexes_from(match)
        return match, map_ids, free_ids, free_new_ids

    def get_statistics(self, skeleton1, skeleton2, prediction=None, ratio=3.6, amount=0.1):
        skeleton1, skeleton2 = skeleton1[:, :2], skeleton2[:, :2]  # TODO: append confidence dependency

        ft = fluctuation_threshold(skeleton1)
        d_sk = skeleton_distance(skeleton1, skeleton2)
        r, phi = xy2rphi(V(skeleton2 - skeleton1, self.keypoint_weights))
        if prediction is None:
            return [1, np.inf, 0, 1, np.inf, 0, d_sk / ft], locality_criterion(r, ft, ratio, amount)
        else:
            d_sk_pred = skeleton_distance(skeleton1 + prediction, skeleton2)
            d_CM_pred = CM_distance(skeleton1 + prediction, skeleton2, self.keypoint_weights)
            r_pred, phi_pred = xy2rphi(V(prediction, self.keypoint_weights))

            angle = np.abs(phi - phi_pred)
            angle = 360 - angle if angle > 180 else angle

        return [d_sk_pred / d_sk, r / r_pred, angle, d_CM_pred / r, d_CM_pred / r_pred, r_pred / ft,
                d_sk / ft], locality_criterion(r, ft, ratio, amount)

    @staticmethod
    def get_indexes_from(match):
        idx = np.array(list(match.stack().sort_values(ascending=False).index))

        axis1, axis2 = [], []
        for a, b in idx:
            if (a not in axis1) and (b not in axis2):
                axis1.append(a), axis2.append(b)

        free_ids = [i for i in match.index if i not in axis1]
        free_new_ids = [i for i in match.columns if i not in axis2]
        map_ids = pd.DataFrame([axis1, axis2]).T.sort_values(0).to_numpy()
        return map_ids, free_ids, free_new_ids


def locality_criterion(r, ft, ratio=3.6, amount=0.1):
    return r < ratio * ft / amount


class KF:
    def __init__(self, pos=0.0, var1=50, var2=10):
        self.f = KalmanFilter(dim_x=2, dim_z=1)
        self.f.x = np.array([pos, 0.])
        self.f.F = np.array([[1., 1.],
                             [0., 1.]])
        self.f.H = np.array([[1., 0.]])
        self.f.P = np.array([[var1, 0.],
                             [0., var1]])
        self.f.R = np.array([[var2]])

    def predict(self):
        self.f.predict()

    def update(self, z):
        self.f.update(z)

    def get_value(self):
        return self.f.x[0]


class KF_smooth_predictor:
    def __init__(self, x=0, var_smooth=(1000, 1000), var_pred=(100, 20)):
        self.kf_smooth = KF(x, *var_smooth)
        self.prev = x
        self.state = 0
        self.var_pred = var_pred

    def diff_smooth(self, x):

        self.kf_smooth.predict()
        self.kf_smooth.update(x)
        diff = self.kf_smooth.get_value() - self.prev
        self.prev = self.kf_smooth.get_value()

        return diff

    def predict(self, x=None):
        if self.state == 0:

            self.diff = self.diff_smooth(x)
            self.kf_pred = KF(self.diff, *self.var_pred)
            self.state = 1

        else:
            self.kf_pred.update(self.diff_smooth(x))
            self.kf_pred.predict()

        return self.kf_pred.get_value()


class KF_smooth_predictor_2D:
    def __init__(self, x=0, y=0, var_smooth=(1000, 1000), var_pred=(100, 20)):
        self.kfx = KF_smooth_predictor(x, var_smooth, var_pred)
        self.kfy = KF_smooth_predictor(y, var_smooth, var_pred)

    def predict(self, x, y):
        return [self.kfx.predict(x), self.kfy.predict(y)]


class KF_skeleton_predictor:
    def __init__(self, skeleton, var_smooth=(100, 100), var_pred=(100, 20)):
        self.skeleton_predictor = defaultdict()
        self.shape = skeleton.shape

        for i, point in enumerate(skeleton):
            self.skeleton_predictor[i] = KF_smooth_predictor_2D(*point, var_smooth, var_pred)

    def predict(self, skeleton):
        assert skeleton.shape == self.shape

        preds = []
        for i, point in enumerate(skeleton):
            preds.append(self.skeleton_predictor[i].predict(*point))

        return np.array(preds)


def xy2rphi(points):
    points = np.array(points)
    x, y = points.T
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.degrees(np.arctan2(y, x))
    return np.array([r, phi]).T


def ceneter_mass(skeleton, keypoint_weights):
    return skeleton.T @ keypoint_weights


def V(prediction, keypoint_weights):
    return prediction.T @ keypoint_weights


def skeleton_distance(skeleton1, skeleton2):
    return np.sqrt(((skeleton1 - skeleton2) ** 2).sum(axis=1)).mean()


def CM_distance(skeleton1, skeleton2, keypoint_weights):
    return np.sqrt(
        ((ceneter_mass(skeleton1, keypoint_weights) - ceneter_mass(skeleton2, keypoint_weights)) ** 2).sum(axis=0))


def fluctuation_threshold(skeleton, amount=0.1):
    return (skeleton.max(0) - skeleton.min(0)).mean() * amount


def crop_person(skeleton, img, pad=20):
    x0, y0, x1, y1 = np.int32([skeleton.min(0), skeleton.max(0)]).flatten()
    return img[y0 - pad:y1 + pad, x0 - pad:x1 + pad]


def crop_person_fixed(skeleton, img, keypoint_weights, crop=100):
    x, y = ceneter_mass(skeleton, keypoint_weights).astype(int)
    return img[y - crop:y + crop, x - crop:x + crop]


def show_pred(skeleton, pred, img):
    for x, y, dx, dy in np.hstack((skeleton, pred)):
        cv2.line(img, (int(x), int(y)), (int(x + dx), int(y + dy)), (255, 0, 0), thickness=1)
        cv2.circle(img, (int(x + dx), int(y + dy)), thickness=-1, color=(255, 0, 0), radius=1)
    return img


def show_V(skeleton, pred, img, keypoint_weights):
    x, y = ceneter_mass(skeleton, keypoint_weights).astype(int)
    dx, dy = V(pred)
    cv2.line(img, (int(x), int(y)), (int(x + dx), int(y + dy)), (255, 0, 0), thickness=1)
    cv2.circle(img, (int(x + dx), int(y + dy)), thickness=-1, color=(255, 0, 0), radius=1)
    return img
