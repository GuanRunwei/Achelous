from random import sample, shuffle
import re
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import albumentations as A
from utils.utils import cvtColor, preprocess_input, preprocess_input_radar
from utils_seg.utils import preprocess_input as preprocess_input_seg
from utils_seg_line.utils import generate_black_images
import random as rd
import matplotlib.pyplot as plt
import os


def visualize(image):
    plt.figure(figsize=(20, 10))
    plt.axis('off')
    plt.imshow(image)


transform_rain = A.Compose(
    [A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=9, p=1)],
)

transform_flare = A.Compose(
    [A.RandomSunFlare(flare_roi=(0.4, 0.4, 1, 0.5), angle_lower=0.8, p=1)],
)

transform_fog = A.Compose(
    [A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, alpha_coef=0.05, p=1)],
)


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, num_classes_seg, epoch_length, radar_root, \
                 mosaic, mixup, mosaic_prob, mixup_prob,
                 seg_dataset_path, water_seg_dataset_path, train, special_aug_ratio=0.7):
        super(YoloDataset, self).__init__()

        # ------------------------- 通用 --------------------------- #
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.epoch_length = epoch_length
        self.train = train
        self.epoch_now = -1
        self.length = len(self.annotation_lines)
        # ---------------------------------------------------------- #

        # ------------------------- 检测 --------------------------- #
        self.num_classes = num_classes
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.special_aug_ratio = special_aug_ratio
        # ---------------------------------------------------------- #

        # ------------------------ 语义分割 ------------------------- #
        self.seg_dataset_path = seg_dataset_path
        self.num_classes_seg = num_classes_seg
        # ----------------------------------------------------------- #

        # ------------------------ 水岸线分割 ------------------------ #
        self.water_seg_dataset_path = water_seg_dataset_path
        # ----------------------------------------------------------- #

        # ------------------------- 雷达 --------------------------- #
        self.radar_root = radar_root
        # ---------------------------------------------------------- #

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        name = self.annotation_lines[index]

        name = os.path.splitext(name.split('/')[-1].split(' ')[0])[0]

        # ---------------------- 分割数据 ------------------- #
        png = Image.open(os.path.join(self.seg_dataset_path, name + ".png"))
        try:
            png_w = Image.open(os.path.join(self.water_seg_dataset_path, name + '.png'))
        except:
            png_w = generate_black_images(water_line_root=self.water_seg_dataset_path, image_id=name,
                                          image_width=1920, image_height=1080)
        # -------------------------------------------------- #

        image, box, radar, png, png_w = self.get_random_data(self.annotation_lines[index], self.input_shape, png, png_w, name, random=self.train)

        image = np.transpose(preprocess_input_seg(np.array(image, dtype=np.float64)), [2, 0, 1])
        box = np.array(box, dtype=np.float64)
        radar = np.array(radar, dtype=np.float64)
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

        png = np.array(png)
        png[png >= self.num_classes_seg] = self.num_classes_seg

        png_w = np.array(png_w)
        png_w[png_w >= 2] = 2
        # -------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        # -------------------------------------------------------#
        seg_labels = np.eye(self.num_classes_seg + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes_seg + 1))

        seg_w_labels = np.eye(2 + 1)[png_w.reshape([-1])]
        seg_w_labels = seg_w_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), 2 + 1))

        return image, box, radar, png, png_w, seg_labels, seg_w_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, seg_label, seg_w_label, id, jitter=.3, hue=.1, sat=0.7, val=0.4, random=False):
        # ------------------------------#
        #   雷达特征读取
        # ------------------------------#
        radar_path = self.radar_root + '/' + id + '.npz'
        radar_data = np.load(radar_path)['arr_0']

        line = annotation_line.split()
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        seg_label = Image.fromarray(np.array(seg_label))
        seg_w_label = Image.fromarray(np.array(seg_w_label))
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape
        # ------------------------------#
        #   获得预测框
        # ------------------------------#
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        # ---------------------------------#
        #   将图像多余的部分加上灰条
        # ---------------------------------#
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', [w, h], (128, 128, 128))
        new_image.paste(image, (dx, dy))

        segment_label = seg_label.resize((nw, nh), Image.NEAREST)
        new_label = Image.new('L', [w, h], (0))
        new_label.paste(segment_label, (dx, dy))

        seg_w_label = seg_w_label.resize((nw, nh), Image.NEAREST)
        new_w_label = Image.new('L', [w, h], (0))
        new_w_label.paste(seg_w_label, (dx, dy))

        # ---------------------------------#
        #   自然天气数据增强
        # ---------------------------------#
        # if random:
        #     weather_random_number = rd.randint(0, 100)
        #
        #     new_image = np.array(new_image, np.float32)
        #     if 0 <= weather_random_number < 15:
        #         new_image = transform_rain(image=new_image)
        #         new_image = new_image['image']
        #     if 15 <= weather_random_number < 30:
        #         new_image = transform_flare(image=new_image)
        #         new_image = new_image['image']
        #     if 30 <= weather_random_number < 65:
        #         new_image = transform_fog(image=new_image)
        #         new_image = new_image['image']

        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        # new_image.save("train.png")
        # new_label.save('train-label.png')

        return new_image, box, radar_data, new_label, new_w_label

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            # ---------------------------------#
            #   每一行进行分割
            # ---------------------------------#
            line_content = line.split()
            # ---------------------------------#
            #   打开图片
            # ---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)

            # ---------------------------------#
            #   图片的大小
            # ---------------------------------#
            iw, ih = image.size
            # ---------------------------------#
            #   保存框的位置
            # ---------------------------------#
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # ---------------------------------#
            #   是否翻转图片
            # ---------------------------------#
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # ------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            # ------------------------------------------#
            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # -----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            # -----------------------------------------------#
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            # ---------------------------------#
            #   对box进行重新处理
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # ---------------------------------#
        #   将图片分割，放在一起
        # ---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对框进行进一步的处理
        # ---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    radars = []
    pngs = []
    pngs_w = []
    seg_labels = []
    seg_w_labels = []
    for img, box, radar, png, png_w, seg_label, seg_w_label in batch:
        images.append(img)
        bboxes.append(box)
        radars.append(radar)
        pngs.append(png)
        pngs_w.append(png_w)
        seg_labels.append(seg_label)
        seg_w_labels.append(seg_w_label)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    radars = torch.from_numpy(np.array(radars)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    pngs_w = torch.from_numpy(np.array(pngs_w)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    seg_w_labels = torch.from_numpy(np.array(seg_w_labels)).type(torch.FloatTensor)
    return images, bboxes, radars, pngs, pngs_w, seg_labels, seg_w_labels


