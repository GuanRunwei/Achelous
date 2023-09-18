import colorsys
import os
import time
import pandas as pd
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont, ImageEnhance
from sklearn.preprocessing import MinMaxScaler, normalize
from nets.Achelous import Achelous, Achelous3T
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config, preprocess_input_radar)
import torch.nn.functional as F
from PIL import Image
from utils.utils_bbox import decode_outputs, non_max_suppression
from utils_seg.utils import resize_image as resize_image_seg
import cv2


class achelous(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path"        : "model_data/mv_gdf_nano_ps_s2.pth",
        "radar_root"        : "E:/Big_Datasets/water_surface/benchmark_new/WaterScenes_new/radar/VOCradar320",
        "radar_pc_root"     : "E:/Big_Datasets/water_surface/benchmark_new/WaterScenes_new/radar/radar_0220/radar",
        "classes_path"      : 'model_data/waterscenes_benchmark.txt',
        "export_path"       : 'export_results',
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape": [320, 320],
        # ---------------------------------------------------------------------#
        #   语义分割的类别数量
        # ---------------------------------------------------------------------#
        "num_classes_seg": 9,
        # ---------------------------------------------------------------------#
        #   所使用的Achelous的版本，'SO', 'S1', 'S2'
        # ---------------------------------------------------------------------#
        "phi": 'S2',
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.25,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   2D雷达map用了几种特征
        # ---------------------------------------------------------------------#
        "radar_channels": 3,
        # ---------------------------------------------------------------------#
        #   backbone
        # ---------------------------------------------------------------------#
        "backbone": 'mv',
        # ---------------------------------------------------------------------#
        #   neck
        # ---------------------------------------------------------------------#
        "neck": 'gdf',
        # ---------------------------------------------------------------------#
        #   spp
        # ---------------------------------------------------------------------#
        "spp": True,
        # ---------------------------------------------------------------------#
        #   detection head
        # ---------------------------------------------------------------------#
        "nano": True,
        # ---------------------------------------------------------------------#
        #   radar point semantic segmentation 模型
        # ---------------------------------------------------------------------#
        "is_radar_seg": True,
        "radar_pc": 'pn',
        # ---------------------------------------------------------------------#
        #   radar point semantic segmentation 输入特征数量
        # ---------------------------------------------------------------------#
        "radar_pc_features_num": 5,
        "radar_pc_seg_features": ['x', 'y', 'z', 'comp_velocity', 'rcs'],
        "radar_pc_seg_labels": ['label'],
        "radar_pc_align_num": 512,
        # ---------------------------------------------------------------------#
        #   radar point semantic segmentation 类别数量
        # ---------------------------------------------------------------------#
        "radar_pc_classes": 8,
        "radar_pc_cls_color": {0:'b', 1:'g', 2:'r', 3:'m', 4:'y', 5:'orange', 6:'violet', 7:'peru'},
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": True,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Achelous
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        if self.num_classes_seg <= 21:
            self.colors_seg = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]
            self.colors_seg_line = list(reversed([ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]))
        else:
            hsv_tuples = [(x / self.num_classes_seg, 1., 1.) for x in range(self.num_classes_seg)]
            self.colors_seg = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors_seg = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors_seg))

        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        if self.is_radar_seg:
            self.net = Achelous(num_det=self.num_classes, num_seg=self.num_classes_seg,
                                radar_channels=self.radar_channels,
                                backbone=self.backbone, neck=self.neck, nano_head=self.nano,
                                resolution=self.input_shape[0],
                                pc_seg=self.radar_pc, pc_classes=self.radar_pc_classes,
                                pc_channels=self.radar_pc_features_num, phi=self.phi, spp=self.spp)

        else:
            self.net = Achelous3T(num_det=self.num_classes, num_seg=self.num_classes_seg,
                                  radar_channels=self.radar_channels,
                                  backbone=self.backbone, neck=self.neck, nano_head=self.nano,
                                  resolution=self.input_shape[0], phi=self.phi, spp=self.spp)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, image_id, crop=False, count=False, export_all=False):
        # ---------------------------------------------------#
        #   获得输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)

        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        _, nw, nh = resize_image_seg(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # ------------------------------#
        #   读取雷达特征map
        # ------------------------------#
        radar_path = os.path.join(self.radar_root, image_id + '.npz')
        radar_data = np.load(radar_path)['arr_0']
        radar_data = torch.from_numpy(preprocess_input_radar(radar_data)).type(torch.FloatTensor).unsqueeze(0)

        # -------------------------------- 麻烦的点云读取 ---------------------------------- #
        radar_pc_file = pd.read_csv(os.path.join(self.radar_pc_root, image_id + '.csv'), index_col=0)
        radar_pc_features = radar_pc_file[self.radar_pc_seg_features]
        radar_pc_labels = radar_pc_file[self.radar_pc_seg_labels]

        # --------------------- 投影到相机平面 ------------------------- #
        radar_pc_u = radar_pc_file[['u']]
        radar_pc_v = radar_pc_file[['v']]
        radar_pc_power = radar_pc_file[['rcs']]

        radar_pc_indexes = np.random.choice(radar_pc_features.shape[0], self.radar_pc_align_num, replace=True)

        radar_pc_u = np.asarray(radar_pc_u)
        radar_pc_v = np.asarray(radar_pc_v)
        radar_pc_power = np.asarray(radar_pc_power)
        radar_pc_u = radar_pc_u[radar_pc_indexes]
        radar_pc_v = radar_pc_v[radar_pc_indexes]
        radar_pc_power = radar_pc_power[radar_pc_indexes]

        radar_uv = np.concatenate([radar_pc_u, radar_pc_v], axis=1)

        radar_pc_features = np.asarray(radar_pc_features)
        radar_pc_labels = np.asarray(radar_pc_labels)

        align_radar_pc_features = radar_pc_features[radar_pc_indexes]
        align_radar_pc_labels = radar_pc_labels[radar_pc_indexes]
        align_radar_pc_features = normalize(X=align_radar_pc_features, axis=0)
        align_radar_pc_labels = align_radar_pc_labels

        align_radar_pc_features = torch.from_numpy(np.array(align_radar_pc_features, dtype=np.float32)).type(
            torch.FloatTensor).unsqueeze(0).permute(0, 2, 1).cuda()
        align_radar_pc_labels = torch.from_numpy(np.array(align_radar_pc_labels, dtype=np.int32)). \
            type(torch.LongTensor).cuda()
        # --------------------------------------------------------------------------------- #

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            if self.is_radar_seg:
                outputs, output_seg, output_seg_line, output_seg_pc = self.net(images, radar_data,
                                                                               align_radar_pc_features)
                outputs = decode_outputs(outputs, self.input_shape, 0)

                output_seg_pc = output_seg_pc[0]
                output_seg_pc_cls = torch.argmax(output_seg_pc, dim=1).unsqueeze(1)
                # output_seg_pc_cls = np.array([[self.radar_pc_cls_color[key.item()]] for key in output_seg_pc_cls])

                output_seg_pc_collections = np.concatenate([radar_pc_u, radar_pc_v, radar_pc_power,
                                                            output_seg_pc_cls.cpu().numpy()], axis=1)

                output_seg_pc_collections = np.unique(output_seg_pc_collections, axis=0)

                fig = plt.figure()
                plt.scatter(x=output_seg_pc_collections[:, 0], y=output_seg_pc_collections[:, 1], s=output_seg_pc_collections[:, 2], c=output_seg_pc_collections[:, 3], alpha=0.98)
            else:
                outputs, output_seg, output_seg_line = self.net(images, radar_data)
                outputs = decode_outputs(outputs, self.input_shape, 0)

            output_seg = output_seg[0]
            output_seg_line = output_seg_line[0]

            # -------------------------------------------------------- #
            # ---------------------------------------------------#
            #   语义分割 取出每一个像素点的种类
            # ---------------------------------------------------#
            output_seg = F.softmax(output_seg.permute(1, 2, 0), dim=-1).cpu().numpy()
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            output_seg = output_seg[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            output_seg = cv2.resize(output_seg, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            output_seg = output_seg.argmax(axis=-1)
            output_seg[(output_seg != 0) & (output_seg != 8)] = 0

            # -------------------------------------------------------- #

            # -------------------------------------------------------- #
            # ---------------------------------------------------#
            #   水岸线分割 取出每一个像素点的种类
            # ---------------------------------------------------#
            output_seg_line = F.softmax(output_seg_line.permute(1, 2, 0), dim=-1).cpu().numpy()
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            output_seg_line = output_seg_line[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                         int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            output_seg_line = cv2.resize(output_seg_line, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            output_seg_line = output_seg_line.argmax(axis=-1)
            # -------------------------------------------------------- #

            # ---------------------------------------------------------#
            #   语义分割
            # ---------------------------------------------------------#
            seg_img = np.reshape(np.array(self.colors_seg, np.uint8)[np.reshape(output_seg, [-1])],
                                 [orininal_h, orininal_w, -1])
            seg_line_img = np.reshape(np.array(self.colors_seg_line, np.uint8)[np.reshape(output_seg_line, [-1])],
                                 [orininal_h, orininal_w, -1])
            # ------------------------------------------------#
            #   将新图片转换成Image的形式
            # ------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
            image_line = Image.fromarray(np.uint8(seg_line_img))
            # ------------------------------------------------#
            #   将新图与原图及进行混合
            # ------------------------------------------------#
            image = Image.blend(old_img, image, 0.45)
            # image = Image.blend(image, image_line, 0.3)

            # contrast_enhancer = ImageEnhance.Contrast(image)
            # # 传入调整系数1.2
            # image = contrast_enhancer.enhance(1.1)

            bright_enhancer = ImageEnhance.Brightness(image)
            # 传入调整系数1.2
            image = bright_enhancer.enhance(1.3)

            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            if predicted_class == 'sailor':
                continue
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            # if export_all is False:
            #     print(label, top, left, bottom, right)
            # else:
            #     continue

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        if export_all is False:
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.imshow(image)
            plt.savefig("export_results/" + image_id + ".jpg", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()

        else:
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image)
            plt.savefig("export_results/" + image_id+".jpg", dpi=300, bbox_inches='tight', pad_inches=0)
        return image

    def detect_heatmap(self, image, image_id, heatmap_save_path):
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        # ---------------------------------------------------#
        #   获得输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # ------------------------------#
        #   读取雷达特征map
        # ------------------------------#
        radar_path = os.path.join(self.radar_root, image_id + '.npz')
        radar_data = np.load(radar_path)['arr_0']
        radar_data = torch.from_numpy(radar_data).type(torch.FloatTensor).unsqueeze(0)

        # -------------------------------- 麻烦的点云读取 ---------------------------------- #
        radar_pc_file = pd.read_csv(os.path.join(self.radar_pc_root, image_id + '.csv'), index_col=0)
        radar_pc_features = radar_pc_file[self.radar_pc_seg_features]
        radar_pc_labels = radar_pc_file[self.radar_pc_seg_labels]

        # --------------------- 投影到相机平面 ------------------------- #
        radar_pc_u = radar_pc_file[['u']]
        radar_pc_v = radar_pc_file[['v']]
        radar_pc_power = radar_pc_file[['rcs']]

        radar_pc_indexes = np.random.choice(radar_pc_features.shape[0], self.radar_pc_align_num, replace=True)

        radar_pc_u = np.asarray(radar_pc_u)
        radar_pc_v = np.asarray(radar_pc_v)
        radar_pc_power = np.asarray(radar_pc_power)
        radar_pc_u = radar_pc_u[radar_pc_indexes]
        radar_pc_v = radar_pc_v[radar_pc_indexes]
        radar_pc_power = radar_pc_power[radar_pc_indexes]

        radar_uv = np.concatenate([radar_pc_u, radar_pc_v], axis=1)

        radar_pc_features = np.asarray(radar_pc_features)
        radar_pc_labels = np.asarray(radar_pc_labels)

        align_radar_pc_features = radar_pc_features[radar_pc_indexes]
        align_radar_pc_labels = radar_pc_labels[radar_pc_indexes]
        align_radar_pc_features = normalize(X=align_radar_pc_features, axis=0)
        align_radar_pc_labels = align_radar_pc_labels

        align_radar_pc_features = torch.from_numpy(np.array(align_radar_pc_features, dtype=np.float32)).type(
            torch.FloatTensor).unsqueeze(0).permute(0, 2, 1).cuda()
        align_radar_pc_labels = torch.from_numpy(np.array(align_radar_pc_labels, dtype=np.int32)). \
            type(torch.LongTensor).cuda()
        # --------------------------------------------------------------------------------- #

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            if self.is_radar_seg:
                outputs, output_seg, output_seg_line, output_seg_pc = self.net(images, radar_data,
                                                                               align_radar_pc_features)

            else:
                outputs, output_seg, output_seg_line = self.net(images, radar_data)

        outputs = [output.cpu().numpy() for output in outputs]
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(sub_output, [0, 2, 3, 1])[0]
            score = np.max(sigmoid(sub_output[..., 5:]), -1) * sigmoid(sub_output[..., 4])
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximum(mask, normed_score)

        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200)
        print("Save to the " + heatmap_save_path)
        plt.cla()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # ------------------------------#
        #   读取雷达特征map
        # ------------------------------#
        radar_path = os.path.join(self.radar_root, image_id + '.npz')
        radar_data = np.load(radar_path)['arr_0']
        radar_data = torch.from_numpy(radar_data).type(torch.FloatTensor).unsqueeze(0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs, _ = self.net(images, radar_data)
            outputs = decode_outputs(outputs, self.input_shape)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return