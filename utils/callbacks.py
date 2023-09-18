import os
import re
import torch
import matplotlib
import torch.nn as nn
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler, normalize
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import decode_outputs, non_max_suppression
from .utils_map import get_coco_map, get_map


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        if os.path.exists(self.log_dir):
            pass
        else:
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, radar_path, local_rank,
                 radar_pc_seg_path, is_radar_pc_seg, radar_pc_seg_features, radar_pc_seg_label, radar_pc_num,
                 map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5,
                 letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.map_out_path = map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period
        self.radar_path = radar_path
        self.local_rank = local_rank
        self.radar_pc_seg_path = radar_pc_seg_path
        self.is_radar_pc_seg = is_radar_pc_seg
        self.radar_pc_seg_features = radar_pc_seg_features
        self.radar_pc_seg_label = radar_pc_seg_label
        self.radar_pc_num = radar_pc_num

        self.maps = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_map_txt(self, image_id, image, radar_data, class_names, map_out_path, local_rank):
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

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda(self.local_rank)
                radar_data = radar_data.cuda(self.local_rank)
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#

            # ----------------- 重参数化 -------------------- #
            for child in self.net.children():
                if isinstance(child, nn.Module):
                    print('deploy:', type(child).__name__)
                    child.deploy = True

            for module in self.net.modules():
                if hasattr(module, 'reparameterize'):
                    print('reparameterize:', type(module).__name__)
                    module.reparameterize()
            # ----------------------------------------------- #

            if self.is_radar_pc_seg:
                # -------------------------------- 麻烦的点云读取 ---------------------------------- #
                radar_pc_file = pd.read_csv(os.path.join(self.radar_pc_seg_path, image_id + '.csv'), index_col=0)
                radar_pc_features = radar_pc_file[self.radar_pc_seg_features]
                radar_pc_labels = radar_pc_file[self.radar_pc_seg_label]

                radar_pc_features = np.asarray(radar_pc_features)
                radar_pc_labels = np.asarray(radar_pc_labels)

                radar_pc_indexes = np.random.choice(radar_pc_features.shape[0], self.radar_pc_num, replace=True)

                align_radar_pc_features = radar_pc_features[radar_pc_indexes]
                align_radar_pc_labels = radar_pc_labels[radar_pc_indexes]
                align_radar_pc_features = normalize(X=align_radar_pc_features, axis=0)
                align_radar_pc_labels = align_radar_pc_labels

                align_radar_pc_features = torch.from_numpy(np.array(align_radar_pc_features, dtype=np.float32)).type(
                    torch.FloatTensor).unsqueeze(0).permute(0, 2, 1).cuda(self.local_rank)
                align_radar_pc_labels = torch.from_numpy(np.array(align_radar_pc_labels, dtype=np.int32)).\
                    type(torch.LongTensor).cuda(self.local_rank)
                # --------------------------------------------------------------------------------- #

                outputs = self.net(images, radar_data, align_radar_pc_features)[0]
            else:
                outputs = self.net(images, radar_data)[0]
            outputs = decode_outputs(outputs, self.input_shape, local_rank)
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

        top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes = top_boxes[top_100]
        top_conf = top_conf[top_100]
        top_label = top_label[top_100]

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

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            for annotation_line in tqdm(self.val_lines):
                line = annotation_line.split()

                # ------------------------------#
                #   读取雷达特征map
                # ------------------------------#
                name = os.path.splitext(annotation_line.split('/')[-1].split(' ')[0])[0]

                radar_path = os.path.join(self.radar_path, name + '.npz')
                radar_data = np.load(radar_path)['arr_0']
                radar_data = torch.from_numpy(radar_data).type(torch.cuda.FloatTensor).unsqueeze(0)

                image_id = name
                # ------------------------------#
                #   读取图像并转换成RGB图像
                # ------------------------------#
                image = Image.open(line[0])
                # ------------------------------#
                #   获得预测框
                # ------------------------------#
                gt_boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
                # ------------------------------#
                #   获得预测txt
                # ------------------------------#
                self.get_map_txt(image_id, image, radar_data, self.class_names, self.map_out_path, self.local_rank)

                # ------------------------------#
                #   获得真实框txt
                # ------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")

            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth=2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s' % str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)
