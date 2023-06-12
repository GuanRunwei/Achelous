import os
import re
import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
import pandas as pd
import cv2
import shutil
import numpy as np
from sklearn.preprocessing import MinMaxScaler, normalize
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils_seg.utils import cvtColor, preprocess_input, resize_image
from sklearn.metrics import confusion_matrix
from utils_seg_pc.utils_metrics import mean_iou, get_transform_label_preds


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
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, radar_path, jpg_path,
                 local_rank, radar_pc_seg_path, is_radar_pc_seg, radar_pc_seg_features, radar_pc_seg_label, radar_pc_num,
                 miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = image_ids
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag
        self.period = period
        self.radar_path = radar_path
        self.jpg_path = jpg_path
        self.local_rank = local_rank
        self.radar_pc_seg_path = radar_pc_seg_path
        self.is_radar_pc_seg = is_radar_pc_seg
        self.radar_pc_seg_features = radar_pc_seg_features
        self.radar_pc_seg_label = radar_pc_seg_label
        self.radar_pc_num = radar_pc_num

        self.image_ids = [os.path.splitext(image_id.split('/')[-1].split(' ')[0])[0]for image_id in image_ids]

        self.mious = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_pc_seg_results(self, image, radar_data, image_id):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda(self.local_rank)
                radar_data = radar_data.cuda(self.local_rank)

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#

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
            align_radar_pc_labels = torch.from_numpy(np.array(align_radar_pc_labels, dtype=np.int32)). \
                    type(torch.LongTensor).cuda(self.local_rank)
            # --------------------------------------------------------------------------------- #
            pr = self.net(images, radar_data, align_radar_pc_features)[3][0]
            # ---------------------------------------------------#

        labels, preds = get_transform_label_preds(pr, align_radar_pc_labels)

        return labels, preds

    def on_epoch_end(self, epoch, model_eval):
        y_trues = []
        y_preds = []

        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir = self.dataset_path + '/'
            pred_dir = os.path.join(self.miou_out_path, 'pc-segmentation-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                # ------------------------------#
                #   读取雷达特征map
                # ------------------------------#
                radar_path = os.path.join(self.radar_path, image_id + '.npz')
                radar_data = np.load(radar_path)['arr_0']
                radar_data = torch.from_numpy(radar_data).type(torch.cuda.FloatTensor).unsqueeze(0)

                # -------------------------------#
                #   从文件中读取图像
                # -------------------------------#
                image_path = os.path.join(self.jpg_path, image_id + ".jpg")
                image = Image.open(image_path)
                # ------------------------------#
                #   获得预测txt
                # ------------------------------#
                gts, preds = self.get_pc_seg_results(image, radar_data, image_id)
                y_trues.extend(gts)
                y_preds.extend(preds)

            cm_pc = confusion_matrix(y_trues, y_preds)
            mious, miou = mean_iou(cm_pc)
            for i, item, in enumerate(mious):
                print("Class ", i, " mIoU:", item)

            print("total mIoU:", miou)
            self.mious.append(miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(miou))
                f.write("\n")

            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='PC Seg Miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get pc seg miou done.")
            shutil.rmtree(self.miou_out_path)
