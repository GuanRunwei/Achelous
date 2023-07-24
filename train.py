# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import datetime
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.Achelous import *
from loss.detection_loss import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils_seg.callbacks import EvalCallback as EvalCallback_seg
from utils_seg_line.callbacks import EvalCallback as EvalCallback_seg_line
from utils.dataloader import YoloDataset, yolo_dataset_collate, yolo_dataset_collate_all
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch
from utils_seg.callbacks import LossHistory as LossHistory_seg
from utils_seg_line.callbacks import LossHistory as LossHistory_seg_line
from utils_seg_pc.callbacks import LossHistory as LossHistory_seg_pc
from utils_seg_pc.callbacks import EvalCallback as EvalCallback_seg_pc
import argparse


if __name__ == "__main__":
    # =========== 参数解析实例 =========== #
    parser = argparse.ArgumentParser()

    # 添加参数解析
    parser.add_argument("--cuda", type=str, default="True")
    parser.add_argument("--ddp", type=str, default="False")
    parser.add_argument("--fp16", type=str, default="True")
    parser.add_argument("--is_pc", help="use pc seg", type=str, default="False")
    parser.add_argument("--backbone", type=str, default='en')
    parser.add_argument("--neck", type=str, default='gdf')
    parser.add_argument("--nd", type=str, default="True")
    parser.add_argument("--phi", type=str, default='S0')
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr_init", type=float, default=0.03)
    parser.add_argument("--lr_decay", type=str, default=5e-4)
    parser.add_argument("--opt", type=str, default='sgd')
    parser.add_argument("--pc_num", type=int, default=512)
    parser.add_argument("--nw", type=int, default=4)
    parser.add_argument("--dice", type=str, default="True")
    parser.add_argument("--focal", type=str, default="True")
    parser.add_argument("--pc_model", type=str, default='pn')
    parser.add_argument("--spp", type=str, default='True')
    parser.add_argument("--local_rank", default=-1, type=int, help='node rank for distributed training')

    args = parser.parse_args()

    # ==================================== #

    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    Cuda = True if args.cuda == 'True' else False

    # ---------------------------------------------------------------------#
    distributed = True if args.ddp == 'True' else False
    # ---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = True if args.fp16 == 'True' else False
    # ---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    # ---------------------------------------------------------------------#
    classes_path = 'model_data/waterscenes_benchmark.txt'
    model_path = ''

    # ------------------------------------------------------#
    #   backbone (4 options): ef (EfficientFormer), en (EdgeNeXt), ev (EdgeViT), mv (MobileViT)
    # ------------------------------------------------------#
    backbone = args.backbone

    # ------------------------------------------------------#
    #   neck (2 options): gdf (Ghost-Dual-FPN), cdf (CSP-Dual-FPN)
    # ------------------------------------------------------#
    neck = args.neck

    # ------------------------------------------------------#
    #   spp: True->SPP, False->SPPF
    # ------------------------------------------------------#
    spp = True if args.spp == 'True' else False

    # ------------------------------------------------------#
    #   detection head (2 options): normal -> False, lightweight -> True
    # ------------------------------------------------------#
    lightweight = True if args.nd == 'True' else False

    # ------------------------------------------------------#
    #   input_shape     all models support 320*320, all models except mobilevit support 416*416
    # ------------------------------------------------------#
    input_shape = [args.resolution, args.resolution]
    # ------------------------------------------------------#
    #   The size of model, three options: S0, S1, S2
    # ------------------------------------------------------#
    phi = args.phi
    # ------------------------------------------------------#

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，Freeze_Train = True，此时仅仅进行冻结训练。
    #
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 300，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 300，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）从0开始训练：
    #       Init_Epoch = 0，UnFreeze_Epoch >= 300，Unfreeze_batch_size >= 16，Freeze_Train = False（不冻结训练）
    #       其中：UnFreeze_Epoch尽量不小于300。optimizer_type = 'sgd'，Init_lr = 1e-2，mosaic = True。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    # ------------------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 0
    Freeze_batch_size = 32
    # ------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #                           SGD需要更长的时间收敛，因此设置较大的UnFreeze_Epoch
    #                           Adam可以使用相对较小的UnFreeze_Epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = args.bs
    # ------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------------------#
    Freeze_Train = False

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = args.lr_init
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = args.opt
    momentum = 0.937
    weight_decay = 5e-4
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    # ------------------------------------------------------------------#
    lr_decay_type = args.lr_decay
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = 5
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = 'logs'
    # ------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #                   安装pycocotools库后，评估体验更佳。
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = 5
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------------------#
    num_workers = args.nw

    # ========================================  Dataset Path =========================================== #
    # ----------------------------------------------------#
    # 雷达feature map路径
    # ----------------------------------------------------#
    radar_file_path = "E:/Big_Datasets/water_surface/benchmark_new/WaterScenes_new/radar/VOCradar320"

    # ----------------------------------------------------#
    #   获得目标检测图片路径和标签
    # ----------------------------------------------------#
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    # ----------------------------------------------------#
    #   jpg图像路径
    # ----------------------------------------------------#
    jpg_path = "E:/Big_Datasets/water_surface/benchmark_new/WaterScenes_new/images/images"

    # ------------------------------------------------------------------#
    # 语义分割数据集路径
    # ------------------------------------------------------------------#
    se_seg_path = "E:/Big_Datasets/water_surface/benchmark_new/WaterScenes_new/semantic/SegmentationClass/SegmentationClass"

    # ------------------------------------------------------------------#
    # 水岸线分割数据集路径
    # ------------------------------------------------------------------#
    wl_seg_path = "E:/Big_Datasets/water_surface/benchmark_new/WaterScenes_new/waterline/SegmentationClassPNG/SegmentationClassPNG"

    # ------------------------------------------------------------------#
    # 是否需要训练毫米波雷达点云分割
    # ------------------------------------------------------------------#
    is_radar_pc_seg = True if args.is_pc == 'True' else False

    pc_seg_model = args.pc_model
    # ------------------------------------------------------------------#
    # 每个batch的点云数量
    # ------------------------------------------------------------------#
    radar_pc_num = args.pc_num

    # ------------------------------------------------------------------#
    # 毫米波雷达点云分割路径
    # ------------------------------------------------------------------#
    radar_pc_seg_path = "E:/Big_Datasets/water_surface/benchmark_new/WaterScenes_new/radar/radar_0220/radar"

    # ------------------------------------------------------------------#
    # 毫米波雷达点云分割属性, 其中label表示雷达目标的语义标签
    # ------------------------------------------------------------------#
    radar_pc_seg_features = ['x', 'y', 'z', 'comp_velocity', 'rcs']
    radar_pc_seg_label = ['label']

    radar_pc_classes = 8
    radar_pc_channels = len(radar_pc_seg_features)
    # ================================================================================================== #

    # ============================ segmentation hyperparameters ============================= #
    # -----------------------------------------------------#
    #   num_classes     训练自己的数据集必须要修改的
    #                   自己需要的分类个数+1，如2+1
    # -----------------------------------------------------#
    num_classes_seg = 9

    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    dice_loss = True if args.dice == 'True' else False

    # ------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    # ------------------------------------------------------------------#
    focal_loss = True if args.focal == 'True' else False

    # ------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes_seg], np.float32)
    cls_weights_wl = np.ones([2], np.float32)

    # ------------------------------------------------------------------#
    #   save_dir_seg        分割权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir_seg = 'logs_seg'
    save_dir_seg_wl = 'logs_seg_line'
    save_dir_seg_pc = 'logs_seg_pc'

    # ======================================================================================= #

    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    # ------------------------------------------------------#
    #   创建模型
    # ------------------------------------------------------#
    if is_radar_pc_seg:
        model = Achelous(resolution=input_shape[0], num_det=num_classes, num_seg=num_classes_seg, phi=phi,
                         backbone=backbone, neck=neck, nano_head=lightweight, pc_seg=pc_seg_model,
                         pc_channels=radar_pc_channels, pc_classes=radar_pc_classes, spp=spp).cuda(local_rank)
    else:
        model = Achelous3T(resolution=input_shape[0], num_det=num_classes, num_seg=num_classes_seg, phi=phi,
                           backbone=backbone, neck=neck, spp=spp,
                           nano_head=lightweight).cuda(local_rank)
    weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # ----------------------#
    #   获得损失函数
    # ----------------------#
    yolo_loss = YOLOLoss(num_classes, fp16)
    # ----------------------#
    #   记录Loss
    # ----------------------#
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    log_dir_seg = os.path.join(save_dir_seg, "loss_" + str(time_str))
    log_dir_seg_wl = os.path.join(save_dir_seg_wl, "loss_" + str(time_str))
    log_dir_seg_pc = os.path.join(save_dir_seg_pc, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    loss_history_seg = LossHistory_seg(log_dir_seg, model, input_shape=input_shape)
    loss_history_seg_wl = LossHistory_seg_line(log_dir_seg_wl, model, input_shape=input_shape)
    loss_history_seg_pc = LossHistory_seg_pc(log_dir_seg_pc, model, input_shape=input_shape)

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # ----------------------------#
    #   多卡同步Bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   多卡平行运行
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.to(device)

    # ----------------------------#
    #   权值平滑
    # ----------------------------#
    ema = ModelEMA(model_train)

    # ---------------------------#
    #   读取检测数据集对应的txt
    # ---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        backbone=backbone, neck=neck, lightweight_head=lightweight, is_radar_pc_seg=is_radar_pc_seg,
        fp16=fp16, phi=phi, is_focal=focal_loss, is_dice=dice_loss, use_spp=spp,
        classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
        Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
        Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
        lr_decay_type=lr_decay_type, save_period=save_period, save_dir=save_dir, num_workers=num_workers,
        num_train=num_train, num_val=num_val
    )
    # ---------------------------------------------------------#
    #   总训练世代指的是遍历全部数据的总次数
    #   总训练步长指的是梯度下降的总次数
    #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
    #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
    # ----------------------------------------------------------#
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
            num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
            total_step, wanted_step, wanted_epoch))

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.backbone.parameters():
                param.requires_grad = False

        # -------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        if ema:
            ema.updates = epoch_step * Init_Epoch

        # ---------------------------------------#
        #   构建数据集加载器。
        # ---------------------------------------#
        if is_radar_pc_seg:
            train_dataset = YoloDataset(annotation_lines=train_lines, input_shape=input_shape, num_classes=num_classes,
                                        epoch_length=UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0,
                                        train=False, special_aug_ratio=0, radar_root=radar_file_path,
                                        num_classes_seg=num_classes_seg, seg_dataset_path=se_seg_path,
                                        water_seg_dataset_path=wl_seg_path, radar_pc_seg_dataset_path=radar_pc_seg_path,
                                        is_radar_pc_seg=is_radar_pc_seg, radar_pc_seg_features=radar_pc_seg_features,
                                        radar_pc_seg_label=radar_pc_seg_label, radar_pc_num=radar_pc_num)

            val_dataset = YoloDataset(annotation_lines=val_lines, input_shape=input_shape, num_classes=num_classes,
                                      epoch_length=UnFreeze_Epoch, \
                                      mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False,
                                      special_aug_ratio=0, radar_root=radar_file_path,
                                      num_classes_seg=num_classes_seg, seg_dataset_path=se_seg_path,
                                      water_seg_dataset_path=wl_seg_path, radar_pc_seg_dataset_path=radar_pc_seg_path,
                                      is_radar_pc_seg=is_radar_pc_seg, radar_pc_seg_features=radar_pc_seg_features,
                                      radar_pc_seg_label=radar_pc_seg_label, radar_pc_num=radar_pc_num)

        else:
            train_dataset = YoloDataset(annotation_lines=train_lines, input_shape=input_shape, num_classes=num_classes,
                                        epoch_length=UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0,
                                        train=False, special_aug_ratio=0, radar_root=radar_file_path,
                                        num_classes_seg=num_classes_seg, seg_dataset_path=se_seg_path,
                                        water_seg_dataset_path=wl_seg_path, radar_pc_seg_dataset_path=radar_pc_seg_path,
                                        radar_pc_seg_features=[])

            val_dataset = YoloDataset(annotation_lines=val_lines, input_shape=input_shape, num_classes=num_classes,
                                      epoch_length=UnFreeze_Epoch, \
                                      mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False,
                                      special_aug_ratio=0, radar_root=radar_file_path,
                                      num_classes_seg=num_classes_seg, seg_dataset_path=se_seg_path,
                                      water_seg_dataset_path=wl_seg_path, radar_pc_seg_dataset_path='',
                                      radar_pc_seg_features=[])

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        # ---------------------------------------#
        #   构建Dataloader。
        # ---------------------------------------#
        if is_radar_pc_seg:
            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate_all, sampler=train_sampler)
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate_all, sampler=val_sampler)

        else:
            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                     eval_flag=eval_flag, period=eval_period, radar_path=radar_file_path,
                                     radar_pc_seg_path=radar_pc_seg_path, local_rank=local_rank, is_radar_pc_seg=is_radar_pc_seg,
                                     radar_pc_seg_features=radar_pc_seg_features, radar_pc_seg_label=radar_pc_seg_label,
                                     radar_pc_num=radar_pc_num)
        eval_callback_seg = EvalCallback_seg(model, input_shape, num_classes_seg, val_lines, se_seg_path,
                                             log_dir_seg, Cuda, eval_flag=eval_flag, period=eval_period,
                                             radar_path=radar_file_path, radar_pc_seg_path=radar_pc_seg_path,
                                             local_rank=local_rank, jpg_path=jpg_path, is_radar_pc_seg=is_radar_pc_seg,
                                             radar_pc_seg_features=radar_pc_seg_features,
                                             radar_pc_seg_label=radar_pc_seg_label, radar_pc_num=radar_pc_num)
        eval_callback_seg_wl = EvalCallback_seg_line(model, input_shape, 2, val_lines, wl_seg_path,
                                             log_dir_seg_wl, Cuda, eval_flag=eval_flag, period=eval_period,
                                             radar_path=radar_file_path, local_rank=local_rank,
                                             radar_pc_seg_path=radar_pc_seg_path, jpg_path=jpg_path, is_radar_pc_seg=is_radar_pc_seg,
                                                     radar_pc_seg_features=radar_pc_seg_features,
                                                     radar_pc_seg_label=radar_pc_seg_label, radar_pc_num=radar_pc_num)
        eval_callback_seg_pc = EvalCallback_seg_pc(model, input_shape, 2, val_lines, wl_seg_path,
                                                     log_dir_seg_wl, Cuda, eval_flag=eval_flag, period=eval_period,
                                                     radar_path=radar_file_path, local_rank=local_rank,
                                                     radar_pc_seg_path=radar_pc_seg_path, jpg_path=jpg_path,
                                                     is_radar_pc_seg=is_radar_pc_seg,
                                                     radar_pc_seg_features=radar_pc_seg_features,
                                                     radar_pc_seg_label=radar_pc_seg_label, radar_pc_num=radar_pc_num)

        # ---------------------------------------#
        #   开始模型训练123
        # ---------------------------------------#
        train_index = 0
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # ---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            # ---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # -------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                # -------------------------------------------------------------------#
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                # ---------------------------------------#
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                if ema:
                    ema.updates = epoch_step * epoch

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch
            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, loss_history_seg, loss_history_seg_wl,
                          loss_history_seg_pc, eval_callback, eval_callback_seg, eval_callback_seg_wl,
                          eval_callback_seg_pc, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, dice_loss, focal_loss, cls_weights,
                          cls_weights_wl, num_classes_seg, local_rank, is_radar_pc_seg)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
            loss_history_seg.writer.close()
            loss_history_seg_wl.writer.close()
