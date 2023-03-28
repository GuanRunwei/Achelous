import os

import torch
from tqdm import tqdm

from utils.utils import get_lr

from loss.segmentation_loss import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)

from utils_seg.utils import get_lr
from utils_seg.utils_metrics import f_score

from loss.multitaskloss import HUncertainty
from loss.mgda import MGDA
from loss.pc_seg_loss import NllLoss
import torch.nn.functional as F


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, loss_history_seg, loss_history_seg_wl, loss_history_seg_pc, eval_callback, eval_callback_seg, eval_callback_seg_w, eval_callback_seg_pc, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, dice_loss, focal_loss, cls_weights, cls_weights_wl, num_class_seg, local_rank=0, is_radar_pc_seg=False):
    total_loss_det = 0
    total_loss_seg = 0
    total_loss_seg_w = 0
    total_loss_seg_pc = 0
    total_f_score = 0
    total_f_score_w = 0

    val_loss_det = 0
    val_loss_seg = 0
    val_loss_seg_w = 0
    val_loss_seg_pc = 0
    val_f_score = 0
    val_f_score_w = 0

    total_loss = 0
    val_total_loss = 0

    if local_rank >= 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        if is_radar_pc_seg:
            images, targets, radars, pngs, pngs_w, seg_labels, seg_w_labels, radar_pc_features, radar_pc_labels = \
                batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8]

        else:
            images, targets, radars, pngs, pngs_w, seg_labels, seg_w_labels = batch[0], batch[1], batch[2], batch[3], \
                                                                              batch[4], batch[5], batch[6]

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            weights_wl = torch.from_numpy(cls_weights_wl)

            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                radars = radars.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                pngs_w = pngs_w.cuda(local_rank)
                seg_labels = seg_labels.cuda(local_rank)
                seg_w_labels = seg_w_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
                weights_wl = weights_wl.cuda(local_rank)
                if is_radar_pc_seg:
                    radar_pc_features = radar_pc_features.cuda(local_rank)
                    radar_pc_labels = radar_pc_labels.cuda(local_rank)

        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            if is_radar_pc_seg:
                outputs, outputs_seg, outputs_seg_w, outputs_seg_pc = model_train(images, radars, radar_pc_features)
                nll_loss = NllLoss()
                loss_pc_seg = nll_loss(F.log_softmax(outputs_seg_pc).permute(0, 2, 1).squeeze(-1), radar_pc_labels)
            else:
                outputs, outputs_seg, outputs_seg_w = model_train(images, radars)

            # ----------------------------------- 计算损失 ------------------------------------ #
            if focal_loss:
                loss_seg = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
                loss_seg_w = Focal_Loss(outputs_seg_w, pngs_w, weights_wl, num_classes=2)
            else:
                loss_seg = CE_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
                loss_seg_w = CE_Loss(outputs_seg_w, pngs_w, weights_wl, num_classes=2)

            if dice_loss:
                main_dice = Dice_loss(outputs_seg, seg_labels)
                main_dice_w = Dice_loss(outputs_seg_w, seg_w_labels)
                loss_seg = loss_seg + main_dice
                logg_seg_w = loss_seg_w + main_dice_w

            loss_det = yolo_loss(outputs, targets)

            mtl = HUncertainty(task_num=3)
            mgda = MGDA()

            if is_radar_pc_seg:
                total_loss = mtl(loss_seg, logg_seg_w, loss_det) + loss_pc_seg
                # total_loss = mgda.backward([loss_seg, logg_seg_w, loss_det, loss_pc_seg])
            else:
                total_loss = mtl(loss_seg, logg_seg_w, loss_det)
                # total_loss = mgda.backward([loss_seg, logg_seg_w, loss_det])
            # -------------------------------------------------------------------------------- #

            with torch.no_grad():
                train_f_score = f_score(outputs_seg, seg_labels)
                train_f_score_w = f_score(outputs_seg_w, seg_w_labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            total_loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                if is_radar_pc_seg:
                    outputs, outputs_seg, outputs_seg_w, outputs_seg_pc = model_train(images, radars, radar_pc_features)
                    nll_loss = NllLoss()
                    loss_pc_seg = nll_loss(F.log_softmax(outputs_seg_pc).permute(0, 2, 1), radar_pc_labels.squeeze(-1))
                else:
                    outputs, outputs_seg, outputs_seg_w = model_train(images, radars)

                # ----------------------------------- 计算损失 ------------------------------------ #
                if focal_loss:
                    loss_seg = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
                    loss_seg_w = Focal_Loss(outputs_seg_w, pngs_w, weights_wl, num_classes=2)
                else:
                    loss_seg = CE_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
                    loss_seg_w = CE_Loss(outputs_seg_w, pngs_w, weights_wl, num_classes=2)

                if dice_loss:
                    main_dice = Dice_loss(outputs_seg, seg_labels)
                    main_dice_w = Dice_loss(outputs_seg_w, seg_w_labels)
                    loss_seg = loss_seg + main_dice
                    logg_seg_w = loss_seg_w + main_dice_w

                loss_det = yolo_loss(outputs, targets)

                mtl = HUncertainty(task_num=3)
                mgda = MGDA()

                if is_radar_pc_seg:
                    total_loss = mtl(loss_seg, logg_seg_w, loss_det) + loss_pc_seg
                    # total_loss = mgda.backward([loss_seg, logg_seg_w, loss_det, loss_pc_seg])
                else:
                    total_loss = mtl(loss_seg, logg_seg_w, loss_det)
                    # total_loss = mgda.backward([loss_seg, logg_seg_w, loss_det])
                # -------------------------------------------------------------------------------- #

                with torch.no_grad():
                    train_f_score = f_score(outputs_seg, seg_labels)
                    train_f_score_w = f_score(outputs_seg_w, seg_w_labels)

            # ----------------------#
            #   back-propagation
            # ----------------------#
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        total_loss_det += loss_det.item()
        total_loss_seg += loss_seg.item()
        total_loss_seg_w += loss_seg_w.item()
        if is_radar_pc_seg:
            total_loss_seg_pc += loss_pc_seg.item()
        total_loss += total_loss_det + total_loss_seg + total_loss_seg_w + total_loss_seg_pc
        total_f_score += train_f_score.item()
        total_f_score_w += train_f_score_w.item()

        if local_rank >= 0:
            if is_radar_pc_seg:
                pbar.set_postfix(**{'detection loss': total_loss_det / (iteration + 1),
                                'se seg loss': total_loss_seg / (iteration + 1),
                                'wl seg loss': total_loss_seg_w / (iteration + 1),
                                'pc seg loss': total_loss_seg_pc / (iteration + 1),
                                'total loss': total_loss / (iteration + 1),
                                'f score se': total_f_score / (iteration + 1),
                                'f score wl': total_f_score_w / (iteration + 1),
                                'lr': get_lr(optimizer)})
            else:
                pbar.set_postfix(**{'detection loss': total_loss_det / (iteration + 1),
                                    'se seg loss': total_loss_seg / (iteration + 1),
                                    'wl seg loss': total_loss_seg_w / (iteration + 1),
                                    'total loss': total_loss / (iteration + 1),
                                    'f score se': total_f_score / (iteration + 1),
                                    'f score wl': total_f_score_w / (iteration + 1),
                                    'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0 or local_rank == 1:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        if is_radar_pc_seg:
            images, targets, radars, pngs, pngs_w, seg_labels, seg_w_labels, radar_pc_features, radar_pc_labels = \
                batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8]

        else:
            images, targets, radars, pngs, pngs_w, seg_labels, seg_w_labels = batch[0], batch[1], batch[2], batch[3], \
                                                                              batch[4], batch[5], batch[6]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                radars = radars.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                pngs_w = pngs_w.cuda(local_rank)
                seg_labels = seg_labels.cuda(local_rank)
                seg_w_labels = seg_w_labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
                if is_radar_pc_seg:
                    radar_pc_features = radar_pc_features.cuda(local_rank)
                    radar_pc_labels = radar_pc_labels.cuda(local_rank)
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            if is_radar_pc_seg:
                outputs, outputs_seg, outputs_seg_w, outputs_seg_pc = model_train(images, radars, radar_pc_features)
                nll_loss = NllLoss()
                loss_pc_seg = nll_loss(F.log_softmax(outputs_seg_pc).permute(0, 2, 1), radar_pc_labels.squeeze(-1))
            else:
                outputs, outputs_seg, outputs_seg_w = model_train(images, radars)

            if focal_loss:
                loss_seg = Focal_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
                loss_seg_w = Focal_Loss(outputs_seg_w, pngs_w, weights_wl, num_classes=2)
            else:
                loss_seg = CE_Loss(outputs_seg, pngs, weights, num_classes=num_class_seg)
                loss_seg_w = CE_Loss(outputs_seg_w, pngs_w, weights_wl, num_classes=2)

            if dice_loss:
                main_dice = Dice_loss(outputs_seg, seg_labels)
                main_dice_w = Dice_loss(outputs_seg_w, seg_w_labels)
                loss_seg = loss_seg + main_dice
                loss_seg_w = loss_seg_w + main_dice_w

            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs_seg, seg_labels)
            _f_score_w = f_score(outputs_seg_w, seg_w_labels)

            # ----------------------#
            #   计算损失
            # ----------------------#
            loss_value = yolo_loss(outputs, targets)
            loss_value_seg = loss_seg
            loss_value_seg_w = loss_seg_w
            if is_radar_pc_seg:
                loss_value_seg_pc = loss_pc_seg
            val_f_score += _f_score.item()
            val_f_score_w += _f_score_w.item()

        val_loss_det += loss_value.item()
        val_loss_seg += loss_value_seg.item()
        val_loss_seg_w += loss_value_seg_w.item()
        if is_radar_pc_seg:
            val_loss_seg_pc += loss_value_seg_pc.item()
        val_total_loss = val_loss_det + val_loss_seg + val_loss_seg_w + val_loss_seg_pc

        if local_rank >= 0:
            if is_radar_pc_seg:
                pbar.set_postfix(**{'detection val_loss': val_loss_det / (iteration + 1),
                                    'se seg val_loss': val_loss_seg / (iteration + 1),
                                    'wl seg val_loss': val_loss_seg_w / (iteration + 1),
                                    'pc seg val_loss': val_loss_seg_pc / (iteration + 1),
                                    'val loss': val_total_loss / (iteration + 1),
                                    'f_score se': val_f_score / (iteration + 1),
                                    'f_score wl': val_f_score_w / (iteration + 1),
                                    })
            else:
                pbar.set_postfix(**{'detection val_loss': val_loss_det / (iteration + 1),
                                    'se seg val_loss': val_loss_seg / (iteration + 1),
                                    'wl seg val_loss': val_loss_seg_w / (iteration + 1),
                                    'val loss': val_total_loss / (iteration + 1),
                                    'f_score se': val_f_score / (iteration + 1),
                                    'f_score wl': val_f_score_w / (iteration + 1),
                                    })
            pbar.update(1)

    if local_rank >= 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss_det / epoch_step, val_loss_det / epoch_step_val)
        loss_history_seg.append_loss(epoch + 1, total_loss_seg / epoch_step, val_loss_seg / epoch_step_val)
        loss_history_seg_wl.append_loss(epoch + 1, total_loss_seg_w / epoch_step, val_loss_seg_w / epoch_step_val)
        loss_history_seg_pc.append_loss(epoch + 1, total_loss_seg_pc / epoch_step, val_loss_seg_pc / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        eval_callback_seg.on_epoch_end(epoch + 1, model_train_eval)
        eval_callback_seg_w.on_epoch_end(epoch + 1, model_train_eval)
        eval_callback_seg_pc.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        if is_radar_pc_seg:
            print(
                'Total Loss: %.3f || Val Loss Det: %.3f  || Val Loss Seg: %.3f || Val Loss Seg L: %.3f || Val Loss Seg PC: %.3f' % (
                (total_loss / epoch_step,
                 val_loss_det / epoch_step_val,
                 val_loss_seg / epoch_step_val,
                 val_loss_seg_w / epoch_step_val,
                 val_loss_seg_pc / epoch_step_val)))
        else:
            print(
                'Total Loss: %.3f || Val Loss Det: %.3f  || Val Loss Seg: %.3f || Val Loss Seg L: %.3f' % (
                    (total_loss / epoch_step,
                     val_loss_det / epoch_step_val,
                     val_loss_seg / epoch_step_val,
                     val_loss_seg_w / epoch_step_val,
                     )))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if is_radar_pc_seg:
            if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
                torch.save(save_state_dict, os.path.join(save_dir,
                                                         "ep%03d-loss%.3f-det_val_loss%.3f-seg_val_loss%.3f-seg_wl_val_loss%.3f-seg_pc_val_loss%.3f.pth" % (
                                                             epoch + 1, val_total_loss / epoch_step,
                                                             val_loss_det / epoch_step_val,
                                                             val_loss_seg / epoch_step_val,
                                                             val_loss_seg_w / epoch_step_val,
                                                             val_loss_seg_pc / epoch_step_val)))

            if len(loss_history.val_loss) <= 1 or (val_total_loss / epoch_step_val) <= min(loss_history.val_loss) + min(
                    loss_history_seg.val_loss):
                print('Save best model to best_epoch_weights.pth')
                torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        else:
            if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
                torch.save(save_state_dict, os.path.join(save_dir,
                                                         "ep%03d-loss%.3f-det_val_loss%.3f-seg_val_loss%.3f-seg_wl_val_loss%.3f.pth" % (
                                                             epoch + 1, val_total_loss / epoch_step,
                                                             val_loss_det / epoch_step_val,
                                                             val_loss_seg / epoch_step_val,
                                                             val_loss_seg_w / epoch_step_val)))

            if len(loss_history.val_loss) <= 1 or (val_total_loss / epoch_step_val) <= min(loss_history.val_loss) + min(
                    loss_history_seg.val_loss):
                print('Save best model to best_epoch_weights.pth')
                torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))