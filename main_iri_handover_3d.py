#from utils import iri_handover_3d as datasets
from utils import mediapipe_handover_3d as datasets
from model import MixAttention
from utils.opt import Options
from utils import util
from utils import log
from utils import plots
from utils.sampler import Sampler
from torch.utils.data import WeightedRandomSampler
from torch.nn.functional import one_hot

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
#import h5py
import torch.optim as optim

#from utils.tb_visualizer import TBVisualizer

from torch.utils.tensorboard import SummaryWriter

import utils.data_utils as data_utils

from utils.data_utils import iri_discretize_pose, iri_undiscretize_pose, iri_undiscretize_pose_prob



def main(opt):
    # Define parameters
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')

    input_n = opt.input_n
    output_n = opt.output_n

    in_features = opt.in_features  # 48
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    n_bins = opt.n_bins
    num_heads = opt.num_heads
    # goal_condition = opt.goal_condition
    goal_features = opt.goal_features
    part_condition = opt.part_condition
    obstacles_condition = opt.obstacles_condition
    fusion_model = opt.fusion_model
    device = opt.device

    # create tensorboardX visualizer
    writer = SummaryWriter(opt.ckpt)

    net_pred = MixAttention.MixAttention(output_n=output_n, in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                          num_stage=opt.num_stage, dct_n=opt.dct_n, num_heads=num_heads,
                                         goal_features=goal_features, part_condition=part_condition,
                                         obstacle_condition=obstacles_condition, fusion_model=fusion_model,
                                         phase=opt.phase, intention=opt.intention).to(device)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        print(opt.ckpt)
        print(opt.exp)
        # opt.ckpt = "main_iri_handover_3d_in50_out25_ks10_dctn20_heads_4_goalfeats_3_part_False_fusion_1"
        # model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        model_path_len = './ckpt_best.pth.tar'
        # model_path_len = "/home/jlaplaza/PycharmProjects/HisRepItself/checkpoint/main_iri_handover_3d_in50_out25_ks10_dctn20_heads_4_goalfeats_3_part_False_fusion_1/ckpt_best.pth.tar"
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    print('>>> loading datasets')

    if not opt.is_eval:
        # dataset = datasets.Datasets(opt, split=0)
        # actions = ["walking", "eating", "smoking", "discussion", "directions",
        #            "greeting", "phoning", "posing", "purchases", "sitting",
        #            "sittingdown", "takingphoto", "waiting", "walkingdog",
        #            "walkingtogether"]
        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        # sampler = Sampler(dataset.intention_classes(), class_per_batch=4, batch_size=32)
        weights = torch.empty(len(dataset))
        for i, sample in enumerate(dataset):
            if sample['intention_goal'] == 0:
                weights[i] = 1 / (862./1033)

            elif sample['intention_goal'] == 1:
                weights[i] = 1 / (38./1033)

            elif sample['intention_goal'] == 2:
                weights[i] = 1 / (113./1033)

            elif sample['intention_goal'] == 3:
                weights[i] = 1 / (20./1033)

            elif sample['intention_goal'] == 4:
                weights[i] = 1 / (20./1033)

        sampler = WeightedRandomSampler(weights=weights, num_samples=20, replacement=False)
        # data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        # data_loader = DataLoader(dataset, num_workers=0, pin_memory=True, batch_sampler=sampler)
        data_loader = DataLoader(dataset, num_workers=4, pin_memory=True, sampler=sampler, batch_size=opt.batch_size)
        valid_dataset = datasets.Datasets(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_sampler = Sampler(valid_dataset.intention_classes(), class_per_batch=5, batch_size=4)
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        # valid_loader = DataLoader(valid_dataset, num_workers=0, pin_memory=True, batch_sampler=valid_sampler)

    test_dataset = datasets.Datasets(opt, split=1)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, writer=writer)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_ang_iri']))
    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            # if epo % 100 == 0:
            #    lr_now = opt.lr_now
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt, writer=writer)
            #print(f"train | skeleton error: {ret_train['m_xyz_iri']} m, right hand error: {ret_train['m_right_hand']} m, "
            #      f"phase acc: {ret_train['m_phase_acc']}, intention acc: {ret_train['m_intention_acc']}")
            # print('train error: {:.3f}, right hand error: {:.3f}'.format(ret_train['m_xyz_iri'], ret_train['m_right_hand']))
            writer.add_scalar("avg_train_loss", ret_train['m_xyz_iri'], epo)
            writer.add_scalar("avg_train_loss_hand", ret_train['m_right_hand'], epo)
            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt,
                                  epo=epo, writer=writer)
            #print(f"validation | skeleton error: {ret_valid['m_xyz_iri']} m, right hand error: {ret_valid['m_right_hand']} m, "
            #      f"phase acc: {ret_valid['m_phase_acc']}, intention acc: {ret_valid['m_intention_acc']}")

            # print('validation error: {:.3f}, right hand error: {:.3f}'.format(ret_valid['m_xyz_iri'], ret_valid['m_right_hand']))
            writer.add_scalar("avg_val_loss", ret_valid['m_xyz_iri'], epo)
            writer.add_scalar("avg_val_loss_hand", ret_valid['m_right_hand'], epo)

            ret_test = run_model(net_pred, is_train=32, data_loader=test_loader, opt=opt,
                                 epo=epo, writer=writer)
            #print(f"test | skeleton error: {ret_test['#25'][0]} m, right hand error: {ret_test['#25'][1]} m, "
            #    f"phase acc: {ret_test['#25'][2]}, intention acc: {ret_test['#25'][3]}")
            # print('testing error: {:.3f}'.format(ret_test['#1']))

            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            # log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid['m_xyz_iri'] < err_best:
                err_best = ret_valid['m_xyz_iri']
                is_best = True
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_xyz_iri'],
                           'right_hand_err': ret_valid['m_right_hand'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, part_condition=False,
              obstacle_condition=False, writer=None, device=0):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_xyz = 0
    if is_train <= 1:
        m_xyz_seq = 0
        m_right_hand_error = 0
        m_phase_acc = 0
        m_intention_acc = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_xyz_seq = np.zeros([opt.output_n])
        m_right_hand_error = np.zeros([opt.output_n])
        m_phase_acc = np.zeros([opt.output_n])
        m_intention_acc = np.zeros([opt.output_n])

    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    goal_features = opt.goal_features

    phase_loss = nn.BCELoss()
    intention_loss = nn.CrossEntropyLoss()
    #intention_loss = nn.CrossEntropyLoss(weight=torch.tensor([15614, 1399, 1088, 224]).to(device).float())
    #intention_loss = nn.CrossEntropyLoss(weight=torch.tensor([1/15614, 1/1399, 1/1088, 1/224]).to(device).float())

    dim_used = [0, 1, 2, #nose (0, 1, 2)
                                   #4, 5, 6,       #left_eye_inner
                                   #8, 9, 10,      #left_eye
                                   #12, 13, 14,    #left_eye_outer
                                   #16, 17, 18,    #right_eye_inner
                                   #20, 21, 22,    #right_eye
                                   #24, 25, 26,    #right_eye_outer
                                   #28, 29, 30,    #left_ear
                                   #32, 33, 34,    #right_ear
                                   #36, 37, 38,    #mouth_left
                                   #40, 41, 42,    #mouth_right
                                   44, 45, 46,    #left_shoulder (3, 4, 5)
                                   48, 49, 50,    #right_shoulder (6, 7, 8)
                                   52, 53, 54,    #left_elbow (9, 10, 11)
                                   56, 57, 58,    #right_elbow (12, 13, 14)
                                   60, 61, 62,    #left_wrist (15, 16, 17)
                                   64, 65, 66,    #right_wrist (18, 19, 20)
                                   #68, 69, 70,    #left_pinky
                                   #72, 73, 74,    #right_pinky
                                   #76, 77, 78,    #left_index
                                   #80, 81, 82,    #right_index
                                   #84, 85, 86,    #left_thumb
                                   #88, 89, 90,    #right_thumb
                                   92, 93, 94,  #left_hip (21, 22, 23)
                                   96, 97, 98]  #right_hip (24, 25, 26)

    seq_in = opt.kernel_size

    itera = 1
    idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    st = time.time()
    for i, batch in enumerate(data_loader):
        xyz_iri = batch['xyz'].to(device)
        # print(f'xyz_iri.shape: {xyz_iri.shape}')
        xyz_end_effector = batch['end_effector'].to(device)
        obstacles = batch['obstacles'].to(device)
        obstacles_ = obstacles
        obstacles = obstacles.view(-1, in_n+out_n, 9).float()
        # obstacles = torch.unsqueeze(obstacles, dim=1)
        # obstacles = obstacles.repeat(1, in_n+out_n, 1)
        # obstacles = torch.unsqueeze(obstacles, dim=1).float()
        # xyz_iri = xyz_iri[:, :, dim_used]

        phase = batch['phase'].to(device)
        phase_goal = batch['phase_goal'].to(device)
        # print(f'phase.shape: {phase.shape}')

        intention = batch['intention'].to(device)
        intention_goal = batch['intention_goal'].long().to(device)

        intention_goal = torch.zeros_like(intention_goal) + 4

        one_hot_format = True
        if one_hot_format:
            intention_goal = torch.squeeze(one_hot(intention_goal, num_classes=5).float(), dim=1)

        xyz_goal =[]

        if goal_features > 0:
            xyz_goal = xyz_end_effector
            """
            xyz_goal = torch.unsqueeze(xyz_end_effector, -1)
            xyz_iri_ = xyz_iri.view(xyz_iri.shape[0], -1, 3, 11)
            xyz_goal = xyz_iri_ - xyz_goal
            xyz_goal = xyz_goal.view(xyz_iri.shape[0],  -1, 33)
            """

        batch_size, seq_n, _ = xyz_iri.shape

        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue

        n += batch_size
        bt = time.time()
        xyz_iri = xyz_iri.float()

        xyz_sup = xyz_iri.clone()[:, -out_n - seq_in:]        # [batch_size, out_n+kernel, dim_used]
        xyz_src = xyz_iri.clone()                             # [batch_size, seq_len-1, dim_used]

        phase_sup = phase.clone()[:, -out_n - seq_in:]
        intention_sup = intention.clone()[:, -out_n - seq_in:]

        xyz_target = xyz_sup[:, seq_in:]

        phase_target = phase_sup.clone()[:, seq_in:]
        #phase_goal = torch.mode(phase_target, dim=1)[0]
        #print(f'phase_goal dimensions: {phase_goal}')


        intention_target = intention_sup.clone()[:, seq_in:]
        intention_target = torch.squeeze(intention_target, dim=2)

        #intention_goal = torch.unsqueeze(torch.mode(intention_target, dim=1)[0], dim=1)
        #print(f'intention_goal dimensions: {intention_goal}')

        # print(intention_goal.shape)

        noise = None
        #noise = torch.randn(batch_size, 128, device=device)
        #noise = torch.ones_like(noise) * 0

        xyz_src.requires_grad = True

        # Generator forward
        xyz_out_all, phase_pred, intention_pred = net_pred(xyz_src, output_n=out_n, itera=itera, input_n=in_n, goal=xyz_goal,
                               part_condition=obstacle_condition, obstacles=obstacles, phase=phase, intention=intention,
                               phase_goal=phase_goal, intention_goal=intention_goal) # [batch_size, out_n+kernel, 1, dim_used]

        xyz_out_all = xyz_out_all[:, :, 0]

        xyz_out = xyz_out_all[:, seq_in:]
        phase_pred = phase_pred[:, :, seq_in:].permute((0, 2, 1))
        intention_pred = intention_pred[:, :, seq_in:]
        # print(xyz_out.shape)

        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            loss_x = torch.mean(torch.sum(torch.abs(xyz_out - xyz_target)[:, :, [0, 3, 6, 9, 12, 15, 18, 21, 24]], dim=2))
            loss_y = torch.mean(torch.sum(torch.abs(xyz_out - xyz_target)[:, :, [1, 4, 7, 10, 13, 16, 19, 22, 25]], dim=2))
            loss_z = torch.mean(torch.sum(torch.abs(xyz_out - xyz_target)[:, :, [2, 5, 8, 11, 14, 17, 20, 23, 26]], dim=2))

            # loss_xyz = torch.mean(torch.sum(torch.abs(xyz_out - xyz_target), dim=2))
            wx = 1
            # wy = 10
            # wz = 2
            wy = 1
            wz = 1

            loss_xyz = wx * loss_x + wy * loss_y + wz * loss_z

            # we = 50
            we = 1

            loss_end_effector = torch.mean(torch.sum(torch.abs(xyz_out - xyz_target)[:, :, [18, 19, 20]], dim=2))

            wf = 5
            loss_free_hand = torch.mean(torch.sum(torch.abs(xyz_out - xyz_target)[:, :, [15, 16, 17]], dim=2))

            wo = 1
            loss_obstacle = 0

            for i in range(3):
                intersection_left_hip = xyz_out[:, :, [21, 22, 23]] - obstacles_[:, -out_n:, :, i]
                intersection_right_hip = xyz_out[:, :, [24, 24, 26]] - obstacles_[:, -out_n:, :, i]

                if torch.any(torch.abs(intersection_left_hip[:, :, 0]) < 0.1):
                    if torch.any(torch.abs(intersection_left_hip[:, :, 1]) < 0.1):
                        loss_obstacle = 50

                if torch.any(torch.abs(intersection_right_hip[:, :, 0]) < 0.1):
                    if torch.any(torch.abs(intersection_right_hip[:, :, 1]) < 0.1):
                        loss_obstacle = 50

            wp = 1
            loss_phase = phase_loss(phase_pred, phase_target)

            wi = 1
            loss_intention = intention_loss(intention_pred, intention_target.long())

            loss_all = loss_xyz + we * loss_end_effector + wf * loss_free_hand + wo * loss_obstacle + wp * loss_phase + wi * loss_intention
            optimizer.zero_grad()
            loss_all.backward()
            grad_norm = nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()

            # update log values
            # l_xyz += loss_xyz.cpu().data.numpy() * batch_size
            l_xyz += loss_all.cpu().data.numpy() * batch_size

        if is_train <= 1:  # if is validation or train simply output the overall mean error

            with torch.no_grad():
                xyz_gt = xyz_iri[:, in_n:in_n + out_n]
                xyz_out = xyz_out_all[:, seq_in:]

                err_xyz_seq = torch.mean(torch.norm(xyz_out - xyz_gt, dim=1))
                # right_hand_error = torch.mean(torch.norm(xyz_out - xyz_gt, dim=1)[:, [12, 13, 14]])

                # right_hand_error = torch.mean(torch.norm((xyz_out - xyz_gt)[:, :, [18, 19, 20]], dim=1))
                right_hand_error = torch.mean(torch.norm((xyz_out - xyz_gt)[:, :, [18, 19, 20]], dim=2))

                phase_pred_ = phase_pred
                phase_pred_[phase_pred_ < 0.5] = 0
                phase_pred_[phase_pred_ > 0.5] = 1
                phase_acc = (phase_pred_ == phase_target).sum() / torch.numel(phase_pred_)
                # print(phase_acc)
                # print((phase_pred == phase_target).sum())

                intention_pred_ = torch.argmax(intention_pred, dim=1)

                intention_acc = (intention_pred_ == intention_target).sum() / torch.numel(intention_pred_)

            m_xyz_seq += err_xyz_seq.cpu().data.numpy() * batch_size
            m_right_hand_error += right_hand_error.cpu().data.numpy() * batch_size
            m_phase_acc += phase_acc.cpu().data.numpy() * batch_size
            m_intention_acc += intention_acc.cpu().data.numpy() * batch_size
            # print(m_phase_acc)

        else:
            with torch.no_grad():
                xyz_gt = xyz_iri[:, in_n:in_n + out_n]
                xyz_out = xyz_out_all[:, seq_in:]
                # print(xyz_out.shape)
                # print(xyz_out[[0]])

                samples = False
                if samples:
                    test = torch.norm(xyz_out - xyz_gt, dim=1)

                    right_hand_error = torch.mean(test[:, [18, 19, 20]])
                    print(right_hand_error)

                    # print("numel: ", torch.numel(test))
                    print("numel1 %: ", torch.numel(test[test < 0.15]) / torch.numel(test) * 100)

                    print("numel2 %: ", torch.numel(test[test < 0.25]) / torch.numel(test) * 100)

                err_xyz_seq = torch.sum(torch.norm(xyz_out - xyz_gt, dim=2), dim=0)
                right_hand_error = torch.sum(torch.norm((xyz_out - xyz_gt)[:, :, [18, 19, 20]], dim=2), dim=0)

                #right_hand_error = torch.mean(torch.norm(xyz_out - xyz_gt, dim=1)[:, :, [18, 19, 20]])
                phase_pred_ = phase_pred
                phase_pred_[phase_pred_ < 0.5] = 0
                phase_pred_[phase_pred_ > 0.5] = 1
                phase_acc = (phase_pred_ == phase_target).sum() / torch.numel(phase_pred_)
                # print(phase_acc)
                # print((phase_pred == phase_target).sum())

                intention_pred_ = torch.argmax(intention_pred, dim=1)

                intention_acc = (intention_pred_ == intention_target).sum() / torch.numel(intention_pred_)

            m_xyz_seq += err_xyz_seq.cpu().data.numpy()
            m_right_hand_error += right_hand_error.cpu().data.numpy()
            m_phase_acc += phase_acc.cpu().data.numpy()
            m_intention_acc += intention_acc.cpu().data.numpy()

        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))

    ret = {}
    if is_train == 0:
        ret["l_xyz"] = l_xyz / n

    if is_train <= 1:
        ret["m_xyz_iri"] = m_xyz_seq / n
        ret["m_right_hand"] = m_right_hand_error / n
        ret["m_phase_acc"] = m_phase_acc / n
        ret["m_intention_acc"] = m_intention_acc / n

    else:
        m_xyz_iri = m_xyz_seq / n
        m_right_hand_iri = m_right_hand_error / n
        m_phase_iri = m_phase_acc / n
        m_intention_iri = m_intention_acc / n

        if opt.n_bins>0:
            for j in range(out_n):
                ret["#{:d}".format(titles[j])] = m_xyz_iri[j]
        else:
            for j in range(out_n):
                ret["#{:d}".format(titles[j])] = m_xyz_iri[j], m_right_hand_iri[j], m_phase_iri[j], m_intention_iri[j]
                # print(ret["#{:d}".format(titles[j])])

    show = True

    if epo in [1, opt.epoch / 5, opt.epoch * 2 / 5, opt.epoch * 3 / 5, opt.epoch * 4 / 5, opt.epoch]:
        batch = 0
        #batch = 20
        #batch = 82
        #batch = 100
        #batch = 0
        #batch = 155
        #batch = 195
        #batch = 230
        #batch = 245
        #batch = 252
        #plots.plot_distribution(xyz_out_poses[batch].cpu(), xyz_out_prob[batch].cpu())

        #input_seq = xyz_iri[batch, :in_n].cpu()
        #plots.animate_iri_handover_sequence(input_seq, end_effector=xyz_goal[batch, :in_n], show=show, keep=False, hide_layout=False, save_figs=False, epoch=epo, train=is_train)
        pred_seq = xyz_out[batch].detach().cpu()
        #plots.animate_iri_handover_sequence(pred_seq, end_effector=xyz_end_effector[batch, in_n:in_n+out_n+1], color='prediction', keep=True, show=show, hide_layout=True, save_figs=True, epoch=epo, train=is_train)
        #gt_seq = xyz_sup[0, -(out_n+1):].cpu()
        gt_seq = xyz_gt[batch].cpu()
        #plots.animate_iri_handover_sequence(gt_seq, end_effector=xyz_end_effector[batch, in_n:in_n+out_n+1], color='target', keep=True, show=show, hide_layout=True, save_figs=True, epoch=epo, train=is_train)

        video_buf, fig, ax, frame = plots.animate_mediapipe_target_and_prediction(gt_seq, pred_seq,  end_effector=xyz_end_effector[batch, in_n:in_n+out_n+1].cpu().numpy(), obstacles=obstacles_[batch].cpu().numpy(), show=show,  hide_layout=False, epoch=epo, train=is_train)
        video = torch.cat(video_buf).unsqueeze(0)
        writer.add_video("prediction vs target", video, epo)

    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
