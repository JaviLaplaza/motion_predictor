from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import GCN
from model import AttModel
import utils.util as util
import numpy as np

class MixAttention(Module):

    def __init__(self, input_n=50, output_n=25, in_features=33, kernel_size=10, d_model=512, num_stage=2, dct_n=10, num_heads=1, goal_features=-1,
                 part_condition=False, fusion_model=0, obstacle_condition=False, phase=False, intention=False):
        super(MixAttention, self).__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n
        self.fusion_model = fusion_model

        self.atnn = nn.ModuleList()


        self.atnn.append(AttModel.MultiHeadAttModel(in_features=in_features, kernel_size=kernel_size,
                                      d_model=d_model, num_stage=num_stage, dct_n=dct_n, num_heads=num_heads, parts=1))

        n = 2
        n_ = 2

        if goal_features > 0:
            n += 1
            self.features = nn.Linear(in_features=goal_features, out_features=in_features)
            self.atnn.append(AttModel.MultiHeadAttModel(in_features=in_features, kernel_size=kernel_size,
                                      d_model=d_model, num_stage=num_stage, dct_n=dct_n, num_heads=num_heads))

        if part_condition:
            n += 1
            self.atnn.append(AttModel.MultiHeadAttModel(in_features=in_features, kernel_size=kernel_size,
                                                        d_model=d_model, num_stage=num_stage, dct_n=dct_n,
                                                        num_heads=num_heads, parts=3))

        if obstacle_condition:
            #self.obstacle_features = nn.ModuleList()
            n += 1
            self.obstacle_features = nn.Linear(in_features=3*3, out_features=in_features)
            #self.obstacle_features = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3))
            self.atnn.append(AttModel.MultiHeadAttModel(in_features=in_features, kernel_size=kernel_size,
                                                        d_model=d_model, num_stage=num_stage, dct_n=dct_n,
                                                        num_heads=num_heads))
            #for obstacle in obstacles:
            #n += 1
            #self.obstacle_features.append(nn.Linear(in_features=3, out_features=in_features))
            #self.atnn.append(AttModel.MultiHeadAttModel(in_features=in_features, kernel_size=kernel_size,
            #                                            d_model=d_model, num_stage=num_stage, dct_n=dct_n,
            #                                            num_heads=num_heads))


        self.phase_detector = []
        if phase:
            self.phase_detector = nn.Sequential(nn.Conv1d(in_channels=27, out_channels=48, kernel_size=1, padding=0),
                                              nn.ReLU(),
                                              nn.Conv1d(in_channels=48, out_channels=1, kernel_size=1, padding=0),
                                              nn.Sigmoid())

            self.phase_condition = nn.Sequential(nn.Linear(in_features=1, out_features=d_model),
                                                 nn.ReLU(),
                                                 nn.BatchNorm1d(d_model),
                                                 nn.Linear(in_features=d_model, out_features=dct_n),
                                                 nn.ReLU(),
                                                 nn.BatchNorm1d(dct_n))

            self.phase_condition_pre = nn.Conv1d(in_channels=1, out_channels=in_features, kernel_size=1, padding=0)

            if fusion_model == 0:
                n += 1

        self.intention_detector = []
        if intention:
            self.intention_detector = nn.Sequential(nn.Conv1d(in_channels=27, out_channels=48, kernel_size=1, padding=0),
                                              nn.ReLU(),
                                              nn.Conv1d(in_channels=48, out_channels=5, kernel_size=1, padding=0))

            """
            self.intention_condition = nn.Sequential(nn.Linear(in_features=1, out_features=d_model),
                                                 nn.ReLU(),
                                                 nn.BatchNorm1d(d_model),
                                                 nn.Linear(in_features=d_model, out_features=dct_n),
                                                 nn.ReLU(),
                                                 nn.BatchNorm1d(dct_n))
            """

            self.intention_condition = nn.Sequential(nn.Linear(in_features=5, out_features=d_model),
                                                     nn.ReLU(),
                                                     nn.BatchNorm1d(d_model),
                                                     nn.Linear(in_features=d_model, out_features=dct_n),
                                                     nn.ReLU(),
                                                     nn.BatchNorm1d(dct_n))


            self.intention_condition_pre = nn.Conv1d(in_channels=1, out_channels=in_features, kernel_size=1, padding=0)
            if fusion_model == 0:
                n += 1

        noise = True
        if noise:
            self.noise_embedding = nn.Sequential(nn.Linear(in_features=128, out_features=d_model),
                                                     nn.ReLU(),
                                                     nn.BatchNorm1d(d_model),
                                                     nn.Linear(in_features=d_model, out_features=dct_n),
                                                     nn.ReLU(),
                                                     nn.BatchNorm1d(dct_n),)


        self.gcn = GCN.GCN(output_n=output_n, input_feature=(dct_n) * n, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        if fusion_model == 1:
            self._dct_att_tmp_weights = nn.Parameter(torch.ones(n-1))

            self.gcn = GCN.GCN(output_n=output_n, input_feature=(dct_n) * n_, hidden_feature=d_model, p_dropout=0.3,
                               num_stage=num_stage,
                               node_n=in_features)
            #self.fusion_module = GCN.FusionGCN(input_feature=(dct_n) * n, hidden_feature=d_model, p_dropout=0.3,
            #                   node_n=in_features)

            self.fusion_module = GCN.FusionGCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                               node_n=in_features)

        elif fusion_model == 2:
            self.gcn = GCN.GCN(output_n=output_n, input_feature=(dct_n) * n_, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features, input_n=input_n, kernel_n=kernel_size)

            #self.fusion_module = GCN.FusionGCN(input_feature=(dct_n) * (n-1), hidden_feature=d_model, p_dropout=0,
            #                                   output_feature=(n - 1), node_n=in_features)

            self.fusion_module = GCN.FusionGCN(input_feature=in_features * (n - 1), hidden_feature=d_model, p_dropout=0.3,
                                               output_feature=(n - 1), node_n=35)


    def forward(self, src, output_n=25, input_n=50, itera=1, goal=[], part_condition=False, obstacles=[],
                phase=[], intention=[], phase_goal=torch.tensor([]), intention_goal=torch.tensor([]), z=torch.tensor([])):
        dct_n = self.dct_n
        src_tmp = src[:, :input_n].clone()
        # Create DCT matrix and its inverse
        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float()
        idct_m = torch.from_numpy(idct_m).float()

        if torch.cuda.is_available():
            src = src.cuda()
            dct_m = dct_m.cuda()
            idct_m = idct_m.cuda()
            

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        #dct_att_tmp = self.atnn[0](src, output_n=25, input_n=50, itera=1, dct_m=dct_m)
        dct_att_tmp = []

        inputs = [src]
        

        if goal != []:
            if torch.cuda.is_available():
                goal = goal.cuda()
            goal = self.features(goal)

            inputs.append(goal)

            #dct_goal_tmp = self.atnn[1](goal, output_n=25, input_n=50, itera=1, dct_m=dct_m)

            #dct_att_tmp = torch.cat((dct_att_tmp, dct_goal_tmp), dim=-1)

        if part_condition:
            inputs.append(src)
            #dct_parts_tmp = self.atnn[2](src, output_n=25, input_n=50, itera=1, dct_m=dct_m)
            #dct_att_tmp = torch.cat((dct_att_tmp, dct_parts_tmp), dim=-1)

        if obstacles != []:
            obstacles_enc = self.obstacle_features(obstacles)
            inputs.append(obstacles_enc)

        if len(phase_goal.shape) > 1:
            phase_goal = self.phase_condition(phase_goal)

        if len(intention_goal.shape) > 1:
            intention_goal = self.intention_condition(intention_goal)

        if len(z.shape) > 1:
            z = self.noise_embedding(z)
            #print(z.shape)

        # Generate internal variables U
        for source, module in zip(inputs, self.atnn):
            dct_source_tmp = module(source, output_n=self.output_n, input_n=self.input_n, itera=1, dct_m=dct_m)
            
            dct_source_tmp = dct_source_tmp.transpose(1, 2)
            dct_source_tmp = torch.unsqueeze(dct_source_tmp, dim=0)
            #print(dct_source_tmp.shape)
            dct_att_tmp.append(dct_source_tmp)


        # Set the temporal variable to input the GCN
        input_gcn = src_tmp[:, :, idx]

        phase_pred = torch.empty((0, 0, 0))

        intention_pred = torch.empty((0, 0, 0))

        if torch.cuda.is_available():
            phase_pred = phase_pred.cuda()
            intention_pred = intention_pred.cuda()
            intention_goal = intention_goal.cuda()
            input_gcn = input_gcn.cuda()

        if self.fusion_model == 0:
            dct_att_tmp = torch.squeeze(torch.cat(dct_att_tmp, dim=-1), dim=0)

            # Compute the DCT coeff for the GCN input seq
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)

            #print(f"dct_in_tmp dimensions: {dct_in_tmp.shape}")
            #print(f"dct_att_tmp dimensions: {dct_att_tmp.shape}")
            #print(f"phase_goal dimensions: {phase_goal.shape}")
            #print(f"intention_goal dimensions: {intention_goal.shape}")

            phase_goal = torch.unsqueeze(phase_goal, dim=1)
            phase_goal = self.phase_condition_pre(phase_goal)

            intention_goal = torch.unsqueeze(intention_goal, dim=1)
            intention_goal = self.intention_condition_pre(intention_goal)

            dct_att_tmp = torch.cat((dct_att_tmp, phase_goal, intention_goal), dim=-1)

            # Concatenate the DCT coeff and the att output
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)

            dct_out_tmp, phase_pred, intention_pred = self.gcn(dct_in_tmp)
            #print(f"dct_out_tmp dimensions: {dct_out_tmp.shape}")
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            outputs.append(out_gcn.unsqueeze(2))

            outputs = torch.cat(outputs, dim=2)

            return outputs, phase_pred, intention_pred

        elif self.fusion_model == 1:
            dct_att_tmp = torch.cat(dct_att_tmp).permute(1, 2, 3, 0)
            dct_att_tmp = torch.sum(self._dct_att_tmp_weights * dct_att_tmp, dim=3)

            # dct_att_tmp += torch.unsqueeze(phase_goal, dim=1)

            dct_att_tmp += torch.unsqueeze(intention_goal, dim=1)

            #dct_att_tmp += torch.unsqueeze(z, dim=1)

            #print(len(dct_att_tmp))
            #dct_att_tmp = torch.cat(dct_att_tmp, dim=-1)
            #dct_att_tmp = self.fusion_module(dct_att_tmp)
            #print(dct_att_tmp.shape)


            # Compute the DCT coeff for the GCN input seq
            dct_in_tmp_ = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0).double(), input_gcn.transpose(1, 2).double())

            # Concatenate the DCT coeff and the att output
            dct_in_tmp = torch.cat([dct_in_tmp_.transpose(1, 2), dct_att_tmp.transpose(1, 2)], dim=-1)
            b, f, _ = dct_in_tmp.shape

            dct_out_tmp, phase_pred, intention_pred = self.gcn(dct_in_tmp) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PHASE AND INTENTION  ADDED

            #dct_att_tmp_weighted = torch.zeros(b, f, dct_n * 2)
            ##dct_att_tmp_weighted = dct_in_tmp_

            #dct_att_tmp_weighted_ = torch.mul(torch.unsqueeze(dct_coef[:, :, 0], dim=-1), dct_in_tmp[:, :, dct_n:dct_n * 2])

            #for item in range(1, dct_coef.shape[-1]):
            #    dct_att_tmp_weighted_ += torch.mul(torch.unsqueeze(dct_coef[:, :, item], dim=-1), dct_in_tmp[:, :, dct_n+dct_n*item:dct_n*2 + dct_n*item])

            #dct_att_tmp_weighted = torch.cat((dct_att_tmp_weighted, dct_att_tmp_weighted_), dim=-1)

            #dct_out_tmp = self.gcn(dct_att_tmp_weighted)
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0).float(),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2).float())

            #print(f'dct_out_tmp dimensions: {dct_out_tmp[:, :, :dct_n].shape}')
            #print(f'idct_m dimensions: {idct_m[:, :dct_n].shape}')

            outputs.append(out_gcn.unsqueeze(2))

            outputs = out_gcn
            #print(f'outputs dimmensions: {outputs.shape}')

            #outputs = torch.cat(outputs, dim=2)

            #outputs = torch.squeeze(outputs, dim=2)
            outputs = outputs.permute((0, 2, 1))

            #if self.phase_detector != []:
            #    phase_pred = self.phase_detector(outputs)

            #if self.intention_detector != []:
            #    intention_pred = self.intention_detector(outputs)

            outputs = outputs.permute((0, 2, 1))
            outputs = torch.unsqueeze(outputs, dim=2)


            return outputs, phase_pred, intention_pred

        elif self.fusion_model == 2:

            x_hat = []

            # Compute the DCT coeff for the GCN input seq
            dct_in_tmp_ = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)

            for u in dct_att_tmp:
                # Concatenate the DCT coeff and the att output
                dct_in_tmp = torch.cat([dct_in_tmp_, u], dim=-1)

                dct_out_tmp = self.gcn(dct_in_tmp)
                out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                       dct_out_tmp[:, :, :dct_n].transpose(1, 2))
                x_hat.append(out_gcn.unsqueeze(2))

                #outputs = torch.cat(outputs, dim=2)
                #print(x_hat[0].shape)

            x_hat_weighted = torch.squeeze(torch.zeros_like(x_hat[0]), dim=2)
            x_hat = torch.cat(x_hat, dim=-1)[:, :, 0]

            x_hat_weights = self.fusion_module(x_hat)

            for item in range(x_hat_weights.shape[-1]):
                x_hat_weighted += torch.mul(x_hat[:, :, item*33:33*(1 + item)], torch.unsqueeze(x_hat_weights[:, :, item], dim=-1))

            x_hat_weighted = torch.unsqueeze(x_hat_weighted, dim=2)

            return x_hat_weighted, phase_pred, intention_pred





if __name__ == "__main__":
    model = MixAttention()
    src = torch.rand([8, 75, 33])
    print(model(src).shape)
