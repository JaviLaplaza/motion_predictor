from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
import model.GCN as GCN
import model.AttModel as AttModel
import utils.util as util
import numpy as np

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
        
    def forward(self, x):
        print(x.shape)
        return x

class MixAttention(Module):

    def __init__(self, input_n=50, output_n=25, in_features=33, kernel_size=10, d_model=512, num_stage=2, dct_n=10,
                 itera=1, num_heads=1, goal_condition=False, fusion_model=0, obstacle_condition=False,
                 phase_condition=False, intention_condition=False, phase_prediction=False, intention_prediction=False,
                 robot_path_condition=False, device=0):
        super(MixAttention, self).__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n
        self.fusion_model = fusion_model
        self.device = device

        self.goal_condition = goal_condition
        self.obstacle_condition = obstacle_condition
        self.phase_condition = phase_condition
        self.intention_condition = intention_condition
        self.phase_prediction = phase_prediction
        self.intention_prediction = intention_prediction
        self.robot_path_condition = robot_path_condition

        self._fusion_model = fusion_model

        self.atnn = nn.ModuleList()

        self.atnn.append(AttModel.MultiHeadAttModel(in_features=in_features, kernel_size=kernel_size,
                                      d_model=d_model, num_stage=num_stage, dct_n=dct_n, num_heads=num_heads, parts=1, device=self.device))

        n = 2
        n_ = 2

        if self.goal_condition:
            n += 1
            self.features = nn.Linear(in_features=3, out_features=in_features)
            self.atnn.append(AttModel.MultiHeadAttModel(in_features=in_features, kernel_size=kernel_size,
                                      d_model=d_model, num_stage=num_stage, dct_n=dct_n, num_heads=num_heads, device=self.device))

        # if part_condition:
        #     n += 1
        #     self.atnn.append(AttModel.MultiHeadAttModel(in_features=in_features, kernel_size=kernel_size,
        #                                                 d_model=d_model, num_stage=num_stage, dct_n=dct_n,
        #                                                 num_heads=num_heads, parts=3))

        if self.obstacle_condition:
            n += 1
            self.obstacle_features = nn.Linear(in_features=3*3, out_features=in_features)
            self.atnn.append(AttModel.MultiHeadAttModel(in_features=in_features, kernel_size=kernel_size,
                                                        d_model=d_model, num_stage=num_stage, dct_n=dct_n,
                                                        num_heads=num_heads, device=self.device))

        if self.robot_path_condition:
            # self.robot_path_condition = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=d_model, kernel_size=6,
            #                                          bias=False),
            #                                nn.ReLU(),
            #                                nn.BatchNorm1d(d_model),
            #                                nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
            #                                          bias=False),
            #                                nn.ReLU(),
            #                                nn.BatchNorm1d(d_model))

            n += 1
            self.path_features = nn.Linear(in_features=2, out_features=in_features)
            self.atnn.append(AttModel.MultiHeadAttModel(in_features=in_features, kernel_size=kernel_size,
                                                        d_model=d_model, num_stage=num_stage, dct_n=dct_n,
                                                        num_heads=num_heads, device=self.device))

        self.phase_detector = []
        if self.phase_condition:
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
        if self.intention_condition:
            # self.intention_network = nn.Sequential(nn.Linear(in_features=5, out_features=d_model),
            self.intention_network = nn.Sequential(nn.Linear(in_features=1, out_features=d_model),
                                                     nn.BatchNorm1d(num_features=d_model),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=d_model, out_features=dct_n),
                                                     nn.BatchNorm1d(num_features=dct_n),
                                                     nn.ReLU()
                                                     )

            self.intention_predictor = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=5, padding=0),
                                                     # PrintLayer(),
                                                     nn.Dropout(0.3),
                                                     nn.BatchNorm1d(num_features=d_model),
                                                     nn.ReLU(),
                                                     nn.Conv1d(in_channels=d_model, out_channels=int(d_model/2), kernel_size=4, padding=0),
                                                     # PrintLayer(),
                                                     nn.Dropout(0.3),
                                                     nn.BatchNorm1d(num_features=int(d_model/2)),
                                                     nn.ReLU(),
                                                     nn.Flatten(),
                                                     # PrintLayer(),
                                                     nn.Linear(in_features=11008, out_features=int(d_model/4)),
                                                     nn.Dropout(0.3),
                                                     # PrintLayer(),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=int(d_model/4), out_features=4))

            self.intention_condition_pre = nn.Conv1d(in_channels=1, out_channels=in_features, kernel_size=1, padding=0)
            if fusion_model == 0:
                n += 1

        if self.intention_prediction:
            self.intention_detector = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=5, padding=0),
                                                    # PrintLayer(),
                                                    nn.Dropout(0.3),
                                                    nn.BatchNorm1d(num_features=d_model),
                                                    nn.ReLU(),
                                                    nn.Conv1d(in_channels=d_model, out_channels=int(d_model/2), kernel_size=4, padding=0),
                                                    # PrintLayer(),
                                                    nn.Dropout(0.3),
                                                    nn.BatchNorm1d(num_features=int(d_model/2)),
                                                    nn.ReLU(),
                                                    nn.Flatten(),
                                                    # PrintLayer(),
                                                    nn.Linear(in_features=7168, out_features=int(d_model/4)),
                                                    nn.Dropout(0.3),
                                                    # PrintLayer(),
                                                    nn.ReLU(),
                                                    nn.Linear(in_features=int(d_model/4), out_features=4))



        noise = False
        if noise:
            self.noise_embedding = nn.Sequential(nn.Linear(in_features=128, out_features=d_model),
                                                     nn.ReLU(),
                                                     nn.BatchNorm1d(d_model),
                                                     nn.Linear(in_features=d_model, out_features=dct_n),
                                                     nn.ReLU(),
                                                     nn.BatchNorm1d(dct_n))
        self.gcn = GCN.GCN(output_n=output_n, input_feature=(dct_n) * n, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage, node_n=in_features, phase_pred=phase_prediction, intention_pred=intention_prediction)

        if self.fusion_model == 1:
            self._dct_att_tmp_weights = nn.Parameter(torch.ones(n-1)) # .to(self.device)

            self.gcn = GCN.GCN(output_n=output_n, input_feature=(dct_n) * n_, hidden_feature=d_model, p_dropout=0.3,
                               num_stage=num_stage, node_n=in_features, phase_pred=phase_prediction, intention_pred=intention_prediction)
            # self.fusion_module = GCN.FusionGCN(input_feature=(dct_n) * n, hidden_feature=d_model, p_dropout=0.3,
            #                   node_n=in_features)

            # self.fusion_module = GCN.FusionGCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
            #                   node_n=in_features)

        elif self.fusion_model == 2:
            self.gcn = GCN.GCN(output_n=output_n, input_feature=(dct_n) * n_, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features, input_n=input_n, kernel_n=kernel_size, phase_pred=phase_prediction, intention_pred=intention_prediction)

            #self.fusion_module = GCN.FusionGCN(input_feature=(dct_n) * (n-1), hidden_feature=d_model, p_dropout=0,
            #                                   output_feature=(n - 1), node_n=in_features)

            # self.fusion_module = GCN.FusionGCN(input_feature=in_features * (n - 1), hidden_feature=d_model, p_dropout=0.3,
            #                                    output_feature=(n - 1), node_n=35)


    def forward(self, src, output_n=25, input_n=50, itera=1, goal=torch.Tensor([]), obstacles=torch.Tensor([]),
                robot_path=torch.Tensor([]), phase=torch.Tensor([]), intention=torch.Tensor([]), phase_goal=torch.tensor([]),
                intention_goal=torch.tensor([]), z=torch.tensor([])):
        dct_n = self.dct_n

        src_tmp = src[:, :input_n].clone().to(self.device)

        # Create DCT matrix and its inverse
        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().to(self.device)
        idct_m = torch.from_numpy(idct_m).float().to(self.device)
        

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        #dct_att_tmp = self.atnn[0](src, output_n=25, input_n=50, itera=1, dct_m=dct_m)
        dct_att_tmp = []

        inputs = [src]
        

        if self.goal_condition:
            goal = self.features(goal)
            inputs.append(goal)

            #dct_goal_tmp = self.atnn[1](goal, output_n=25, input_n=50, itera=1, dct_m=dct_m)
            #dct_att_tmp = torch.cat((dct_att_tmp, dct_goal_tmp), dim=-1)

        #if part_condition:
        #   inputs.append(src)
        #   dct_parts_tmp = self.atnn[2](src, output_n=25, input_n=50, itera=1, dct_m=dct_m)
        #   dct_att_tmp = torch.cat((dct_att_tmp, dct_parts_tmp), dim=-1)

        #if obstacles != []:
        if self.obstacle_condition:
            obstacles_enc = self.obstacle_features(obstacles)
            inputs.append(obstacles_enc)

        if self.robot_path_condition:
            robot_enc = self.path_features(robot_path)
            inputs.append(robot_enc)

        if self.phase_condition:
            phase_goal = self.phase_condition(phase_goal)
            

        # Initialize pre_intention_predicition variable
        pre_intention_prediction = torch.empty((0, 0, 0)).to(self.device)


        if self.intention_condition:
            # print(f"src_tmp[0]: {src_tmp[0]}")
            pre_intention_prediction = self.intention_predictor(src_tmp.float())
            # print(f"pre_intention_prediction: {pre_intention_prediction}")
            # print(f"pre_intention_prediction.shape: {pre_intention_prediction.shape}")
            pre_intention_prediction_idx = torch.argmax(nn.LogSoftmax(dim=1)(pre_intention_prediction), dim=1, keepdim=True)
            # print(f"pre_intention_prediction_idx: {pre_intention_prediction_idx}")
            intention_goal = self.intention_network(pre_intention_prediction_idx.float())
            # intention_goal = self.intention_network(intention_goal)
            

        #if len(z.shape) > 1:
        #    z = self.noise_embedding(z)
        #    #print(z.shape)


        # Generate internal variables U
        for source, module in zip(inputs, self.atnn):
            dct_source_tmp = module(source, output_n=self.output_n, input_n=self.input_n, itera=1, dct_m=dct_m)
            dct_source_tmp = torch.unsqueeze(dct_source_tmp, dim=0)
            #print(dct_source_tmp.shape)
            dct_att_tmp.append(dct_source_tmp)

        # Set the temporal variable to input the GCN
        input_gcn = src_tmp.permute((0, 2, 1))[:, idx].to(self.device)

        phase_pred = torch.empty((0, 0, 0)).to(self.device)

        intention_pred = torch.empty((0, 0, 0)).to(self.device)

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
            dct_att_tmp = dct_att_tmp.to(self.device)
            dct_att_tmp = torch.sum(self._dct_att_tmp_weights * dct_att_tmp, dim=3)

            if self.phase_condition:
                dct_att_tmp += torch.unsqueeze(phase_goal, dim=1)

            if self.intention_condition:
                dct_att_tmp += torch.unsqueeze(intention_goal, dim=1)


            #dct_att_tmp += torch.unsqueeze(z, dim=1)

            #print(len(dct_att_tmp))
            #dct_att_tmp = torch.cat(dct_att_tmp, dim=-1)
            #dct_att_tmp = self.fusion_module(dct_att_tmp)
            #print(dct_att_tmp.shape)


            # Compute the DCT coeff for the GCN input seq
            dct_in_tmp_ = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn.float()).transpose(1, 2).to(self.device)

            # Concatenate the DCT coeff and the att output
            dct_in_tmp = torch.cat([dct_in_tmp_, dct_att_tmp], dim=-1)
            b, f, _ = dct_in_tmp.shape

            dct_out_tmp, phase_pred, intention_pred = self.gcn(dct_in_tmp) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! PHASE AND INTENTION  ADDED

            #dct_att_tmp_weighted = torch.zeros(b, f, dct_n * 2)
            ##dct_att_tmp_weighted = dct_in_tmp_

            #dct_att_tmp_weighted_ = torch.mul(torch.unsqueeze(dct_coef[:, :, 0], dim=-1), dct_in_tmp[:, :, dct_n:dct_n * 2])

            #for item in range(1, dct_coef.shape[-1]):
            #    dct_att_tmp_weighted_ += torch.mul(torch.unsqueeze(dct_coef[:, :, item], dim=-1), dct_in_tmp[:, :, dct_n+dct_n*item:dct_n*2 + dct_n*item])

            #dct_att_tmp_weighted = torch.cat((dct_att_tmp_weighted, dct_att_tmp_weighted_), dim=-1)

            #dct_out_tmp = self.gcn(dct_att_tmp_weighted)
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))

            #print(f'dct_out_tmp dimensions: {dct_out_tmp[:, :, :dct_n].shape}')
            #print(f'idct_m dimensions: {idct_m[:, :dct_n].shape}')

            outputs.append(out_gcn.unsqueeze(2))

            outputs = out_gcn
            #print(f'outputs dimmensions: {outputs.shape}')

            #outputs = torch.cat(outputs, dim=2)

            #outputs = torch.squeeze(outputs, dim=2)
            # outputs = outputs.permute((0, 2, 1))

            #if self.phase_detector != []:
            #    phase_pred = self.phase_detector(outputs)

            if self.intention_prediction:
               intention_pred = self.intention_detector(outputs.permute(0, 2, 1))
               # print(f"post_intention_prediction: {intention_pred}")
               post_intention_prediction_idx = torch.argmax(nn.LogSoftmax(dim=1)(intention_pred), dim=1, keepdim=True)
               # print(f"post_intention_prediction_idx: {post_intention_prediction_idx}")

            # outputs = outputs.permute((0, 2, 1))
            outputs = torch.unsqueeze(outputs, dim=2)


            return outputs, pre_intention_prediction, phase_pred, intention_pred

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
    src = torch.rand([1, 27, 50])
    print(model(src).shape)
