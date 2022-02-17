from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import GCN
import utils.util as util
import numpy as np

class MultiHeadAttModel(Module):
    def __init__(self, in_features=33, kernel_size=10, d_model=512, num_stage=2, dct_n=10, num_heads=1, parts=1):
        super(MultiHeadAttModel, self).__init__()
        self.heads = nn.ModuleList([AttHeadModel(in_features, kernel_size, d_model, num_stage, dct_n, parts) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * 20, 20)

    def forward(self, src, output_n=25, input_n=50, itera=1, dct_m=[]):
        return self.linear(torch.cat([h(src, output_n, input_n, itera, dct_m) for h in self.heads], dim=-1))


class AttHeadModel(Module):
    def __init__(self, in_features=33, kernel_size=10, d_model=512, num_stage=2, dct_n=10, parts=1):
        super(AttHeadModel, self).__init__()

        self.in_features = [in_features]
        if parts == 3: self.in_features = [15, 9, 9]
        if parts == 33: self.in_features = np.ones(parts)

        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        self.parts = parts
        # ks = int((kernel_size + 1) / 2)

        assert kernel_size == 10

        self.convQ = nn.ModuleList()
        self.convK = nn.ModuleList()

        for features in self.in_features:
            self.convQ.append(nn.Sequential(nn.Conv1d(in_channels=features, out_channels=d_model, kernel_size=6,
                                                     bias=False),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(d_model),
                                           nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                                     bias=False),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(d_model)))

            self.convK.append(nn.Sequential(nn.Conv1d(in_channels=features, out_channels=d_model, kernel_size=6,
                                                     bias=False),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(d_model),
                                           nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                                     bias=False),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(d_model)))


    def forward(self, src, output_n=25, input_n=50, itera=1, dct_m=[]):
        """
        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        dct_n = self.dct_n


        if dct_m == []:
            # Create DCT matrix and its inverse
            dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
            dct_m = torch.from_numpy(dct_m).float()
            idct_m = torch.from_numpy(idct_m).float()

            if torch.cuda.is_available():
                dct_m = dct_m.cuda()
                idct_m = idct_m.cuda()

        # Take only the input seq
        src = src[:, :input_n]  # [bs,in_n,dim]
        src_tmp = src.clone()
        bs = src.shape[0]

        full_body = torch.unsqueeze(src_tmp, 0)


        if self.parts > 1:
            if self.parts == 3:
                right_arm_index = [6, 7, 8,
                             9, 10, 11,
                             12, 13, 14]

                left_arm_index = [15, 16, 17,
                            18, 19, 20,
                            21, 22, 23]
                torso_index = [0, 1, 2,
                         3, 4, 5,
                         24, 25, 26,
                         27, 28, 29,
                         30, 31, 32]

                right_arm_src_tmp = src[:, :, right_arm_index]
                left_arm_src_tmp = src[:, :, left_arm_index]
                torso_src_tmp = src[:, :, torso_index]


                full_body = [torso_src_tmp, right_arm_src_tmp, left_arm_src_tmp]


        full_body_dct = torch.Tensor().cuda()
        for i, part in enumerate(full_body):
            src_tmp = part


            # Temporal variables for keys and query
            src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()  # [batch, dims, input_n-output_n]
            src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone() # [batch, dims, kernel, bins]

            # Compute number of subsequences
            vn = input_n - self.kernel_size - output_n + 1

            # Compute number of frames per subsequence
            vl = self.kernel_size + output_n

            idx = np.expand_dims(np.arange(vl), axis=0) + \
                  np.expand_dims(np.arange(vn), axis=1)

            # Get raw poses corresponding to the Value of each subsequence
            src_value_tmp = src_tmp[:, idx].clone().reshape([bs * vn, vl, -1])

            # Obtain the DCT corresponding to the Value raw poses
            src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).\
                reshape([bs, vn, dct_n, -1]).\
                transpose(2, 3).\
                reshape([bs, vn, -1])  # [32,40,66*11]

            # Obtain the K features
            key_tmp = self.convK[i](src_key_tmp / 1000.0)
            #print(f'convK input shape: {src_key_tmp.shape}')
            #print(f'convK output shape: {key_tmp.shape}')

            for j in range(itera):
                # Obtain the Q features
                query_tmp = self.convQ[i](src_query_tmp / 1000.0)
                #print(f'convQ input shape: {src_query_tmp.shape}')
                #print(f'convQ output shape: {query_tmp.shape}')

                # Obtain the scores
                score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15

                # Normalize scores
                att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))

                # Obtain the attention results
                dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                    [bs, -1, dct_n])

            full_body_dct = torch.cat((full_body_dct, dct_att_tmp), dim=1)

        #return dct_att_tmp
        return full_body_dct

if __name__ == "__main__":
    model = MultiHeadAttModel()
    src = torch.rand([1, 75, 33]).cuda()
    print(model(src).shape)