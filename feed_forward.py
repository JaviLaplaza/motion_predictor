from model import MixAttention
import torch



class Infer():
    def __init__(self, opt):
        # Define parameters
        lr_now = opt.lr_now
        start_epoch = 1
        # opt.is_eval = True
        print('>>> create models')

        self.input_n = opt.input_n
        self.output_n = opt.output_n

        self.in_features = opt.in_features  # 48
        self.d_model = opt.d_model
        self.kernel_size = opt.kernel_size
        self.num_heads = opt.num_heads
        # goal_condition = opt.goal_condition
        self.goal_features = -1
        self.part_condition = opt.part_condition
        self.obstacles_condition = opt.obstacles_condition
        self.fusion_model = opt.fusion_model
        self.device = opt.device

        self.net_pred = MixAttention.MixAttention(output_n=self.output_n, in_features=self.in_features, kernel_size=self.kernel_size,
                                             d_model=self.d_model, num_stage=opt.num_stage, dct_n=opt.dct_n, num_heads=self.num_heads,
                                             goal_features=self.goal_features, part_condition=self.part_condition,
                                             obstacle_condition=self.obstacles_condition, fusion_model=self.fusion_model,
                                             phase=opt.phase, intention=opt.intention).to(self.device)

        print(">>> total params: {:.2f}M".format(sum(p.numel() for p in self.net_pred.parameters()) / 1000000.0))

        # Load weights
        model_path_len = './ckpt_best.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        self.net_pred.load_state_dict(ckpt['state_dict'])

    def forward(self, sequence):


        # Generator forward
        xyz_out_all, phase_pred, intention_pred = self.net_pred(sequence, output_n=self.output_n, itera=1, input_n=self.input_n)  # [batch_size, out_n+kernel, 1, dim_used]

        xyz_out_all = xyz_out_all[:, :, 0]

        xyz_out = xyz_out_all[:, self.input_n:]
        phase_pred = phase_pred[:, :, self.input_n:].permute((0, 2, 1))
        intention_pred = intention_pred[:, :, self.input_n:]

        return xyz_out, phase_pred, intention_pred

