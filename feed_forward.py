from model import MixAttention
from collections import OrderedDict
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
        self.goal_condition = opt.goal_condition
        self.part_condition = opt.part_condition
        self.obstacles_condition = opt.obstacles_condition
        self.fusion_model = opt.fusion_model
        self.device = opt.device

        self.net_pred = MixAttention.MixAttention(input_n=self.input_n, output_n=self.output_n, in_features=self.in_features, kernel_size=self.kernel_size,
                                             d_model=self.d_model, num_stage=opt.num_stage, dct_n=opt.dct_n, num_heads=self.num_heads,
                                             goal_condition=self.goal_condition,
                                             obstacle_condition=self.obstacles_condition, fusion_model=self.fusion_model,
                                             phase_condition=opt.phase_condition, intention_condition=opt.intention_condition, phase_prediction=opt.phase_prediction,
                                             intention_prediction=opt.intention_prediction).to(self.device)
        
                                             
        self.net_pred.eval()

        print(">>> total params: {:.2f}M".format(sum(p.numel() for p in self.net_pred.parameters()) / 1000000.0))

        # Load weights
        #model_path_len = './ckpt_best.pth.tar'
        # model_path_len = opt.weights_file
        model_path_len = '/home/irilab/iri-lab/motion_prediction_ws/src/human_detection/iri_motion_prediction/weights/ckpt_best.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        # self.net_pred.load_state_dict(ckpt['state_dict'])
        # print(self.net_pred.state_dict().keys())
        # print(ckpt.keys())
        # matches = 0
        # i = 0
        # for model_name, model_param in self.net_pred.state_dict().items():
        #     i += 1
        #     for ckpt_name, ckpt_param in ckpt.items():
        #         if model_name in ckpt_name:
        #             model_param = ckpt_param

        # self.net_pred.load_state_dict(torch.load(model_path_len), strict=False)
        
        new_state_dict = OrderedDict()
        for k, v in ckpt.items():
          name = k[7:]
          new_state_dict[name] = v
        self.net_pred.load_state_dict(new_state_dict, strict=False)
                    
        
                    

        # self.net_pred.load_state_dict(ckpt)

        self._dimensions_to_use = [0, 1, 2, #nose (0, 1, 2)
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

    def forward(self, sequence):
        upper_body_sequence = sequence[:, self._dimensions_to_use]
        upper_body_sequence = torch.from_numpy(upper_body_sequence)
        upper_body_sequence = torch.unsqueeze(upper_body_sequence, dim=0)
        upper_body_sequence = upper_body_sequence.permute(0, 2, 1)

        # Generator forward
        xyz_out_all, intention_goal, phase_pred, intention_pred = self.net_pred(upper_body_sequence, output_n=self.output_n, itera=1, input_n=self.input_n, intention_goal=torch.tensor([0]))  # [batch_size, out_n+kernel, 1, dim_used]
        

        xyz_out_all = xyz_out_all[:, :, 0]

        xyz_out = xyz_out_all[:, self.kernel_size:]
        phase_pred = torch.FloatTensor(phase_pred) # [:, :, self.kernel_size:]#.permute((0, 2, 1))
        # intention_pred = intention_pred[:, self.kernel_size:]
        phase_pred = torch.zeros_like(intention_pred)
        

        return xyz_out, intention_goal, phase_pred, intention_pred

