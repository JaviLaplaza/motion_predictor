## Human motion prediction
This is the code for the human skeleton tracker.

### Installation

Install the following dependencies:
```
pip install torch==1.4.0 torchvision==0.5.0
pip install future scipy matplotlib pandas tensorboard
```

If you are in Ubuntu 20.04, you might need to do the following installation:

```
pip install grpcio==1.20.1
```

### Get the data
You can request access to the dataset [here](https://drive.google.com/file/d/1kXTmMPh2anxYkT7C5Fla6Z1YOhLL9_dM/view?usp=sharing).

Directory structure: 
```shell script
H3.6m
|-- S1
|-- S2
|-- S3
|-- ...
`-- S10
```

An additional "Tests" folder containing a few tests performed before the actual data collection.

Put the all downloaded datasets in ./datasets directory or any other path. You can modify the file "opt.py" by changing the value of the "--root_path" parameter with your real dataset path.

### Training
All the running args are defined in [opt.py](utils/opt.py). We use following commands to train on different datasets and representations.

Simple train with no added features:
```bash
python main_iri_handover_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 40 --skip_rate 5 --batch_size 256 --test_batch_size 128 --in_features 27 --num_heads 5 
```

Train with robot end effector data:
```bash
python main_iri_handover_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 40 --skip_rate 5 --batch_size 256 --test_batch_size 128 --in_features 27 --num_heads 5 --goal_features 3 --fusion_model 1
```

Train with obstacle position:
```bash
python main_iri_handover_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 40 --skip_rate 5 --batch_size 256 --test_batch_size 128 --in_features 27 --num_heads 5--obstacles_condition --fusion_model 1
```

Train with intention and phase classificator:
```bash
python main_iri_handover_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 40 --skip_rate 5 --batch_size 256 --test_batch_size 128 --in_features 27 --num_heads 5 --fusion_model 1 --phase --intention
```

Training with all options:
```bash
python main_iri_handover_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 40 --skip_rate 5 --batch_size 256 --test_batch_size 128 --in_features 27 --num_heads 5 --goal_features 3 --obstacles_condition --fusion_model 1 --phase --intention
```

### Evaluation
Simply move the recorded weights from the corresponding checkpoint folder to the root folder and execute the script:

```bash
python main_iri_handover_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 40 --skip_rate 5 --batch_size 256 --test_batch_size 128 --in_features 27 --num_heads 5 --goal_features 3 --obstacles_condition --fusion_model 1 --phase --intention --is_load --is_eval
```

### Acknowledgments
This code is a variation of the work done by Wei Mao, Miaomiao Liu, Mathieu Salzmann.Wei Mao, Miaomiao Liu, Mathieu Salzmann in the paper [History Repeats Itself: Human Motion Prediction via Motion Attention] (https://arxiv.org/abs/2007.11755) presented in ECCV 2020.
The overall code framework (dataloading, training, testing etc.) is adapted from [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline). 

The predictor model code is adapted from [LTD](https://github.com/wei-mao-2019/LearnTrajDep).

Some of our evaluation code and data process code was adapted/ported from [Residual Sup. RNN](https://github.com/una-dinosauria/human-motion-prediction) by [Julieta](https://github.com/una-dinosauria). 

### Licence
MIT
