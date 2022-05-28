SkelAct
=======

SkelAct is an open source repository which provides state-of-the-art skeleton-based action recognition models from Hikvision Research Institute. Currently 5 models from 4 papers have been reimplemented in PyTorch, namely [Two-Stream CNN](https://arxiv.org/abs/1704.07595) (ICMEW'17), [HCN](https://arxiv.org/abs/1804.06055) (IJCAI'18), [HCN-Baseline](https://arxiv.org/abs/2112.04178) (AAAI'22), [Ta-CNN](https://arxiv.org/abs/2112.04178) (AAAI'22) and [Dynamic GCN](https://arxiv.org/abs/2007.14690) (ACM MM'20).

## Installation

SkelAct is based on [MMAction2](https://github.com/open-mmlab/mmaction2/). Follow the instruction below to setup a valid Python environment.

```shell
conda create -n skelact python=3.9 -y
conda activate skelact
conda install pytorch=1.11.0 torchvision=0.12.0 cudatoolkit=11.3 -c pytorch -y
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmaction2  # tested mmaction2 v0.24.0
```

## Get Started

### Data Preparation

Use [gen_ntu_rgbd_raw.py](tools/data/gen_ntu_rgbd_raw.py) to preprocess the NTU RGB+D dataset. Put the dataset in `data/` with the following structure.

```
data/
└── ntu
    └── nturgb+d_skeletons_60_3d
        ├── xsub
        │   ├── train.pkl
        │   └── val.pkl
        └── xview
            ├── train.pkl
            └── val.pkl
```

### Train

You can use the following command to train a model.

```shell
./tools/run.sh ${CONFIG_FILE} ${GPU_IDS} ${SEED}
```

Example: train HCN model on the joint data of NTU RGB+D using 2 GPUs with seed 0.

```shell
./tools/run.sh configs/hcn/hcn_ntu60_xsub_joint.py 0,1 0
```

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test HCN model on the joint data of NTU RGB+D.

```shell
python tools/test.py configs/hcn/hcn_ntu60_xsub_joint.py \
    work_dirs/hcn_ntu60_xsub_joint/best_top1_acc_epoch_475.pth \
    --eval top_k_accuracy --cfg-options "gpu_ids=[0]"
```

## Models

### FLOPs and Params

Model | GFLOPs<sup>1</sup> | Params (M)
------|--------|-----------
Two-Stream CNN | 0.098 | 0.785
HCN | 0.196 | 1.047
HCN-Baseline | 0.196 | 0.538
Ta-CNN | 0.147 | 0.532
Dynamic GCN | 2.395 | 3.75

<sup>1</sup> Calculated with [get_flops.py](tools/analysis/get_flops.py), which may differ from the numbers reported in the papers.

### Performance

All the following models are trained using 2 TITAN X Pascal GPUs. Note that for simplicity we do not strictly follow the details (e.g. data preprocessing) of the original implementations, which causes the slight accuracy difference.

- NTU RGB+D XSub

Model | Config | Our Acc (5 seeds<sup>2</sup>) | Our Acc (mean±std) | Paper Acc
------|--------|-------------------------------|--------------------|----------
Two-Stream CNN | [tscnn_ntu60_xsub_joint.py](configs/tscnn/tscnn_ntu60_xsub_joint.py) | 83.93, 83.78, 83.56, 84.04, 84.13 | 83.89±0.20 | 83.2
HCN | [hcn_ntu60_xsub_joint.py](configs/hcn/hcn_ntu60_xsub_joint.py) | 86.69, 85.89, 86.17, 86.63, 87.09 | 86.49±0.42 | 86.5
HCN-Baseline | [hcnb_ntu60_xsub_joint.py](configs/tacnn/hcnb_ntu60_xsub_joint.py) | 87.89, 87.26, 87.71, 87.72, 87.77 | 87.67±0.21 | 87.4
Ta-CNN | [tacnn_ntu60_xsub_joint.py](configs/tacnn/tacnn_ntu60_xsub_joint.py) | 88.65, 88.53, 88.76, 88.49, 88.21 | 88.53±0.19 | 88.8
Dynamic GCN | [dgcn_65e_ntu60_xsub_joint.py](configs/dgcn/dgcn_65e_ntu60_xsub_joint.py) | 88.49, 89.03, 88.87, 88.88, 89.23 | 88.90±0.24 | 89.2

<sup>2</sup> Seed = {0, 1, 2, 3, 4}

- NTU RGB+D XView

Model | Config | Our Acc<sup>3</sup> | Paper Acc
------|--------|---------------------|----------
Two-Stream CNN | [tscnn_ntu60_xview_joint.py](configs/tscnn/tscnn_ntu60_xview_joint.py) | 90.23 | 89.3
HCN | [hcn_ntu60_xview_joint.py](configs/hcn/hcn_ntu60_xview_joint.py) | 92.35 | 91.1
Ta-CNN | [tacnn_ntu60_xview_joint.py](configs/tacnn/tacnn_ntu60_xview_joint.py) | 93.91 | -
Dynamic GCN | [dgcn_65e_ntu60_xview_joint.py](configs/dgcn/dgcn_65e_ntu60_xview_joint.py) | 94.23 | -

<sup>3</sup> Seed = 0

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

```BibTeX
@inproceedings{li2017skeleton,
  title={Skeleton-based Action Recognition with Convolutional Neural Networks},
  author={Li, Chao and Zhong, Qiaoyong and Xie, Di and Pu, Shiliang},
  booktitle={2017 IEEE International Conference on Multimedia \& Expo Workshops},
  pages={597--600},
  year={2017}
}

@inproceedings{li2018co-occurrence,
  title={Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation},
  author={Li, Chao and Zhong, Qiaoyong and Xie, Di and Pu, Shiliang},
  booktitle={Proceedings of the 27th International Joint Conference on Artificial Intelligence},
  pages={786--792},
  year={2018}
}

@inproceedings{ye2020dynamic,
  title={Dynamic GCN: Context-enriched Topology Learning for Skeleton-based Action Recognition},
  author={Ye, Fanfan and Pu, Shiliang and Zhong, Qiaoyong and Li, Chao and Xie, Di and Tang, Huiming},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={55--63},
  year={2020}
}

@inproceedings{xu2022topology,
  title={Topology-aware Convolutional Neural Network for Efficient Skeleton-based Action Recognition},
  author={Xu, Kailin and Ye, Fanfan and Zhong, Qiaoyong and Xie, Di},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

## Acknowledgement

SkelAct heavily depends on [MMAction2](https://github.com/open-mmlab/mmaction2/). We appreciate all contributors to the excellent framework.

