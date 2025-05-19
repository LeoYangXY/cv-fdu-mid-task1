项目结构：
resnet101_finetune/
├── data/101_ObjectCategories/      # 数据集根目录
├── hp_search_results/              # 超参搜索结果
│   └── results.txt                 # 实验记录文件
├── runs/                           # TensorBoard日志
├── saved_model/                    # 模型保存目录
├── finetune_resnet.py              # 主训练脚本
├── hyperparam_search.py            # 超参搜索脚本
└── README.md                       # 本文件

resnet101_finetune/
├── finetune_resnet.py        # 主训练脚本，包含模型定义和训练逻辑
├── hp_search.py              # 超参数搜索实现脚本
├── hp_search_results/        # 搜索结果目录
│   └── results.txt           # 所有实验结果的记录文件
├── saved_model/              # 保存的最佳模型目录
└── README.md                 # 项目说明文档

快速入门
1. 环境准备
pip install torch torchvision tqdm tensorboard pillow
2. 单次训练
python finetune_resnet.py \
    --data_path ./data/101_ObjectCategories \
    --batch_size 32 \
    --num_epochs 30 \
    --lr_fc 0.01 \
    --lr_other 0.0001 \
    --pretrained True
3. 启动超参搜索
python hyperparam_search.py
实验结果
所有实验结果自动记录在：
hp_search_results/results.txt
