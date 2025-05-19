import os
import torch
from finetune_resnet import datasetOf101Object, build_model, train_net
import argparse

# 设置搜索参数范围
LR_FC_LIST = [0.1, 0.01, 0.001]     # 全连接层学习率
NUM_EPOCHS_LIST = [20, 30, 50]       # 训练轮数

# 固定其他参数（可调整）
FIXED_ARGS = {
    "data_path": "/workspace/cv_midterm/data/101_ObjectCategories",
    "batch_size": 32,
    "lr_other": 0.0001,              # 其他层学习率固定
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "step_size": 5,
    "gamma": 0.1,
    "log_interval": 10,
    "pretrained": True
}

def main():
    # 创建结果保存目录
    os.makedirs("hp_search_results", exist_ok=True)
    
    # 保存所有实验结果
    results_log = []

    # 遍历所有参数组合
    for lr_fc in LR_FC_LIST:
        for num_epochs in NUM_EPOCHS_LIST:
            print(f"\n=== 开始实验: lr_fc={lr_fc}, num_epochs={num_epochs} ===")

            # 动态更新参数
            args = argparse.Namespace(**{
                **FIXED_ARGS,
                "lr_fc": lr_fc,
                "num_epochs": num_epochs
            })

            # 数据加载
            train_dataset = datasetOf101Object(args.data_path, is_train=True)
            test_dataset = datasetOf101Object(args.data_path, is_train=False)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
            )

            # 模型构建
            model = build_model(is_pretrained_flag=args.pretrained)

            # 训练模型
            results = train_net(
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                model=model,
                num_epochs=args.num_epochs,
                lr_fc=args.lr_fc,
                lr_other=args.lr_other,
                batch_size=args.batch_size,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                step_size=args.step_size,
                gamma=args.gamma,
                log_interval=args.log_interval,
                is_pretrained=args.pretrained
            )

            # 保存结果
            result_str = f"lr={lr_fc}, epochs={num_epochs}: " \
                        f"Test Acc={results['best_acc']:.2f}% | " \
                        f"Train Loss={results['train_loss']:.4f}, Test Loss={results['test_loss']:.4f}"
            results_log.append(result_str)
            
            # 写入文件
            with open("hp_search_results/results.txt", "a") as f:
                f.write(result_str + "\n")

    # 打印最终结果摘要
    print("\n=== 实验结果摘要 ===")
    for line in results_log:
        print(line)

if __name__ == "__main__":
    main()