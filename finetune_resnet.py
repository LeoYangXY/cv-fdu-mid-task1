import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
import argparse  # 修改点1：添加命令行解析库
from tqdm import tqdm  # 修改点2：添加进度条
import datetime

#==========================数据读取==========================
class datasetOf101Object(Dataset):
    #注意我们要在这里实现train-val-test的划分，因为只提供给了我们一个数据集
    #然后Caltech-101 某些类别仅有 31~50 张图像，若再分验证集会进一步减少训练数据，影响模型性能，所以我们不设定val
    #按照官方论文：每个类别 30 张训练，其余测试
    def __init__(self,path,is_train=True,train_nums=30):
        self.path=path
        self.transforms=transforms.Compose([
                    transforms.Resize(256),#转化大小
                    transforms.CenterCrop(224),#裁剪出中心部分
                    transforms.ToTensor(),#转化为矩阵
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#根据经验值归一化
                ])
        # 获取所有类别并排除 BACKGROUND_Google（官方的操作）
        self.classes = sorted([cls for cls in os.listdir(path) if cls != "BACKGROUND_Google"])#os.listdir就是展开一层
        
        self.class2idx=dict()
        self.idx2class=dict()
        for idx,cur_class in enumerate(self.classes):
            self.class2idx[cur_class]=idx
            self.idx2class[idx]=cur_class
        
        self.samples = []
        for cur_class in self.classes:
            cur_path=os.path.join(path,cur_class)#使用os.path.join进行路径拼接
            all_images = sorted([img for img in os.listdir(cur_path)])#使用sorted，确保可重复性
            selected_images = all_images[:train_nums] if is_train else all_images[train_nums:]# 按标准划分：前train_nums张为训练集，其余为测试集
            for img_name in selected_images:
                img_path = os.path.join(cur_path, img_name)
                self.samples.append((img_path, self.class2idx[cur_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        img_path,idx_of_class=self.samples[idx]
        img=Image.open(img_path).convert("RGB")#使用Image库把image_path的图片打开为图片矩阵，然后使用convert确保其为RGB模式
        img=self.transforms(img)
        return img,idx_of_class


#========================导入模型=====================
def build_model(is_pretrained_flag=True):
   
    model = models.resnet18(pretrained=is_pretrained_flag)#加载预训练模型,官方对于resnet18就是在imageNet上训练的
    
    
    #使用print(model)这样子可以查看model的每一层的具体信息
    #然后我们可以使用如下代码遍历模型所有层：
    # # 打印所有子模块的名称和结构
    # for name, layer in model.named_children():
    #     print(f"Layer name: {name}, Type: {type(layer)}")
    #     # 如果是嵌套结构（如Sequential），可以进一步展开
    #     if isinstance(layer, nn.Sequential):
    #         for sub_name, sub_layer in layer.named_children():
    #             print(f"  Sub-layer: {sub_name}, Type: {type(sub_layer)}")


    #拼接新的层
    num_features = model.fc.in_features  # 获取输入维度（512）
    model.fc = nn.Linear(num_features, 101)  # 替换为101输出

    return model

#============================不同超参数训练=====================
def train_net(train_dataloader, test_dataloader, model, num_epochs=30, 
              lr_fc=0.01, lr_other=0.0001,  # 修改点3：添加分层学习率参数
              batch_size=32, momentum=0.9,    # 修改点4：添加更多超参数
              weight_decay=1e-4, step_size=5, 
              gamma=0.1, log_interval=10,
              is_pretrained=True):     # 修改点5：添加调度器参数
    # 初始化TensorBoard Writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{current_time}"
    writer = SummaryWriter(log_dir=log_dir)

    #一定一定注意：将模型和后续训练时候的数据都显式地挪到GPU！！！！！！！！！！！！！！
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #损失函数的选择：使用nn.CrossEntropyLoss()还是nn.NLLLoss()
    #如果使用nn.CrossEntropyLoss()，我们需要：
    #输入：模型输出的原始分数（未经过softmax），形状为 (batch_size, num_classes)
    #目标：类别索引（int），形状为 (batch_size, )
    criterion = nn.CrossEntropyLoss()

    # 优化器的选择：SGD或者Adam，不同数据集和任务不同
    # Adam/AdamW：适合快速实验和默认场景（80%情况）
    # SGD + 调度器：需要更多调参但可能达到更高精度
    optimizer = optim.SGD([
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': lr_other},
        {'params': model.fc.parameters(), 'lr': lr_fc}     
    ], 
    momentum=momentum,#Momentum（动量）想象一个小球从山顶滚下山：
                 #没有动量：小球每一步只根据当前坡度决定方向，如果山坡坑坑洼洼（梯度不稳定），小球会来回震荡，下山很慢；
                 #有动量：小球会记住之前的速度方向，即使遇到小坑（噪声梯度）也能保持惯性冲过去，下山更快更稳。
    weight_decay=weight_decay)#L2正则化中的惩罚项的系数

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_acc = 0  # 修改点7：记录最佳准确率
    best_stats = {}  # 修改点8：记录最佳模型状态

    # 修改点9：添加tqdm进度条
    epoch_loop = tqdm(range(num_epochs), desc='Total Training Progress', position=0)
    
    for i in epoch_loop:
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # 修改点10：添加进度条
        batch_loop = tqdm(train_dataloader, desc=f'Epoch {i+1}/{num_epochs}', leave=False)
        for batch_idx, (inputs, labels) in enumerate(batch_loop):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()#每个batch独立计算损失然后做梯度下降，因此我们需要每个batch进行zero_grad，而不是每个epoch进行zero_grad
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #下面的.item和.detach操作是为了把torch类型转换回python类型，断开计算图，避免内存爆炸
            #当然，这里没有正确性问题，因为其后面没有写loss.backward，optimizer.step()去进行实质上的网络更新
            running_loss += loss.item()
            _, predicted = torch.max(outputs.detach(), dim=1)#torch.max返回的是最大值的数值+索引
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 每N个batch记录一次训练loss
            if batch_idx % log_interval == log_interval - 1:
                avg_loss = running_loss / log_interval
                writer.add_scalar('Training Loss (per batch)', 
                                 avg_loss, 
                                 i * len(train_dataloader) + batch_idx)
                running_loss = 0.0
                batch_loop.set_postfix(loss=avg_loss)

        scheduler.step()
        train_loss = running_loss / len(train_dataloader)
        train_acc = 100 * train_correct / train_total
        

        #==========================测试阶段================================
        model.eval()#只是打开dropout，和batchNorm，并没有禁用梯度。禁用梯度需要下面手动再写
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(test_dataloader)
        test_acc = 100 * test_correct / test_total
        
        # 修改点11：保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            
            # 根据是否使用预训练模型确定前缀
            model_prefix = "pretrained" if is_pretrained else "scratch"
            model_filename = f"saved_model/{model_prefix}_best_epoch{i+1}_acc{test_acc:.1f}.pth"
            
            best_stats = {
                'epoch': i+1,
                'model_name': model_filename,  # 使用完整路径
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'state_dict': model.state_dict().copy()#这里才是复制模型的所有层的权重
            }
            
            
            torch.save({
                'state_dict': best_stats['state_dict'],
                'best_stats': best_stats,
                'hyperparams': {
                    'lr_fc': lr_fc,
                    'batch_size': batch_size,
                    'pretrained': is_pretrained
                }
            }, model_filename)  
        
        # 记录epoch级别的指标
        writer.add_scalar('Loss/Train', train_loss, i)
        writer.add_scalar('Loss/Test', test_loss, i)
        writer.add_scalar('Accuracy/Train', train_acc, i)
        writer.add_scalar('Accuracy/Test', test_acc, i)
        
        # 打印结果
        epoch_loop.write('-' * 50)
        epoch_loop.write(f'Epoch {i+1}/{num_epochs}')
        epoch_loop.write(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        epoch_loop.write(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # 新增打印语句（用于手动记录数据）
        print(f"Epoch {i+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")
    
    writer.close()
    
    # 修改点12：返回更多训练信息
    return {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'best_acc': best_acc,
        'best_stats': best_stats
    }

#============================主程序=====================
if __name__ == '__main__':
    # 修改点13：添加命令行参数解析
    parser = argparse.ArgumentParser(description='Caltech-101分类任务训练脚本')
    parser.add_argument('--data_path', type=str, default='/workspace/cv_midterm/data/101_ObjectCategories',
                       help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='训练和测试的batch大小')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='训练的总epoch数')
    parser.add_argument('--lr_fc', type=float, default=0.01,
                       help='全连接层学习率')
    parser.add_argument('--lr_other', type=float, default=0.0001,
                       help='其他层学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='优化器动量参数')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减(L2正则化)系数')
    parser.add_argument('--step_size', type=int, default=5,
                       help='学习率调度器步长')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='学习率衰减系数')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='每多少个batch记录一次训练loss')
    parser.add_argument('--pretrained', 
                        type=lambda x: x.lower() == 'true', default=False)
                        #不然命令行解析的时候，任何字符都会被解析为bool的True
    
    args = parser.parse_args()
    
    # 数据加载
    train_dataset = datasetOf101Object(args.data_path, is_train=True)
    test_dataset = datasetOf101Object(args.data_path, is_train=False)
    
    # 启用多进程数据加载
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # 适当增大batch size
        shuffle=True,
        num_workers=4,  # CPU核心数的一半到3/4
        pin_memory=True  # 启用锁页内存
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,  # 测试时可使用更大batch
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    
    # 模型构建
    model = build_model(is_pretrained_flag=args.pretrained)
    
    # 训练过程
    print("=== 开始训练 ===")
    print(f"超参数配置: batch_size={args.batch_size}, num_epochs={args.num_epochs}")
    print(f"学习率: fc_layer={args.lr_fc}, other_layers={args.lr_other}")
    print(f"优化器参数: momentum={args.momentum}, weight_decay={args.weight_decay}")
    print(f"学习率调度: step_size={args.step_size}, gamma={args.gamma}")
    print(f"使用预训练: {args.pretrained}")
    
    results = train_net(
        train_dataloader, test_dataloader, model,
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
    
    # 输出最终结果
    print("\n=== 训练完成 ===")
    print(f"最佳测试准确率: {results['best_acc']:.2f}% (epoch {results['best_stats']['epoch']})")
    print(f"对应训练准确率: {results['best_stats']['train_acc']:.2f}%")
    print(f"模型已保存为: best_model_epoch{results['best_stats']['epoch']}.pth")