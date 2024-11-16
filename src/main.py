from ResNet18 import *
from ResNet18 import Network, Dataset

# 图片预处理
transform_train = transforms.Compose([
    transforms.CenterCrop([178, 178]),                      # 1. 中心裁切 178x178
    transforms.ColorJitter(0.5, 0.3, 0.2, 0.2),             # 2. 随机调整亮度、对比度、饱和度、色相
    transforms.RandomRotation(10),                          # 3. 随机旋转图片 10°
    transforms.ToTensor(),                                  # 4. 转换为 PyTorch 张量
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 5. 批归一化
])
transform_test = transforms.Compose([
    transforms.CenterCrop([178, 178]),                      # 1. 中心裁切 178x178
    transforms.ToTensor(),                                  # 2. 转换为 PyTorch 张量
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 3. 批归一化
])
transform_getfaces = transforms.Compose([
    transforms.ToPILImage(),                                # 1. 转换为 PIL 图片
    transforms.Resize([178, 178]),                          # 2. 缩放 178x178
    transforms.ToTensor(),                                  # 3. 转换为 PyTorch 张量
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 4. 批归一化
])

# 人脸数据集
def FaceDataset(camera_id, batch_size, total_batch):
    isCapture = False
    console.print("Press [bold yellow]y[/] to start capturing faces")
    
    # 获得相机句柄
    camera = cv2.VideoCapture(camera_id)
    # 加载人脸检测模型
    classifer = cv2.CascadeClassifier("E:\Models\OpenCV\lbpcascades\lbpcascade_frontalface_improved.xml")
    
    dataset = []
    # 拍摄一个（batch）人脸数据集
    while len(dataset) != total_batch:
        batch = torch.tensor([])
        # 拍摄一批（batch_size）人脸
        while batch.shape[0] != batch_size:
            # 拍摄一帧图片
            _, frame = camera.read()
            # 中心裁切最大正方形
            h, w, _ = frame.shape
            a = min(h, w)
            cropped = frame[(h-a)//2:(h+a)//2, (w-a)//2:(w+a)//2]
            # 转化为灰度图
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            # 检测是否有人脸
            for _ in classifer.detectMultiScale(gray, 1.3, 4, minSize=(100, 100)):
                if isCapture:
                    face = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB) # 将 BGR 转换为 RGB
                    face = transform_getfaces(face).unsqueeze(0)    # 图片预处理
                    batch = torch.cat((batch, face))                # 将张量拼接到 batch
                
            # 显示获取的人脸批数
            cv2.putText(cropped, f"{len(dataset)}/{total_batch}", (10, 50), cv2.QT_FONT_NORMAL, 1, (0, 255, 255))
            # 显示预览窗口
            cv2.imshow("GetFaces", cropped)
            if cv2.waitKey(1) & 0xFF == ord("y"):
                isCapture = True
        
        # 将一批人脸添加到数据集
        dataset.append(batch)
        
    # 销毁所有窗口
    cv2.destroyAllWindows()
    
    return dataset

def Log(text, end="\n"):
    console.print(text + " "*(os.get_terminal_size().columns-len(text)-1), end=end)

def Train(datasets_root, checkpoints_root, total_epoch, learning_rate, batch_size, gamma):
    # 加载 CelebA、LFW 数据集
    faces_train = Dataset.CelebA(f"{datasets_root}/CelebA", transform_train)
    faces_test = Dataset.LFW(f"{datasets_root}/LFW_2", transform_test)
    faces_train = DataLoader(faces_train, batch_size, True, pin_memory=True, drop_last=True, num_workers=os.cpu_count())
    faces_test = DataLoader(faces_test, batch_size, False, pin_memory=True, drop_last=True, num_workers=os.cpu_count())

    model = Network.ResNet18(128).to(device, non_blocking=True)             # ResNet18 模型
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)      # Adam 优化器
    criterion = torch.nn.TripletMarginLoss(3.0, reduction="sum")            # 三元损失
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma) # 自动学习率

    # 训练 total_epoch 次
    for epoch in range(total_epoch):
        # 错误集初始化
        errors = []
        
        # 指标初始化（损失值、准确率）
        avg_loss_train = 0
        avg_loss_test = 0
        accuracy_train = 0
        accuracy_test = 0
        
        # 训练 CelebA 数据集
        model.train()
        for step, (anchor, positive, negative) in enumerate(faces_train):
            # 前向传播
            anchor_out = model(anchor.to(device, non_blocking=True))
            positive_out = model(positive.to(device, non_blocking=True))
            negative_out = model(negative.to(device, non_blocking=True))
            loss = criterion(anchor_out, positive_out, negative_out)
            
            # 反向传播
            optimizer.zero_grad()   # 清空上一步的梯度值
            loss.backward()         # 求权重、偏置值的偏导数
            optimizer.step()        # 应用偏导数
            
            with torch.no_grad():
                # 脱离计算图
                loss = loss.detach().item()
                
                # 记录错误项
                if epoch + 1 >= 10 and loss >= 10:
                    errors.append((anchor, positive, negative))
                
                # 指标更新（损失值、准确率）
                avg_loss_train += (loss-avg_loss_train)/(step+1)  # 实时平均损失值
                if loss == 0:
                    accuracy_train += 1/len(faces_train)          # 实时准确率
            
            # 指标显示（损失值、准确率）
            Log(f"[{epoch+1}/{total_epoch} {step+1}/{len(faces_train)}] [Train CelebA] AvgLoss: {avg_loss_train} Loss: {loss} Accuracy: {accuracy_train}", "\r")
            
        # 训练错误集
        random.shuffle(errors)
        for step, (anchor, positive, negative) in enumerate(errors):
            # 前向传播
            anchor_out = model(anchor.to(device, non_blocking=True))
            positive_out = model(positive.to(device, non_blocking=True))
            negative_out = model(negative.to(device, non_blocking=True))
            loss = criterion(anchor_out, positive_out, negative_out)
            
            # 反向传播
            optimizer.zero_grad()   # 清空上一步的梯度值
            loss.backward()         # 求权重、偏置值的偏导数
            optimizer.step()        # 应用偏导数
            
            with torch.no_grad():
                # 脱离计算图
                loss = loss.detach().item()
            
            # 指标显示（损失值）
            Log(f"[{epoch+1}/{total_epoch} {step+1}/{len(errors)}] [Train Errors] Loss: {loss}", "\r")
        
        # 测试 LFW 数据集
        model.eval()
        # 不求偏导数
        with torch.no_grad():
            for step, (anchor, positive, negative) in enumerate(faces_test):
                # 前向传播
                anchor_out = model(anchor.to(device, non_blocking=True))
                positive_out = model(positive.to(device, non_blocking=True))
                negative_out = model(negative.to(device, non_blocking=True))
                loss = criterion(anchor_out, positive_out, negative_out)
                
                # 脱离计算图
                loss = loss.detach().item()
                
                # 指标更新（损失值、准确率）
                avg_loss_test += (loss-avg_loss_test)/(step+1)    # 实时平均损失值
                if loss == 0:
                    accuracy_test += 1/len(faces_test)            # 实时准确率
                    
                # 指标显示（损失值、准确率）
                Log(f"[{epoch+1}/{total_epoch} {step+1}/{len(faces_test)}] [Test LFW] AvgLoss: {avg_loss_test} Loss: {loss} Accuracy: {accuracy_test}", "\r")
        
        # 调整学习率
        lr_scheduler.step()
        
        # 保存检查点
        checkpoint = {
            "Epoch": epoch+1,                           # 当前 迭代次数
            "AvgLoss": {
                "Train": avg_loss_train,                # 当前 训练 平均损失
                "Test": avg_loss_test                   # 当前 测试 平均损失
            },
            "Accuracy": {
                "Train": accuracy_train,                # 当前 训练 准确率
                "Test": accuracy_test                   # 当前 测试 准确率
            },
            "Model": model.state_dict(),                # 当前 模型 权重、偏置值
            "Optimizer": optimizer.state_dict(),        # 当前 优化器 参数
            "LR_Scheduler": lr_scheduler.state_dict()   # 当前 自动学习率 参数
        }
        torch.save(checkpoint, f"{checkpoints_root}/checkpoint.{epoch+1}.pth")
        
        # 指标显示（损失值、准确率）
        Log(f"[{epoch+1}/{total_epoch}] AvgLossTrain: {avg_loss_train} AvgLossTest: {avg_loss_test} AccuracyTrain: {accuracy_train} AccuracyTest: {accuracy_test}")

@torch.no_grad()
def Register(camera_id, checkpoints_path, username):
    # 加载人脸数据
    a_dataset = FaceDataset(camera_id, 8, 8)
    
    model = Network.ResNet18(128).to(device, non_blocking=True)     # ResNet18 模型
    
    # 加载模型权重
    model.load_state_dict(torch.load(f"{checkpoints_path}/checkpoint.35.pth", device)["Model"])
    
    # 测试
    model.eval()
    a_dataset_out = []
    for a_batch in a_dataset:
        # 前向传播
        a_batch_out = model(a_batch.to(device, non_blocking=True))
        
        a_dataset_out.append(a_batch_out)
            
    # 保存人脸向量
    torch.save(a_dataset_out, f"Users/{username}.pt")

@torch.no_grad()
def Verify(camera_id, datasets_root, checkpoints_path, username):
    # 加载人脸数据
    p_dataset = FaceDataset(camera_id, 8, 8)
    n_dataset = [torch.stack([transform_test(Image.open(negative_face)) for negative_face in random.choices(list(os.scandir(f"{datasets_root}/CelebA_1")), k=8)]) for _ in range(8)]
    
    model = Network.ResNet18(128).to(device, non_blocking=True)     # ResNet18 模型
    criterion = torch.nn.TripletMarginLoss(0.0, reduction="sum")    # 三元损失
        
    # 加载模型权重
    model.load_state_dict(torch.load(f"{checkpoints_path}/checkpoint.35.pth", device)["Model"])
    
    # 测试
    model.eval()
    # 前向传播
    a_dataset_out = torch.load(f"Users/{username}.pt", device)
    p_dataset_out = [model(p_batch.to(device, non_blocking=True)) for p_batch in p_dataset]
    n_dataset_out = [model(n_batch.to(device, non_blocking=True)) for n_batch in n_dataset]
    losses = []
    for a_batch_out in a_dataset_out:
        for p_batch_out in p_dataset_out:
            for n_batch_out in n_dataset_out:
                    losses.append(criterion(a_batch_out, p_batch_out, n_batch_out).item())
            
    console.print(not bool(torch.tensor(losses).median().item()))

@group()
def main():
    pass

@main.command()
@option("--datasets-root", "-d", help="Datasets root")
@option("--checkpoints-root", "-c", default="./checkpoints", help="Checkpoints root")
@option("--total-epoch", "-e", default=50, help="Total epoch")
@option("--learning-rate", "-r", default=0.01, help="Learning rate")
@option("--batch-size", "-s", default=8, help="Batch size")
@option("--gamma", "-g", default=0.95, help="The gamma of ExponentialLR")
def train(datasets_root, checkpoints_root, total_epoch, learning_rate, batch_size, gamma):
    Train(datasets_root, checkpoints_root, total_epoch, learning_rate, batch_size, gamma)
    
@main.command()
@option("--camera-id", "-i", default=0, help="Camera ID")
@option("--checkpoints-path", "-c", default="./checkpoints", help="Checkpoints path")
@option("--username", "-n", help="Username")
def register(camera_id, checkpoints_path, username):
    Register(camera_id, checkpoints_path, username)
    
@main.command()
@option("--camera-id", "-i", default=0, help="Camera ID")
@option("--datasets-root", "-d", help="Datasets root")
@option("--checkpoints-path", "-c", default="./checkpoints", help="Checkpoints path")
@option("--username", "-n", help="Username")
def verify(camera_id, datasets_root, checkpoints_path, username):
    Verify(camera_id, datasets_root, checkpoints_path, username)

if __name__ == "__main__":
    main()