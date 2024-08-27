from . import *

class Faces(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)                                          # 数据集根目录
        self.transform = transform                                      # 图片预处理
        self.persons = [entry.name for entry in self.root.iterdir()]    # 所有的人

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, i):
        # 选人
        anchor_person = self.persons[i]                     # 选取第 i 个人
        negative_person = anchor_person
        while negative_person == anchor_person:
            negative_person = random.choice(self.persons)   # 随机选另一个人（除第 i 个人）
        
        # 选人的图片
        anchor_paths = list((self.root / anchor_person).iterdir())
        anchor_path = random.choice(anchor_paths)                                       # 在第 i 个人中随机选取一张人脸作为锚点
        positive_path = random.choice(anchor_paths)                                     # 在第 i 个人中随机选取一张人脸作为正例
        negative_path = random.choice(list((self.root / negative_person).iterdir()))    # 在另一个人（非第 i 个人）中随机选取一张人脸作为反例
        
        # 打开图片
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        # 图片预处理
        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)
            
        return anchor, positive, negative
    
class Faces_PR(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)                                          # 数据集根目录
        self.transform = transform                                      # 图片预处理
        self.persons = [entry.name for entry in self.root.iterdir()]    # 所有的人

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, i):
        # 选人
        anchor_person = self.persons[i]                     # 选取第 i 个人
        negative1_person = anchor_person
        negative2_person = anchor_person
        while negative1_person == anchor_person:
            negative1_person = random.choice(self.persons)   # 随机选另一个人（除第 i 个人）
        while negative2_person == anchor_person or negative2_person == negative1_person:
            negative2_person = random.choice(self.persons)   # 随机选另一个人（除第 i 个人）
        
        # 选人的图片
        anchor_paths = list((self.root / anchor_person).iterdir())
        anchor_path = random.choice(anchor_paths)                                       # 在第 i 个人中随机选取一张人脸作为锚点
        positive_path = random.choice(anchor_paths)                                     # 在第 i 个人中随机选取一张人脸作为正例
        negative1_path = random.choice(list((self.root / negative1_person).iterdir()))    # 在另一个人（非第 i 个人）中随机选取一张人脸作为反例
        negative2_path = random.choice(list((self.root / negative2_person).iterdir()))    # 在另一个人（非第 i 个人）中随机选取一张人脸作为反例
        
        # 打开图片
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative1 = Image.open(negative1_path).convert('RGB')
        negative2 = Image.open(negative2_path).convert('RGB')

        # 图片预处理
        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative1 = self.transform(negative1)
        negative2 = self.transform(negative2)
            
        return anchor, positive, (negative1, negative2)