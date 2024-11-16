from . import *

class CelebA(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)                              # 数据集根目录
        self.transform = transform                          # 图片预处理
        
        with open("datasets/CelebA.pkl", "rb") as celeba:
            self.persons = pickle.load(celeba)              # 所有的人

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, i):
        # 选人
        anchor_person = self.persons[i+1]                                           # 选取编号为 i+1 的人
        negative_person = anchor_person
        while negative_person == anchor_person:
            negative_person = self.persons[random.randint(1, len(self.persons))]    # 随机选另一个人（除编号为 i+1 的人）
        
        # 选人的图片
        anchor_path = self.root / random.choice(anchor_person)         # 在编号为 i+1 的人中随机选取一张人脸作为锚点
        positive_path = self.root / random.choice(anchor_person)       # 在编号为 i+1 的人随机选取一张人脸作为正例
        negative_path = self.root / random.choice(negative_person)     # 在另一个人（非编号为 i+1 的人）中随机选取一张人脸作为反例
        
        # 打开图片
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        # 图片预处理
        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)
            
        return anchor, positive, negative

class LFW(torch.utils.data.Dataset):
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