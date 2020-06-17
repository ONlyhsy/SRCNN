from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset,DataLoader
import cv2
import os
import re
class TrainDataset(Dataset):
    '''root_dir是包含图片的文件目录'''
    def __init__(self,root_dir):
        super(TrainDataset, self).__init__()
        self.names = [name for name in os.listdir(root_dir) if 'm' in name]
        self.root=root_dir
    def __getitem__(self, idx):
        img_name=self.names[idx]
        finder=re.compile('[0-9]*')
        img_labelname=finder.search(img_name)
        img_labelname=img_labelname.group()+'.png'
        img_image_path=self.root+f'\{img_name}'
        img_label_path=self.root+f'\{img_labelname}'
        img_numpy=cv2.imread(img_image_path)
        img_label=cv2.imread(img_label_path)
        img_image_tensor=transforms.ToTensor()(img_numpy)
        img_label_tensor=transforms.ToTensor()(img_label)
        img={'image':img_image_tensor,'label':img_label_tensor}
        return img

    def __len__(self):
        return len(self.names)
class EvalDataset(Dataset):#TODO:以后这里要和训练集区分一下
    '''root_dir是包含图片的文件目录'''
    def __init__(self,root_dir):
        super(EvalDataset, self).__init__()
        self.names = [name for name in os.listdir(root_dir) if 'm' in name]
        self.root=root_dir
    def __getitem__(self, idx):
        img_name=self.names[idx]
        finder=re.compile('[0-9]*')
        img_labelname=finder.search(img_name)
        img_labelname=img_labelname.group()+'.png'
        img_image_path=self.root+f'\{img_name}'
        img_label_path=self.root+f'\{img_labelname}'
        img_numpy=cv2.imread(img_image_path)
        img_label=cv2.imread(img_label_path)
        img_image_tensor=transforms.ToTensor()(img_numpy)
        img_label_tensor=transforms.ToTensor()(img_label)
        img={'image':img_image_tensor,'label':img_label_tensor}
        return img

    def __len__(self):
        return len(self.names)
if __name__ == '__main__':
    root=os.getcwd()+'\data_img\data_figure'
    data = TrainDataset(root)  # 初始化类，设置数据集所在路径以及变换
    dataloader = DataLoader(data, batch_size=2, shuffle=True)  # 使用DataLoader加载数据
    for i_batch, batch_data in enumerate(dataloader):
        print(i_batch)  # 打印batch编号
        print(batch_data['image'].size())  # 打印该batch里面图片的大小
        print(batch_data['label'])  # 打印该batch里面图片的标签