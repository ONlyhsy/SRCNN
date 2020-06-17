from Srcnn.model import SRCNN
from torch import optim,nn
import os
import copy

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from Srcnn.datasets import TrainDataset,EvalDataset
from Srcnn.utils import AverageMeter,calc_psnr
import re
############
#TODO:以后这里会写一些命令行文件操作
############
if __name__ == '__main__':
    num_epochs=int(input('输入训练轮数：'))
    batch_size=int(input('输入批处理数量'))
    learning_rate=float(input('输入学习率'))
    root=os.getcwd()
    path_save='\data_img\data_figure'
    outputs_dir=root

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=SRCNN().to(device)
    criterion = nn.MSELoss()#损失函数
    optimizer = optim.Adam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.conv3.parameters(), }], lr=learning_rate)#优化方案
    train_dataset = TrainDataset(root+path_save)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(root+path_save)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    for epoch in range(num_epochs):
            model.train()
            epoch_losses = AverageMeter()

            with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
                t.set_description('epoch:{}/{}'.format(epoch, num_epochs - 1))

                for data in train_dataloader:
                    # inputs, labels = data
                    inputs=data['image']
                    labels=data['label']

                    # print(labels)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    preds = model(inputs)
                    loss = criterion(preds, labels)

                    epoch_losses.update(loss.item(), len(inputs))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                    t.update(len(inputs))

            torch.save(model.state_dict(), os.path.join(outputs_dir, 'model.pkl'))

            model.eval()
            epoch_psnr = AverageMeter()

            for data in eval_dataloader:
                # inputs, labels = data
                inputs = data['image']
                labels = data['label']
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)

                epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

            print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(outputs_dir, 'best.pkl'))
