import cv2
import os
import h5py
from tqdm import tqdm
import csv
#pre_do这个模块主要时进行一些图像预处理以及生成可以用来训练的数据集
def bigger(filename,times_bigger=4):
    '''这个函数是用来放大图像填充像素到原来清晰的大小，默认扩大高和宽为2倍
        拥有两个参数，filename文件名需要str类型
                    times_bigger默认为2
    '''
    img = cv2.imread(filename)
    h = img.shape[0]
    w = img.shape[1]
    newimg = cv2.resize(img,(w*times_bigger,h*times_bigger),interpolation=cv2.INTER_CUBIC)
    return newimg
def shrink(filename,times_shrink=4):
    '''这个函数是用来缩小图像削减像素到模糊的大小，默认缩小高和宽为2倍（奇数数值缩小2倍会进行取整操作）
            拥有两个参数，filename文件名需要str类型
                        times_bigger默认为2
    '''
    img = cv2.imread(filename)
    h = img.shape[0]
    w = img.shape[1]
    newimg = cv2.resize(img,(int(w/times_shrink),int(h/times_shrink)))
    return newimg


def test_ShrinkandBigger():
    '''这个函数用来测试图片经过预处理之后是否变得模糊不清'''
    fn = 'xdu.jpg'#这个xdu.jpg是放在SRCNN根目录的一张图片
    img_shrink2 = shrink(fn)
    cv2.imwrite('img_shrink.jpg', img_shrink2)
    img_shrink2_bigger2 = bigger('img_shrink.jpg')
    cv2.imshow('shrink2', img_shrink2)#缩小4倍的图片
    cv2.imshow('shrink_bigger2', img_shrink2_bigger2)#扩大4倍的图片
    img = cv2.imread(fn)
    cv2.imshow('origin', img)#原图

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():#TODO:以后要把一部分图像作为训练集，一部分作为测试集
    '''主函数，文件中的图片，并把处理好后的图片命名清楚然后一对一对地放进data_img文件夹中的子文件夹
        data_img中应该是有一个存有所有清晰图片的文件夹data_source
        处理好的模糊图像用01_m的形式存储在data_source里
        然后最好可以把tensor数据用csv格式存放在data_csv文件里，这样可以让这个过程更加透明
        一共有12个图像文件!!!!!!
    '''
    path_img='\data_img\data_source'
    path_save='\data_img\data_figure'
    root=os.getcwd()
    os.chdir(root+path_img)#修改工作路径到data_source
    print('12个图像的数据集处理中...')
    for i in tqdm(range(12)):
        j=i+1
        name=f'{j}.png'
        imgshrink2=shrink(name)
        cv2.imwrite('img_shrink.png',imgshrink2)
        imgblur=bigger('img_shrink.png')
        cv2.imwrite(f'{j}_m.png',imgblur)
        os.remove('img_shrink.png')#图像处理完毕

    print('\n现在开始载入模糊图像数据和清晰图像数据到文件夹data_csv...')
    path_csv = '\data_img\data_csv'
    for name in tqdm(os.listdir(root+path_img)):
        os.chdir(root+path_img)
        img=cv2.imread(name)
        os.chdir(root + path_csv)
        index=name.rfind('.')
        img_name=name[:index]
        with open(f'{img_name}.csv','w') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(['R','G','B'])
            writer.writerow(img)
    print('\n载入图像数据成功，文件已经建立！')

    print('\n现在开始处理各图像大小，确保所有图像大小为128*128')
    os.chdir(root + path_save)  # 改变工作路径到存储图像的路径
    for name in tqdm(os.listdir(root+path_img)):
        img_origin_path=root+path_img+f'\{name}'
        img_reader=cv2.imread(img_origin_path)
        if img_reader.shape>=(512,512,3):
            os.chdir(root+path_img)
            img_save=shrink(f'{name}',times_shrink=2)
            os.chdir(root+path_save)
            cv2.imwrite(f'{name}',img_save)
        else:
            cv2.imwrite(f'{name}',img_reader)


    print('\n清除残余文件中...')
    os.chdir(root+path_csv)
    for item in tqdm(os.listdir(root+path_csv)):
        if 'png' in item:
            os.remove(f'{item}')
    os.chdir(root)
    print('\n残余文件已经删除完毕返回主目录，预处理执行完毕！')

if __name__ == '__main__':
    main()

