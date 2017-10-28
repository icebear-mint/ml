import numpy as np
import PIL.Image as Image
import pickle as p
import os
#import matplotlib.pyplot as pyplot


y=[]
def ImageToMatrix(filenames):
    global y
    dataList = []
    for i in range (3801):

        filename = filenames + str(i) +".png"
        #print(filename)
        img = np.array(Image.open(filename).convert("L"))
        rows,cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if(img[i,j]<=128):
                    img[i,j] = 1
                else:
                    img[i,j] = 0
    #print(img)
        new_data = np.reshape(img,[4096])
        dataList.append(new_data)

    y=np.zeros([3801,62],dtype=np.float32)
    for i in range(10):
        change(i*100,i*100+100,i)
    count=10
    change(1000,1156,count)
    count+=1
    change(1156,1191,count)
    count+=1
    change(1191,1251,count)
    count+=1
    change(1251,1296,count)
    count+=1
    change(1296,1396,count)
    count+=1
    change(1396,1431,count)
    count+=1
    change(1431,1466,count)
    count+=1
    change(1466,1516,count)
    count+=1
    change(1516,1641,count)
    count+=1
    change(1641,1671,count)
    count+=1
    change(1671,1701,count)
    count+=1
    change(1701,1756,count)
    count+=1
    change(1756,1796,count)
    count+=1
    change(1796,1901,count)
    count+=1
    change(1901,2016,count)
    count+=1
    change(2016,2051,count)
    count+=1
    change(2051,2101,count)
    count+=1
    change(2101,2226,count)
    count+=1
    change(2226,2346,count)
    count+=1
    change(2346,2456,count)
    count+=1
    change(2456,2496,count)
    count+=1
    change(2496,2526,count)
    count+=1
    change(2526,2561,count)
    count+=1
    change(2561,2596,count)
    count+=1
    change(2596,2636,count)
    count+=1
    change(2636,2671,count)
    count+=1
    change(2671,2716,count)
    count+=1
    change(2716,2761,count)
    count+=1
    change(2761,2796,count)
    count+=1
    change(2796,2841,count)
    count+=1
    change(2841,2876,count)
    count+=1
    change(2876,2916,count)
    count+=1
    change(2916,2956,count)
    count+=1
    change(2956,3001,count)
    count+=1
    change(3001,3041,count)
    count+=1
    change(3041,3096,count)
    count+=1
    change(3096,3141,count)
    count+=1
    change(3141,3196,count)
    count+=1
    change(3196,3251,count)
    count+=1
    change(3251,3306,count)
    count+=1
    change(3306,3351,count)
    count+=1
    change(3351,3396,count)
    count+=1
    change(3396,3431,count)
    count+=1
    change(3431,3476,count)
    count+=1
    change(3476,3516,count)
    count+=1
    change(3516,3556,count)
    count+=1
    change(3556,3596,count)
    count+=1
    change(3596,3636,count)
    count+=1
    change(3636,3676,count)
    count+=1
    change(3676,3716,count)
    count+=1
    change(3716,3756,count)
    count+=1
    change(3756,3801,count)
    count+=1
    # print('count is {}'.format)
    return np.asarray(np.reshape(dataList,[3801,4096])),np.asarray(y)
    # return np.ones([3801,4096]),np.asarray(y)
def change(s,e,x):
    global y
    for i in range(s,e):
        y[i,x]=1.0


def batch_index(length,batch_size,n_iter=100):
    print("length is {}".format(length))
    index=list(range(length))
    for j in range(n_iter):
        np.random.shuffle(index)
        for i in range(int(length/batch_size)):
            yield index[i*batch_size:(i+1)*batch_size]

if __name__ == '__main__':
    filename = "dataset_/"
    dataList_ ,tr_y= ImageToMatrix(filename)
    #a4=sum(dataList_[:,32])
    #print(a4)
    # np.set_printoptions(threshold='nan')  #全部输出  
    print (dataList_)
    x=[1,2,3,4,5]
    x=np.asarray(x)
    da=dataList_ [0]
    print(x[1:3])
    for i in range(64):
        print(da[i*64:(i+1)*64])

    print(np.shape(dataList_))
    # print (str(tr_y))
    print(np.shape(tr_y))
    cnt=0
    # for i in range(500):
    #     for j in range(4096):
    #         if dataList_[i,j] != 1:
    #             cnt+=1

    # print('unequal cnt is {}'.format(cnt))
    # print('max cnt is {}'.format(500*4096))

