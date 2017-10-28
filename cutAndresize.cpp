  //
//  cutAndresize.cpp
//  new
//
//  Created by Stacey_w on 2017/4/19.
//  Copyright © 2017年 子涵^_^. All rights reserved.
//

#include "cutAndresize.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core_c.h>
#include <stdio.h>
#include <fstream>


//计算垂直投影宽度(从上到下投影）
void verProjection_calculate(IplImage * file, int* vArr, int number){
    
    int width = file->width;
    int height = file->height;
    int i=0, j=0, k=0;
    CvScalar ptr;

    for(k=0; k<number; k++){
        vArr[k] = 0;
    }
    //uchar* ptr = (uchar*)file->imageData;
    for(i=0; i<width; i++){
        for(j=0; j<height; j++){
            ptr = cvGet2D(file, j, i);
            if(ptr.val[0] != 0){
                vArr[i] += 1;
            }
        }
        printf("列:%d,     %d\n", i, vArr[i]);
    }
}

//计算水平投影宽度(从左到右投影）
void horiProjection_calculate(IplImage * file, int* vArr, int number){
    
    int width = file->width;
    int height = file->height;
    int i=0, j=0,k=0;
    CvScalar ptr1;
    
    for(k=0; k< number; k++){
        vArr[k] = 0;
    }
    for(i=0; i<height; i++){
        for(j=0; j<width; j++){
            ptr1 = cvGet2D(file, i, j);
            if(ptr1.val[0] != 0){
                vArr[i] += 1;
            }
        }
        //printf("行:%d,     %d\n", i, vArr[i]);
    }
}

//从像素的变化判断字符间隔并分割，垂直方向切割
int** verProjection_cut(int* vArr, int width,int* number){
    int **a;
    int i, flag = 0;
    int num = 0;
   // int threshold = 2;
    
    a = (int**)malloc(width / 2 * sizeof(int*));
    
    for(i=0; i<width-1; i++){
//        if((vArr[i] <= threshold) && (vArr[i+1] > threshold)){
//            a[num] = (int* )malloc(2 * sizeof(int));
//            a[num][0] = i;
//            flag = 1;
//        }
//        else if((vArr[i] > threshold) && (vArr[i+1] <= threshold) && (flag != 0)){
//            a[num][1] = i;
//            num += 1;
//            flag = 0;
//        }
        if(vArr[i-1] == 0 && vArr[i] > 0){
            a[num] = (int* )malloc(2 * sizeof(int));
            a[num][0] = i;
            flag = 1;
        }else if(vArr[i] > 0 && (flag != 0) && vArr[i+1] == 0){
            a[num][1] = i;
            num += 1;
            flag = 0;
        }

    }
    *number = num;
    return a;
}
//水平方向切割，得到图像中共含有几部分代码
int** horiProjection_cut(int* vArr, int height,int* number){
    int **a;
    int i, flag = 0;
    int num = 0;
    
    a = (int**)malloc(height / 2 * sizeof(int*));
        
    for(i=0; i<height-1; i++){
        if(vArr[i-1] == 0 && vArr[i] > 0){
            a[num] = (int* )malloc(2 * sizeof(int));
            a[num][0] = i;
            flag = 1;
        }else if(vArr[i] > 0 && (flag != 0) && vArr[i+1] == 0){
            a[num][1] = i;
            num += 1;
            flag = 0;
        }
    }
    *number = num;
    return a;
}
////按比例选择阈值
//int histogram_Calculate(IplImage * pic, int number){
//    
//    float range[] = { 0, 255 } ;     //灰度级的范围
//    //const float* ranges = { range };
//    int hist_size = 256, hist_height = 256;//直方图尺寸
//    
//    int i;
//    //创建一维直方图
//    CvHistogram * hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY);
//    //计算灰度图像的一维直方图
//    cvCalcHist(&pic, hist,0,0);
//    
//    long int pixel_all = 0, pixel_Calc = 0;
//    
//    int width = pic->width;//int height = pic->height;
//    
//    for(i=0; i<=width; i++){
//        float value = cvQueryHistValue_1D(hist,i);
//        pixel_all += value;
//        //cout<<ptr[i]<<endl;
//    }
//    
//    for(i=0; i<=width; i++){
//        pixel_Calc += ptr[255 - i];
//        if(((pixel_Calc * 100) / pixel_all) > number){
//            i = 255 - i;
//            break;
//        }
//    }
//    return i;
//}

//对图片二值化处理：黑底白字
void pic_Thresholding(IplImage * file, int threshold){
    uchar* ptr = (uchar*)file->imageData;
    int width = file->width;
    int height = file->height;
    
    for(int i = 0; i < width; i++){
        for(int j=0;j<height;j++){
            if(ptr[i*height+j] >= threshold){
                ptr[i*height+j] = 255;   //
            }else{
                ptr[i*height+j] = 0;
            }
        }
    }
}
//修改图片尺寸
IplImage * Resize(IplImage * src,int w, int h)
{
    
    IplImage * now = cvCreateImage (cvSize (w, h),src->depth,src->nChannels);
    cvResize(src, now);
    
    return now;
    
}

/*void showPixel(IplImage * src){
    int width = src->width;
    int height = src->height;
    int i,j;
    CvScalar ptr;
    
    for(i=0; i<width; i++){
        for(j=0; j<height; j++){
            ptr = cvGet2D(src, i, j);
            std::cout<<ptr.val[0]<<endl;
        }
    }    
}*/

IplImage * cutHoriGraph(IplImage * src, int **vArr, int number){
    int i;
    //CvRect rect;
    IplImage * dst;
    for(i=0;i<number;i++){
        int y = vArr[i][0];
        int height = vArr[i][1]-vArr[i][0];
        //printf("cutted的高度%d,  %d\n",height,i);
        dst = cvCreateImage(cvSize(256,height),IPL_DEPTH_8U, 1);
        cvSetImageROI(src, CvRect(0, y, 256,height));
        cvCopy(src, dst);
        cvResetImageROI(src);
        cvShowImage("cutted", dst);
        cvWaitKey();
    }
    return dst;
}

void cutVerGraph(IplImage * src, int **vArr, int number){
    int i;
    //CvRect rect;
    IplImage * dst;
    int reHeight = src->height;
    int j = 1;
    for(i=0;i< number;i++){
        char filename[500];
        int x = vArr[i][0];
        int width = vArr[i][1]-vArr[i][0];
        dst = cvCreateImage(cvSize(width,reHeight),IPL_DEPTH_8U, 1);
        cvSetImageROI(src, CvRect(x, 0, width, reHeight));
        cvCopy(src, dst);
        dst = Resize(dst, 64, 64);
        if(dst != NULL){
            sprintf(filename, "/Users/apple/Desktop/gra_/show/dataset_s/%d.png", j);
            j++;
        }
        std::cout<<filename<<"   "<<j<<std::endl;
        cvShowImage("char", dst);
        cvWaitKey();
        cvSaveImage(filename, dst);
        cvResetImageROI(src);
//        cvShowImage("char", dst);
//        cvWaitKey();
        //cvSaveImage("/Users/apple/Desktop/tensor/roi.jpg", dst);
        //cvReleaseImage(&dst);
    }
}
