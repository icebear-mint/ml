//
//  main.cpp
//  new
//
//  Created by Stacey_w on 2017/3/14.
//  Copyright © 2017年 子涵^_^. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>
#include <fstream>
#include "textDetection.hpp"
#include "cutAndresize.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <exception>

//using namespace cv;

//图像的Canny边缘检测
#include <opencv2/opencv.hpp>

int main() 
{
    int **pic_xArr,**pic_yArr;
    int xArr[1000], yArr[1000];
    int numOfxchar=0,numOfychar=0;
    //int constThre = 0;
    
    CvMemStorage* storage = cvCreateMemStorage (0);;
    CvSeq* contour = 0;
    int contours = 0;
    int ju;
    cout<<"输入图像中文字是否为黑色:1为是，0为否，请输入"<<endl;
    cin>>ju;
    
    //load graph
    IplImage * output = cvLoadImage("/Users/apple/Desktop/gra_/testpic/24.jpg",1);
    cvShowImage("src", output);
    cvWaitKey();
    
    //文本定位
    IplImage * text = textDetection ( output, ju );
    cvShowImage("textdetection", text);
    cvWaitKey();
    
    //修改尺寸
    IplImage * resize = Resize(text,256,256);
    cvShowImage("resize", resize);
    cvWaitKey();
    //showPixel(resize);
    
    
    //灰度化
    IplImage * gray = cvCreateImage(cvGetSize(resize), IPL_DEPTH_8U, 1);
    cvCvtColor(resize, gray, CV_BGR2GRAY);
    cvShowImage("Gray", gray);
    cvWaitKey();
    //showPixel(gray);
    
    //二值化
    IplImage * thre = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1);
    //constThre = histogram_Calculate(thre, 5);
    //cout<<"阈值为:  "<<constThre<<endl;
    //pic_Thresholding(thre, 200);
    cvThreshold (gray, thre, 200, 255, CV_THRESH_BINARY_INV);
    cvShowImage("thre", thre);
    cvWaitKey();
    
    
    //计算水平方向的投影
    int reheight = thre->height;
    horiProjection_calculate(thre, yArr, reheight);
    pic_yArr = verProjection_cut(yArr, reheight, &numOfychar);
    cout<<"共分割出"<<numOfychar<<"部分字符"<<endl;
    for(int i=0; i<numOfychar; i++){
        printf("pic_yArr[%d]:%d, %d\n", i, pic_yArr[i][0], pic_yArr[i][1]);
    }
    cout<<endl;
    
    //计算垂直方向的投影
    IplImage * cutted = cutHoriGraph(thre, pic_yArr, 1);
    
    int rewidth = cutted->width;
    verProjection_calculate(cutted, xArr, rewidth);
    pic_xArr = verProjection_cut(xArr, rewidth, &numOfxchar);
    cout<<"共分割出"<<numOfxchar<<"个字符"<<endl;
    for(int i=0; i<numOfxchar; i++){
        printf("pic_xArr[%d]:%d, %d\n", i, pic_xArr[i][0], pic_xArr[i][1]);
    }
    
    cutVerGraph(cutted, pic_xArr, numOfxchar);
        
//    int idx = 0;
//    char szName[56] = {0};
//    
//    for (CvSeq* c = contour; c != NULL; c = c->h_next) {
//        CvRect rc =cvBoundingRect(c,0);
//        cvDrawRect(thre, cvPoint(rc.x, rc.y), cvPoint(rc.x + rc.width, rc.y + rc.height), CV_RGB(255, 0, 0));
//        
//        IplImage* imgNo = cvCreateImage(cvSize(rc.width, rc.height), IPL_DEPTH_8U, 3);
//        cvSetImageROI(output, rc);
//        //cvCopyimage(output, imgNo);
//        cvResetImageROI(output);
//        sprintf(szName, "wnd_%d", idx++);
//        cvNamedWindow(szName);
//        cvShowImage(szName, imgNo);
//        cvWaitKey();
//        cvReleaseImage(&imgNo);
//    }
    return 0;
}
