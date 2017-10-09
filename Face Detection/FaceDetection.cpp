// FaceDetection.cpp : Defines the entry point for the console application.
//Adapted from openCV face detection tutorial
//

#include "stdafx.h"
#include <iostream> 
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\videoio.hpp>
#include <opencv2\imgproc\imgproc.hpp> 

using namespace std;
using namespace cv;

CascadeClassifier face_classf;
string face_classifier_dir = "C:\\openCV\\Build\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";


void facedetect_boundingbox(Mat frame) {
	vector<Rect> faces;
	Mat gray_frame;
	cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
	face_classf.detectMultiScale(gray_frame,faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30,30));
	for (size_t i = 0; i < faces.size(); i++)
	{
		rectangle(frame, faces[i], (255,0,0), 3);
	}
	imshow("Display Window 1", frame);
}

int main()
{
	VideoCapture capture;
	Mat frame;
	capture.open(0);
	if (!capture.isOpened()) {
		cout << "video not opened";
		return -1;
	}
	 //Loading the cascades
	if (!face_classf.load(face_classifier_dir)) {
		cout << "--Error loading face cascade classifier--\n";
		return(-1);
	}
	//Reading the video stream 
	while (true) {
		capture.read(frame);
		namedWindow("Display Window 1", CV_WINDOW_AUTOSIZE);
		facedetect_boundingbox(frame);
		//imshow("Display Window 1", frame);
		if (waitKey(1) == 27)
			break;
	}
	capture.release();
	destroyWindow("Display Window 1");
	cout << "helo";
	getchar();
    return 0;
}

