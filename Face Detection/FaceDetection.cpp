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
CascadeClassifier profile_face_classf;
string face_classifier_dir = "C:\\openCV\\Build\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
string prof_face_classifier_dir = "C:\\openCV\\Build\\install\\etc\\haarcascades\\haarcascade_profileface.xml";

void facedetect_boundingbox(Mat frame) {
	vector<Rect> faces;
	vector<Rect> right_profile_faces;
	vector<Rect> left_profile_faces;
	Mat gray_frame;
	Mat flipped_gray_frame;
	cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
	flip(gray_frame, flipped_gray_frame, 1);
	face_classf.detectMultiScale(gray_frame,faces, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(50,50));
	profile_face_classf.detectMultiScale(flipped_gray_frame, right_profile_faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));
	profile_face_classf.detectMultiScale(gray_frame, left_profile_faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));
	if (faces.size()) {
		for (size_t i = 0; i < faces.size(); i++)
		{
			rectangle(frame, faces[i], Scalar(255, 0, 255), 1);
		}
	}
	else if (right_profile_faces.size()) {
		for (size_t i = 0; i < right_profile_faces.size(); i++)
		{
			rectangle(frame, right_profile_faces[i], Scalar(0, 0, 255), 1);
		}
	}
	else if (left_profile_faces.size()) {
		for (size_t i = 0; i < left_profile_faces.size(); i++)
		{
			rectangle(frame, left_profile_faces[i], Scalar(0, 128, 255), 1);
		}
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
	if (!profile_face_classf.load(prof_face_classifier_dir)) {
		cout << "--Error loading profile face cascade classifier--\n";
		return(-1);
	}
	//Reading the video stream 
	while (true) {
		capture.read(frame);
		namedWindow("Display Window 1", CV_WINDOW_AUTOSIZE);
		//namedWindow("Display Window 2", CV_WINDOW_AUTOSIZE);
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

