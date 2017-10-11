// FaceDetection.cpp : Defines the entry point for the console application.
//Adapted from openCV face detection tutorial
//

#include "stdafx.h"
#include <iostream>
#include <thread>
#include <future>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\videoio.hpp>
#include <opencv2\imgproc\imgproc.hpp> 

using namespace std;
using namespace cv;

CascadeClassifier face_classf;
CascadeClassifier rprofile_face_classf;
CascadeClassifier lprofile_face_classf;
string face_classifier_dir = "C:\\openCV\\Build\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
string prof_face_classifier_dir = "C:\\openCV\\Build\\install\\etc\\haarcascades\\haarcascade_profileface.xml";


/*class helper {
public:
	CClassf(Mat img= Mat(1,1,CV_64F,cvScalar(0.)), vector<Rect> face_rect(), double scalefactor=1.1, int min_neighbors=3, int flags=0, cv::Size minsize=Size(30,30)) {
		im = img;
		faces=face_rect;
		scale = scalefactor;
		min_n=min_neighbors;
		f = flags;
		minsz = minsize;
	}

	static vector<Rect> classifiercaller() {
		return face_classf.detectMultiScale(im, faces, scale, min_n, f, minsz);
	}

private:
	Mat im;
	vector<Rect> faces;
	double scale;
	int min_n;
	int f;
	Size minsz;
};*/

void facecf(Mat img, promise<vector<Rect>> face_rect, double scalefactor, int min_neighbors, int flags, cv::Size minsize) {
	vector<Rect> faces;
	face_classf.detectMultiScale(img, faces, scalefactor, min_neighbors, flags, minsize);
	//cout << "\n fromf" << faces.size();
	face_rect.set_value(faces);
}
void rpfacecf(Mat img, promise<vector<Rect>> face_rect, double scalefactor, int min_neighbors, int flags, cv::Size minsize) {
	vector<Rect> rfaces;
	rprofile_face_classf.detectMultiScale(img, rfaces, scalefactor, min_neighbors, flags, minsize);
	//cout << "\n fromrpf" << rfaces.size();
	face_rect.set_value(rfaces);
}
void lpfacecf(Mat img, promise<vector<Rect>> face_rect, double scalefactor, int min_neighbors, int flags, cv::Size minsize) {
	vector<Rect> lfaces;
	lprofile_face_classf.detectMultiScale(img, lfaces, scalefactor, min_neighbors, flags, minsize);
	//cout << "\n fromlpf" << lfaces.size();
	face_rect.set_value(lfaces);
}

void facedetect_boundingbox(Mat frame) {
	//vector<Rect> faces;
	//vector<Rect> right_profile_faces;
	//vector<Rect> left_profile_faces;
	promise<vector<Rect>> faces;
	promise<vector<Rect>> right_profile_faces;
	promise<vector<Rect>> left_profile_faces;
	future<vector<Rect>> ftr_faces= faces.get_future();
	future<vector<Rect>> ftr_right_profile_faces= right_profile_faces.get_future();
	future<vector<Rect>> ftr_left_profile_faces= left_profile_faces.get_future();
	Mat gray_frame, small_gray_frame;
	Mat flipped_gray_frame, small_flipped_gray_frame;
	cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
	flip(gray_frame, flipped_gray_frame, 1);
	resize(gray_frame, small_gray_frame, Size(), 0.5, 0.5);
	resize(flipped_gray_frame, small_flipped_gray_frame, Size(), 0.5, 0.5);
	thread t1(&facecf, small_gray_frame, move(faces), 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));
	thread t2(&rpfacecf, small_flipped_gray_frame, move(right_profile_faces), 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));
	thread t3(&lpfacecf, small_gray_frame, move(left_profile_faces), 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));
	vector<Rect> f = ftr_faces.get();
	t1.detach();
	vector<Rect> lf = ftr_left_profile_faces.get(); 
	t2.detach();
	vector<Rect> rf = ftr_right_profile_faces.get();
	t3.detach();
	
	int scalef = 2;

	if (f.size()) {
		for (size_t i = 0; i < f.size(); i++)
		{
			rectangle(frame,Point(scalef * f[i].x, scalef * f[i].y),Point(scalef * (f[i].x + f[i].width), scalef * (f[i].y + f[i].height)), Scalar(255, 0, 255), 1);
		}
	}
	else if (lf.size()) {
		for (size_t i = 0; i < lf.size(); i++)
		{
			rectangle(frame, Point(scalef * lf[i].x, scalef * lf[i].y), Point(scalef * (lf[i].x + lf[i].width), scalef * (lf[i].y + lf[i].height)), Scalar(0, 255, 255), 3);
		}
	}
	else if (rf.size()) {
		for (size_t i = 0; i < rf.size(); i++)
		{
			rectangle(frame, Point(scalef * rf[i].x, scalef * rf[i].y), Point(scalef * (rf[i].x + rf[i].width), scalef * (rf[i].y + rf[i].height)), Scalar(255, 255, 0), 3);
		}
	}
	//t1.join();
	//t2.join();
	//t3.join();
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
	if (!lprofile_face_classf.load(prof_face_classifier_dir)) {
		cout << "--Error loading profile face cascade classifier--\n";
		return(-1);
	}
	if (!rprofile_face_classf.load(prof_face_classifier_dir)) {
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

