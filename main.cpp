#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include <cstring>
#include <algorithm>
#include  <map>
#include <string>
#define inf 1000

using namespace cv;
using namespace std;

#define ChessBoardWidth 7
#define ChessBoardHeight 7
#define ImageNum 3

int img_height, img_width;
int checked_img_num = 0;

Size s = Size(720, 480);//Size(1440,960);


int main(){

	string str_head = "../image/";
	string str[10] = { "1", "2", "3"};
	
	Mat identity = Mat::eye(3, 3, CV_64F);
        vector<vector<Point3d> >  obj_points;   //real objects' point sets
	vector<vector<Point2d> > img_points;   //image's point sets
	vector<Point3d> obj_temp;              //real objects' point in one image
	double sq_sz = 41.27;                  //real chessboard's square's width


	/**********************************************
	*  find chessboard's corners
	*  if find, save points to img_points
	**********************************************/
	cout << "Image_list: ";
	for (int i = 0; i < ImageNum; i++)
	{
		Mat img1 = imread(str_head + str[i] + ".jpg");
		cout << str[i] + ".jpg" << " ";
		vector<Point2f> corners;
		img_width = img1.cols;
		img_height = img1.rows;

		bool found = findChessboardCorners(img1, Size(7, 7), corners);

		Mat gray;
		cvtColor(img1, gray, CV_BGR2GRAY);
		//cornerSubPix(gray, corners, Size(10, 10), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
		drawChessboardCorners(img1, Size(7, 7), corners, found);

		///*
		namedWindow(str[i], 0);
		imshow(str[i], img1);
		waitKey(10);
		//*/

		//if find chessboard's corners, save it to img_points
		if (found)
		{
			vector<Point2d> img_temp;
			for (int j = 0; j < ChessBoardHeight*ChessBoardWidth; j++)
			{
				Point2d temp = corners[j];
				img_temp.push_back(temp);
			}
			img_points.push_back(img_temp);
			checked_img_num++;
		}
	}

	//construct 3d real objects' point sets
	for (int i = 0; i < ChessBoardHeight; i++) {
		for (int j = 0; j < ChessBoardWidth; j++) {
			obj_temp.push_back(Point3d(double(j * sq_sz), double(i * sq_sz), 0));
		}
	}
	for (int i = 0; i < checked_img_num; i++) obj_points.push_back(obj_temp);


	//calibrate param initialize
	cv::Matx33d K,K2;
	cv::Vec4d D;
	int flag = 0;
	flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flag |= cv::fisheye::CALIB_CHECK_COND;
	flag |= cv::fisheye::CALIB_FIX_SKEW;
	//cout << "flag: "<<flag << endl;
	//calibrate the intrinsic param matrix K and distorted coefficient matrix D 
	double calibrate_error=fisheye::calibrate(obj_points, img_points, Size(img_width, img_height), K, D, noArray(), noArray(), flag, TermCriteria(3, 20, 1e-6));
	getOptimalNewCameraMatrix(K, D, s, 1.0, s);
	//adjust the param will produce different K2, which influence the undistort effect
	fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, Size(720, 480), cv::noArray(), K2, 0.8, s ,1.0);
	cout << endl;
	cout << "K:" << K << endl;
	cout << "D:" << D << endl;
	cout << "calibrate_error:  "<< calibrate_error << endl;
	/*data
	K = Matx33d(1.6833736075323174e+002, 0, 3.6025506560344564e+002, 0,
    1.4979904278996142e+002, 2.4054151911694728e+002, 0, 0, 1);
		
	D = Vec4d(2.3215330780211102e-001, - 6.2911846725966114e-002,
		9.7313650204374616e-002, - 7.7633169890538756e-002);
	*/


	//write as XML file 
	FileStorage fs("intrinsics.xml", FileStorage::WRITE);
	Mat mat = Mat(K);
	fs << "intrinsics" << mat;
	mat = Mat(D);
	fs << "coefficient" << mat;
	fs.release();


	//show the undistorted image
	for (int i = 0; i <ImageNum; i++)
	{
		Mat output;// = Mat(Size(img_height, img_width), CV_8UC3);
		Mat img1 = imread(str_head + str[i] + ".jpg");
		fisheye::undistortImage(img1, output, K, D, K2, s);
		namedWindow("img"+str[i], 0);
		imshow("img"+str[i], output);
		waitKey();
	}

	/*
		namedWindow(str[i], 0);
		imshow(str[i], img1);
		waitKey();
	*/


	//system("pause");
	return 0;
}
