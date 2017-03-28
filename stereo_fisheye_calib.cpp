//by wanglin
#include "opencv2/calib3d/calib3d.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

void StereoCalib(std::vector<std::string> imageFilenames[2], cv::FileStorage fs, Size boardSize, float squareSize, bool displayCorners, bool verbose)
{

	const int maxScale = 2;

	//fiseeye calib requires 64 double
	vector<vector<Point2d> > imagePoints[2];
	vector<vector<Point3d> > objectPoints;

	Size imageSize(640,480);


	imagePoints[0].resize(imageFilenames[0].size());
	imagePoints[1].resize(imageFilenames[1].size());

	for (int i = 0; i < imageFilenames[0].size(); i++)
	{
		for (int k = 0; k < 2; k++)
		{
			Mat img = imread(imageFilenames[k][i], 0);
			if (img.empty())
				break;
			bool found = false;
			vector<Point2d>& corners = imagePoints[k][i];
			found = findChessboardCorners(img, boardSize, corners);
			std::cout<<"find "<<corners.size()<<std::endl;
			if (!found)  break;
			//	cornerSubPix(img, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
		}

	} //finish detect all images

	///////////////////////////////////////////////////////////////////////////////////////////
	objectPoints.resize(imageFilenames[0].size());

	for (int i = 0; i < imageFilenames[0].size(); i++)
	{
		for (int j = 0; j < boardSize.height; j++)
			for (int k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	}
	////////////////////////////////////////////////////////////////////////////////////////////////
#define USECV
#ifdef USECV
	cout << "Running stereo calibration ...\n";
	Mat cameraMatrix[2], distCoeffs[2];

	cv::Vec4d D;
	std::vector<cv::Vec3d> rvecs;
	std::vector<cv::Vec3d> tvecs;
	Mat R11, R22, T11, T22;
	int flag1 = fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_FIX_SKEW;
	cv::fisheye::calibrate(objectPoints, imagePoints[0], imageSize, cameraMatrix[0],
		distCoeffs[0], rvecs, tvecs, flag1, TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5));
	//Rodrigues(rvecs[0], R11);

	cv::fisheye::calibrate(objectPoints, imagePoints[1], imageSize, cameraMatrix[1],
		distCoeffs[1], rvecs, tvecs, flag1, TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5));
	//Rodrigues(rvecs[0], R22);
	cout << cameraMatrix[0] << cameraMatrix[1] << endl;
	cout << distCoeffs[0] << distCoeffs[1] << endl;
	////////////////////////////////////////////////////////////////////
	Mat R, T;
	//flag1 = fisheye::CALIB_USE_INTRINSIC_GUESS + fisheye::CALIB_CHECK_COND + fisheye::CALIB_FIX_SKEW ;
	flag1 = fisheye::CALIB_USE_INTRINSIC_GUESS + + fisheye::CALIB_FIX_SKEW;
	double rms = fisheye::stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1], cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1], imageSize, R, T, flag1,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
	cout << "done with RMS error=" << rms << endl;

#else
	/////////////////////set from matlab  results
	Mat cameraMatrix[2], distCoeffs[2];
#define dtype double
	Mat K0 = (Mat_<dtype>(3, 3) << 273.099531355344823, 0.0, 318.123160490812381, 0.0, 272.13573480448747, 255.021528200709042, 0.0, 0.0, 1.0);
	Mat K1 = (Mat_<dtype>(3, 3) << 271.059859974160759, 0, 318.890070526797786, 0, 269.47740504126125, 248.230635839155667, 0, 0, 1.0);
	cameraMatrix[0] = K0;
	cameraMatrix[1] = K1;

	Mat D0 = (Mat_<dtype>(1, 5) << -0.303276694577959194, 0.0936490075024987156, -0.000295673554223246113, 0.00103329536000749413, 0.0);
	Mat D1 = (Mat_<dtype>(1, 5) << -0.328966167988615221, 0.112770718854509969, -0.00441705240734068319, -0.000689879818191744946, 0.0);
	distCoeffs[0] = D0;
	distCoeffs[1] = D1;

	Mat R = (Mat_<dtype>(3, 3) << 0.9996, -0.0165, 0.0208, 0.0167, 0.9998, -0.0097, -0.0206, 0.0100, 0.9997);
	Mat T = (Mat_<dtype>(3, 1) << -92.0296, 1.2247, -3.1686);

	Mat E = (Mat_<dtype>(3, 3) << -0.026652613, 3.1561491, 1.2560254, -1.252779, -0.94136074, 92.070766, 0.28990824, -92.032911, -0.89488283);
	Mat F = (Mat_<dtype>(3, 3) << -0.00000036004239, 0.000042786496, -0.0061631832, -0.000017022806, -0.000012836545, 0.35035309, 0.0054019439, -0.34864529, 0.0095230128);
#endif    

	Mat F;
	F.create(3, 3, CV_32FC1);

	double err = 0;
	int npoints = 0;
	vector<Vec3d> lines[2];
	for (int i = 0; i < imageFilenames[0].size(); i++)
	{
		int npt = (int)imagePoints[0][i].size();
		Mat imgpt ;
		for (int k = 0; k < 2; k++)
		{
			imgpt = Mat(imagePoints[k][i]);
			cv::fisheye::undistortPoints(imgpt, imagePoints[k][i], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
			//cv::computeCorrespondEpilines(imagePoints[k][i], k + 1, F, lines[k]);
		}
	
	/*	for (int j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] + imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] + imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		} 
		npoints += npt;  */
	}
	//cout << "average epipolar err = " << err / npoints << endl;

	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];
	if(0)
		cv::fisheye::stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, 	R, T, R1, R2, P1, P2, Q, 1, imageSize)	;
		//HARTLEY'S METHOD  use intrinsic parameters of each camera, but compute the rectification transformation directly  from the fundamental matrix
 	else//if(!useCalibrated)
	{
		vector<Point2f> allimgpt[2];
		for (int k = 0; k < 2; k++)
		{
			for (int i = 0; i < imageFilenames[0].size(); i++)
				std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
		}
		F = cv::findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
		Mat H1, H2;
		cv::stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);
		 
		R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
		R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];

		Mat k_new = (Mat_<double>(3, 3) << 210,0,320,0,210,240,0,0,1);  
		P1 = cameraMatrix[0];
		P2 = cameraMatrix[1];
	}
	double bsline = P1.at<double>(0, 0) * T.at<double>(0, 0) / 1000; 
	cv::Mat Pt1 = Mat::zeros(3, 4, P1.type());
	cv::Mat Pt2 = Mat::zeros(3, 4, P1.type());
	P1.copyTo(Pt1.rowRange(0, 3).colRange(0, 3));
	P2.copyTo(Pt2.rowRange(0, 3).colRange(0, 3));
	Pt2.at<double>(0, 3) = bsline; 

	if (fs.isOpened())
	{
		fs << "LEFTheight" << imageSize.height << "LEFTDwidth" << imageSize.width
			<< "LEFTD" << distCoeffs[0] << "LEFTK" << cameraMatrix[0] << "LEFTR" << R1 << "LEFTP" << Pt1
			<< "RIGHTheight" << imageSize.height << "RIGHTwidth" << imageSize.width
			<< "RIGHTD" << distCoeffs[1] << "RIGHTK" << cameraMatrix[1] << "RIGHTR" << R2 << "RIGHTP" << Pt2;

		fs.release();
	}

	// COMPUTE AND DISPLAY RECTIFICATION
	if (!verbose)
		return;

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
	//Precompute maps for cv::remap()
	Mat rmap[2][2];
	cv::fisheye::initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	cv::fisheye::initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;
	double sf;
	int w, h;
	if (!isVerticalStereo)
	{
		sf = 600. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width*sf);
		h = cvRound(imageSize.height*sf);
		canvas.create(h, w * 2, CV_8UC3);
	}
	else
	{
		sf = 300. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width*sf);
		h = cvRound(imageSize.height*sf);
		canvas.create(h * 2, w, CV_8UC3);
	}
	Mat im[2];
	im[0] = imread("83660423000_l.png", 0);
	im[1] = imread("83660423000_r.png", 0);
	for (int i = 0; i < 1; i++)
	{
		for (int k = 0; k < 2; k++)
		{
			// Mat img= imread(goodImageList[i * 2 + k], 0);
			Mat rimg, cimg;
			Mat img = im[k];
			cv::remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
			cv::cvtColor(rimg, cimg, COLOR_GRAY2BGR);
			Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
			cv::resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
			if (true)
			{
				Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
					cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
				rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
			}
		}
		//»­Ïß
		if (!isVerticalStereo)
			for (int j = 0; j < canvas.rows; j += 16) line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
			for (int j = 0; j < canvas.cols; j += 16) line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);

		imshow("rectified", canvas);

		if (i == 0)
			imwrite("rectified.png", canvas);

		char c = (char)waitKey();
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}


int main(int argc, char** argv)
{

	 cv::Size boardSize;
    float squareSize;
    std::string inputDir;
    std::string leftDir;
    std::string rightDir;
    std::string outputFile;
    std::string cameraModel;
    bool verbose;

    std::string leftcalib_file;
    std::string rightcalib_file;

    //========= Handling Program options =========
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("width,w", boost::program_options::value<int>(&boardSize.width)->default_value(9), "Number of inner corners on the chessboard pattern in x direction")
        ("height,h", boost::program_options::value<int>(&boardSize.height)->default_value(6), "Number of inner corners on the chessboard pattern in y direction")
        ("size,s", boost::program_options::value<float>(&squareSize)->default_value(120.f), "Size of one square in mm")
        ("left,l", boost::program_options::value<std::string>(&leftDir)->default_value("images"), "Input directory containing chessboard images")
        ("right,r", boost::program_options::value<std::string>(&rightDir)->default_value("images"), "Input directory containing chessboard images")
        ("output,o", boost::program_options::value<std::string>(&outputFile)->default_value("stereo_camera.yaml"), "Output directory containing calibration data")
        ("verbose,v", boost::program_options::bool_switch(&verbose)->default_value(true), "Verbose output")
        ;



    boost::program_options::positional_options_description pdesc;
    pdesc.add("input", 1);

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
    boost::program_options::notify(vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    if (!boost::filesystem::exists(leftDir) && !boost::filesystem::is_directory(leftDir))
    {
        std::cerr << "# ERROR: Cannot find left directory " << leftDir << "." << std::endl;
        return 1;
    }

    if (!boost::filesystem::exists(rightDir) && !boost::filesystem::is_directory(rightDir))
    {
        std::cerr << "# ERROR: Cannot find right directory " << rightDir << "." << std::endl;
        return 1;
    }
    // look for images in input directory
    std::vector<std::string> imageFilenames[2];
    boost::filesystem::directory_iterator itr;
    for (boost::filesystem::directory_iterator itr(leftDir); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (!boost::filesystem::is_regular_file(itr->status()))
        {
            continue;
        }
        // check if prefix matches
        imageFilenames[0].push_back(itr->path().string());

        if (verbose)
        {
            std::cerr << "# INFO: Adding " << imageFilenames[0].back() << std::endl;
        }
    }

    for (boost::filesystem::directory_iterator itr1(rightDir); itr1 != boost::filesystem::directory_iterator(); ++itr1)
    {
        if (!boost::filesystem::is_regular_file(itr1->status()))
        {
            continue;
        }

        imageFilenames[1].push_back(itr1->path().string());

        if (verbose)
        {
            std::cerr << "# INFO: Adding " << imageFilenames[1].back() << std::endl;
        }
    }


    if (imageFilenames[0].empty() || imageFilenames[1].empty())
    {
        std::cerr << "# ERROR: No chessboard images found." << std::endl;
        return 1;
    }

    if (imageFilenames[0].size() != imageFilenames[1].size())
    {
        std::cerr << "# ERROR: # chessboard images from left and right cameras do not match." << std::endl;
        return 1;
    }

    bool matchImages = true;
    std::sort(imageFilenames[0].begin(), imageFilenames[0].end());
    std::sort(imageFilenames[1].begin(), imageFilenames[1].end());
    if (!matchImages)
    {
        return 1;
    }

    cv::FileStorage fs(outputFile, cv::FileStorage::WRITE);
      printf("Starting Calibration\n");
	StereoCalib(imageFilenames, fs,boardSize, squareSize, true, verbose);
	return 0;
}
