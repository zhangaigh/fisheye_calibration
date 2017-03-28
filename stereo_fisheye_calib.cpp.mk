//by wanglin
#include "opencv2/calib3d/calib3d.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

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

void StereoCalib(const vector<string>& imagelist, Size boardSize, bool displayCorners, bool useCalibrated, bool showRectified)
{
	if (imagelist.size() % 2 != 0)
	{
		cout << "Error: the image list contains odd (non-even) number of elements\n";
		return;
	}

	const int maxScale = 2;
	const float squareSize = 27.f;  // mm

	//fiseeye calib requires 64 double
	vector<vector<Point2d> > imagePoints[2];
	vector<vector<Point3d> > objectPoints;

	Size imageSize;

	//image sequence lrlrlrlrlr....
	int i, j, k, nimages = (int)imagelist.size() / 2;

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	vector<string> goodImageList;

	for (i = j = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			const string& filename = imagelist[i * 2 + k];
			Mat img = imread(filename, 0);
			if (img.empty())
				break;
			if (imageSize == Size())
				imageSize = img.size();
			else if (img.size() != imageSize)
			{
				cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
				break;
			}
			bool found = false;
			vector<Point2d>& corners = imagePoints[k][j];
			for (int scale = 1; scale <= maxScale; scale++)
			{
				Mat timg;
				if (scale == 1)
					timg = img;
				else
					resize(img, timg, Size(), scale, scale);
				found = findChessboardCorners(timg, boardSize, corners);
				if (found)
				{
					if (scale > 1)
					{
						Mat cornersMat(corners);
						cornersMat *= 1. / scale;
					}
					break;
				}
			}
			if (displayCorners)
			{
				cout << filename << endl;
				Mat cimg, cimg1;
				cvtColor(img, cimg, COLOR_GRAY2BGR);
				cv::drawChessboardCorners(cimg, boardSize, corners, found);
				double sf = 640. / MAX(img.rows, img.cols);
				resize(cimg, cimg1, Size(), sf, sf);
				imshow("corners", cimg1);
				char c = (char)waitKey(500);
				if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
					exit(-1);
			}
			else
				putchar('.');
			if (!found)  break;
			//	cornerSubPix(img, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
		}

		if (k == 2)
		{
			goodImageList.push_back(imagelist[i * 2]);
			goodImageList.push_back(imagelist[i * 2 + 1]);
			j++;
		}
	} //finish detect all images

	cout << j << " pairs have been successfully detected.\n";
	nimages = j;
	if (nimages < 2)
	{
		cout << "Error: too little pairs to run the calibration\n";
		return;
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	objectPoints.resize(nimages);

	for (i = 0; i < nimages; i++)
	{
		for (j = 0; j < boardSize.height; j++)
			for (k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	}
	////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "Running fisheye stereo calibration ...\n";
#define USECV
#ifdef USECV
	string outname = "PfromCV.yml";
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
	//cout << cameraMatrix[0] << cameraMatrix[1] << endl;
	//cout << distCoeffs[0] << distCoeffs[1] << endl;
	////////////////////////////////////////////////////////////////////
	Mat R, T;
	flag1 = fisheye::CALIB_USE_INTRINSIC_GUESS + fisheye::CALIB_CHECK_COND + fisheye::CALIB_FIX_SKEW ;

	double rms = fisheye::stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1], cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1], imageSize, R, T, flag1,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
	cout << "done with RMS error=" << rms << endl;

#else
	/////////////////////set from matlab  results
	string outname = "PfromMatlab.yml";
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
	for (i = 0; i < nimages; i++)
	{
		int npt = (int)imagePoints[0][i].size();
		Mat imgpt ;
		for (k = 0; k < 2; k++)
		{
			imgpt = Mat(imagePoints[k][i]);
			cv::fisheye::undistortPoints(imgpt, imagePoints[k][i], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
			cv::computeCorrespondEpilines(imagePoints[k][i], k + 1, F, lines[k]);
		}
		for (j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] + imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] + imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "average epipolar err = " << err / npoints << endl;

	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];
	if(0)
		cv::fisheye::stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, 	R, T, R1, R2, P1, P2, Q, 1, imageSize)	;
		//HARTLEY'S METHOD  use intrinsic parameters of each camera, but compute the rectification transformation directly  from the fundamental matrix
 	else//if(!useCalibrated)
	{
		vector<Point2f> allimgpt[2];
		for (k = 0; k < 2; k++)
		{
			for (i = 0; i < nimages; i++)
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

	// save parameters
	FileStorage fs(outname, FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "LEFTheight" << imageSize.height << "LEFTDwidth" << imageSize.width
			<< "LEFTD" << distCoeffs[0] << "LEFTK" << cameraMatrix[0] << "LEFTR" << R1 << "LEFTP" << Pt1
			<< "RIGHTheight" << imageSize.height << "RIGHTwidth" << imageSize.width
			<< "RIGHTD" << distCoeffs[1] << "RIGHTK" << cameraMatrix[1] << "RIGHTR" << R2 << "RIGHTP" << Pt2;

		fs.release();
	}

	// COMPUTE AND DISPLAY RECTIFICATION
	if (!showRectified)
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
	for (i = 0; i < 1; i++)
	{
		for (k = 0; k < 2; k++)
		{
			// Mat img= imread(goodImageList[i * 2 + k], 0);
			Mat rimg, cimg;
			Mat img = im[k];
			cv::remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
			cv::cvtColor(rimg, cimg, COLOR_GRAY2BGR);
			Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
			cv::resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
			if (useCalibrated)
			{
				Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
					cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
				rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
			}
		}
		//»­Ïß
		if (!isVerticalStereo)
			for (j = 0; j < canvas.rows; j += 16) line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
			for (j = 0; j < canvas.cols; j += 16) line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);

		imshow("rectified", canvas);

		if (i == 0)
			imwrite("rectified.png", canvas);

		char c = (char)waitKey();
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}

//xml file
static bool readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string)*it);
	return true;
}

int main(int argc, char** argv)
{

	 cv::Size boardSize;
    float squareSize;
    std::string inputDir;
    std::string leftDir;
    std::string rightDir;
    std::string outputDir;
    std::string cameraModel;
    std::string cameraNameL, cameraNameR;
    std::string prefixL, prefixR;
    std::string fileExtension;
    bool useOpenCV;
    bool viewResults;
    bool verbose;

    std::string leftcalib_file;
    std::string rightcalib_file;
    char* out_file;

    //========= Handling Program options =========
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("width,w", boost::program_options::value<int>(&boardSize.width)->default_value(9), "Number of inner corners on the chessboard pattern in x direction")
        ("height,h", boost::program_options::value<int>(&boardSize.height)->default_value(6), "Number of inner corners on the chessboard pattern in y direction")
        ("size,s", boost::program_options::value<float>(&squareSize)->default_value(120.f), "Size of one square in mm")
        ("left,l", boost::program_options::value<std::string>(&leftDir)->default_value("images"), "Input directory containing chessboard images")
        ("right,r", boost::program_options::value<std::string>(&rightDir)->default_value("images"), "Input directory containing chessboard images")
        ("output,o", boost::program_options::value<std::string>(&outputDir)->default_value("."), "Output directory containing calibration data")
        ("prefix-l", boost::program_options::value<std::string>(&prefixL)->default_value("left"), "Prefix of images from left camera")
        ("leftcalib_file,u", boost::program_options::value<std::string>(&leftcalib_file)->default_value("left_camera.yaml"),"Left camera calibration")
        ("rightcalib_file,v",boost::program_options::value<std::string>(&rightcalib_file)->default_value("right_camera.yaml"),"Right camera calibration")
        ("prefix-r", boost::program_options::value<std::string>(&prefixR)->default_value("right"), "Prefix of images from right camera")
        ("file-extension,e", boost::program_options::value<std::string>(&fileExtension)->default_value(".bmp"), "File extension of images")
        ("camera-name-l", boost::program_options::value<std::string>(&cameraNameL)->default_value("camera_left"), "Name of left camera")
        ("camera-name-r", boost::program_options::value<std::string>(&cameraNameR)->default_value("camera_right"), "Name of right camera")
        ("opencv", boost::program_options::bool_switch(&useOpenCV)->default_value(false), "Use OpenCV to detect corners")
        ("view-results", boost::program_options::bool_switch(&viewResults)->default_value(false), "View results")
        ("verbose,v", boost::program_options::bool_switch(&verbose)->default_value(false), "Verbose output")
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
    std::vector<std::string> imageFilenamesL, imageFilenamesR;
    boost::filesystem::directory_iterator itr;
    for (boost::filesystem::directory_iterator itr(leftDir); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (!boost::filesystem::is_regular_file(itr->status()))
        {
            continue;
        }
        // check if prefix matches
        imageFilenamesL.push_back(itr->path().string());

        if (verbose)
        {
            std::cerr << "# INFO: Adding " << imageFilenamesL.back() << std::endl;
        }
    }

    for (boost::filesystem::directory_iterator itr1(rightDir); itr1 != boost::filesystem::directory_iterator(); ++itr1)
    {
        if (!boost::filesystem::is_regular_file(itr1->status()))
        {
            continue;
        }

        imageFilenamesR.push_back(itr1->path().string());

        if (verbose)
        {
            std::cerr << "# INFO: Adding " << imageFilenamesR.back() << std::endl;
        }
    }


    if (imageFilenamesL.empty() || imageFilenamesR.empty())
    {
        std::cerr << "# ERROR: No chessboard images found." << std::endl;
        return 1;
    }

    if (imageFilenamesL.size() != imageFilenamesR.size())
    {
        std::cerr << "# ERROR: # chessboard images from left and right cameras do not match." << std::endl;
        return 1;
    }

    bool matchImages = true;
    std::sort(imageFilenamesL.begin(), imageFilenamesL.end());
    std::sort(imageFilenamesR.begin(), imageFilenamesR.end());
    if (!matchImages)
    {
        return 1;
    }

    if (verbose)
    {
        std::cerr << "# INFO: # images: " << imageFilenamesL.size() << std::endl;
    }

    // load images and detect corner points and the corresponding 3d points
    load_image_points(boardSize.width, boardSize.height, squareSize, imageFilenamesL, imageFilenamesR);
    cv::FileStorage fsl(leftcalib_file, cv::FileStorage::READ);
    cv::FileStorage fsr(rightcalib_file, cv::FileStorage::READ);
      printf("Starting Calibration\n");
      cv::Mat K1, K2, R;
      cv::Vec3d T;
      cv::Mat D1, D2;



	//string imagelistfn = ".\\stereo_calib_short.xml";
	string imagelistfn = ".\\stereo_calib.xml";

	Size boardSize;
	bool showRectified = true;
	boardSize.width = 8;
	boardSize.height = 6;

	vector<string> imagelist;
	bool ok = readStringList(imagelistfn, imagelist);
	if (!ok || imagelist.empty())
	{
		cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
		return 0;
	}

	StereoCalib(imagelist, boardSize, false, false, showRectified);

	StereoCalib(imageFilenamesL, imageFilenamesR, boardSize, false, false, showRectified);
	return 0;
}
