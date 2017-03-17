#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include <cstring>
#include <algorithm>
#include  <map>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace cv;
using namespace std;

int checked_img_num = 0;

int main(int argc, char** argv)
{

	cv::Size boardSize;
    float squareSize;
    std::string inputDir;
    std::string outFile;
    std::string cameraModel;
    std::string cameraName;
    std::string prefix;
    std::string fileExtension;
    bool useOpenCV;
    bool viewResults;
    bool verbose;
	
	//========= Handling Program options =========
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("width,w", boost::program_options::value<int>(&boardSize.width)->default_value(9), "Number of inner corners on the chessboard pattern in x direction")
        ("height,h", boost::program_options::value<int>(&boardSize.height)->default_value(6), "Number of inner corners on the chessboard pattern in y direction")
        ("size,s", boost::program_options::value<float>(&squareSize)->default_value(120.f), "Size of one square in mm")
        ("input,i", boost::program_options::value<std::string>(&inputDir)->default_value("images"), "Input directory containing chessboard images")
        ("prefix,p", boost::program_options::value<std::string>(&prefix)->default_value("image"), "Prefix of images")
        ("file-extension,e", boost::program_options::value<std::string>(&fileExtension)->default_value(".bmp"), "File extension of images")
        ("verbose,v", boost::program_options::bool_switch(&verbose)->default_value(false), "Verbose output")
        ("output,o", boost::program_options::value<std::string>(&outFile)->default_value("left_camera.yaml"), "Output calibration file")
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

    if (!boost::filesystem::exists(inputDir) && !boost::filesystem::is_directory(inputDir))
    {
        std::cerr << "# ERROR: Cannot find input directory " << inputDir << "." << std::endl;
        return 1;
    }

    // look for images in input directory
    std::vector<std::string> imageFilenames;
    boost::filesystem::directory_iterator itr;
    for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (!boost::filesystem::is_regular_file(itr->status()))
        {
            continue;
        }

        std::string filename = itr->path().filename().string();

        // check if prefix matches
        if (!prefix.empty())
        {
            if (filename.compare(0, prefix.length(), prefix) != 0)
            {
                continue;
            }
        }

        // check if file extension matches
        if (filename.compare(filename.length() - fileExtension.length(), fileExtension.length(), fileExtension) != 0)
        {
            continue;
        }

        imageFilenames.push_back(itr->path().string());

        if (verbose)
        {
            std::cerr << "# INFO: Adding " << imageFilenames.back() << std::endl;
        }
    }


    if (imageFilenames.empty())
    {
        std::cerr << "# ERROR: No images found." << std::endl;
        return 1;
    }

    if (verbose)
    {
        std::cerr << "# INFO: # There are images: " << imageFilenames.size() << std::endl;
    }

	
	Mat identity = Mat::eye(3, 3, CV_64F);
    vector<vector<Point3d> >  obj_points;   //real objects' point sets
	vector<vector<Point2d> > img_points;   //image's point sets
	vector<Point3d> obj_temp;              //real objects' point in one image

	/**********************************************
	*  find chessboard's corners
	*  if find, save points to img_points
	**********************************************/
	int img_height, img_width;
	for (size_t i = 0; i < imageFilenames.size(); ++i)
	{
		Mat img1 = imread(imageFilenames.at(i), -1);
		vector<Point2f> corners;
		img_width = img1.cols;
		img_height = img1.rows;

		bool found = findChessboardCorners(img1, Size(boardSize.width, boardSize.height), corners);

		Mat gray;
	//	cvtColor(img1, gray, CV_BGR2GRAY);
		//cornerSubPix(gray, corners, Size(10, 10), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
		drawChessboardCorners(img1, Size(boardSize.width, boardSize.height), corners, found);

		//namedWindow(str[i], 0);
		imshow("Image", img1);
		waitKey(50);
		//*/

		//if find chessboard's corners, save it to img_points
		if (found)
		{
			vector<Point2d> img_temp;
			for (int j = 0; j < boardSize.height*boardSize.width; j++)
			{
				Point2d temp = corners[j];
				img_temp.push_back(temp);
			}
			img_points.push_back(img_temp);
			checked_img_num++;
		}
	}

	//construct 3d real objects' point sets
	for (int i = 0; i < boardSize.height; i++) {
		for (int j = 0; j < boardSize.width; j++) {
			obj_temp.push_back(Point3d(double(j * squareSize), double(i * squareSize), 0));
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
	getOptimalNewCameraMatrix(K, D, Size(img_width, img_height), 1.0, Size(img_width, img_height));
	//adjust the param will produce different K2, which influence the undistort effect
	fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, Size(img_width, img_height), cv::noArray(), K2, 0.8, Size(img_width, img_height) ,1.0);
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
	FileStorage fs(outFile, FileStorage::WRITE);
	Mat mat = Mat(K);
	fs << "K" << mat;
	mat = Mat(D);
	fs << "D" << mat;
	fs.release();


	//show the undistorted image
	for (size_t i = 0; i < imageFilenames.size(); ++i)
	{
		Mat output;// = Mat(Size(img_height, img_width), CV_8UC3);
		Mat img1 = imread(imageFilenames.at(i), -1);
		fisheye::undistortImage(img1, output, K, D, K2, Size(img_width, img_height));
		namedWindow("undistorted img", 0);
		imshow("undistorted img", output);
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
