#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;
vector< vector< Point3d > > object_points;
vector< vector< Point2f > > imagePoints1, imagePoints2;
vector< Point2f > corners1, corners2;
vector< vector< Point2d > > left_img_points, right_img_points;

Mat imgL, imgR, gray1, gray2, spl1, spl2;

void load_image_points(int board_width, 
                       int board_height, 
                       float square_size, 
                       std::vector<std::string>  vImageFilenamesL,
                       std::vector<std::string>  vImageFilenamesR) 
{
  Size board_size = Size(board_width, board_height);
  int board_n = board_width * board_height;

  for (size_t i = 0; i < vImageFilenamesL.size(); ++i)
  {
    imgL = imread(vImageFilenamesL.at(i), -1);
    imgR = imread(vImageFilenamesR.at(i), -1);
    bool found1 = false, found2 = false;

    found1 = findChessboardCorners(imgL, board_size, corners1);
    found2 = findChessboardCorners(imgR, board_size, corners2);

    if (found1)
    {
      cv::cornerSubPix(imgL, corners1, cv::Size(5, 5), cv::Size(-1, -1),
      cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      drawChessboardCorners(imgL, board_size, corners1, found1);
    }
    if (found2)
    {
      cv::cornerSubPix(imgR, corners2, cv::Size(5, 5), cv::Size(-1, -1),
      cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      drawChessboardCorners(imgR, board_size, corners2, found2);
    }
    vector<cv::Point3d> obj;
    for( int i = 0; i < board_height; ++i )
      for( int j = 0; j < board_width; ++j )
        obj.push_back(Point3d(double( (float)j * square_size ), double( (float)i * square_size ), 0));

    if (found1 && found2) {
      imagePoints1.push_back(corners1);
      imagePoints2.push_back(corners2);
      object_points.push_back(obj);
    }
  }
  for (int i = 0; i < imagePoints1.size(); i++) {
    vector< Point2d > v1, v2;
    for (int j = 0; j < imagePoints1[i].size(); j++) {
      v1.push_back(Point2d((double)imagePoints1[i][j].x, (double)imagePoints1[i][j].y));
      v2.push_back(Point2d((double)imagePoints2[i][j].x, (double)imagePoints2[i][j].y));
    }
    left_img_points.push_back(v1);
    right_img_points.push_back(v2);
  }
    
}


void showReclifyImages(const cv::Mat& rectifyImageL, const cv::Mat& rectifyImageR)
{

    Size sz1 = rectifyImageL.size();
    Size sz2 = rectifyImageR.size();
    cv::Mat canvas(sz1.height, sz1.width+sz2.width, CV_8UC1);
    rectifyImageL.copyTo(canvas(Rect(0, 0, sz1.width, sz1.height)));
    rectifyImageR.copyTo(canvas(Rect(sz1.width, 0, sz2.width, sz2.height)));

    imshow("rectified image", canvas);
    waitKey();

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
     
      fsl["K"] >> K1;
      fsr["K"] >> K2;
      fsl["D"] >> D1;
      fsr["D"] >> D2;

  cv::Vec4d D;
  std::vector<cv::Vec3d> rvecs;
  std::vector<cv::Vec3d> tvecs;
  int flag1 = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_FIX_SKEW;
  cv::fisheye::calibrate(object_points, imagePoints1, imgL.size(), K1,
    D1, rvecs, tvecs, flag1, TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5));
  //Rodrigues(rvecs[0], R11);

  cv::fisheye::calibrate(object_points, imagePoints2, imgL.size(), K2,
    D2, rvecs, tvecs, flag1, TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5));
  //Rodrigues(rvecs[0], R22);


   int flag = 0;
      // flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
       // flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
  flag |= cv::fisheye::CALIB_CHECK_COND;
  flag |= cv::fisheye::CALIB_FIX_SKEW;
  flag |= cv::fisheye::CALIB_USE_INTRINSIC_GUESS;
  flag |= cv::fisheye::CALIB_FIX_INTRINSIC;

      cv::fisheye::stereoCalibrate(object_points, left_img_points, right_img_points,
          K1, D1, K2, D2, imgL.size(), R, T, flag,
          cv::TermCriteria(3, 12, 0));

      cv::FileStorage fs1("stereo_camera.yaml", cv::FileStorage::WRITE);
      fs1 << "LEFT_K" << Mat(K1);
      fs1 << "RIGHT_K" << Mat(K2);
      fs1 << "LEFT_D" << D1;
      fs1 << "RIGHT_D" << D2;
      //fs1 << "R" << Mat(R);
      //fs1 << "T" << T;

      cout << "before D1 = "<< endl << " "  << D1 << endl << endl;

      printf("Done Calibration\n");

      printf("Starting Rectification\n");
      cv::Mat R1, R2, P1, P2, Q;
      //cv::Mat P1 = cv::Mat::zeros(3, 4, CV_32FC1);
      //cv::Mat P2 = cv::Mat::zeros(3, 4, CV_32FC1);
//#define MVG
#ifdef MVG
    vector<Point2f> allimgpt[2];
    for (int k = 0; k < 2; k ++)
    {
      for (int i = 0; i < imageFilenamesL.size(); i++)
      {
        if (k == 0)
        {
          std::copy(left_img_points[i].begin(), left_img_points[i].end(), back_inserter(allimgpt[k]));
        }
        else
        {

        std::copy(right_img_points[i].begin(), right_img_points[i].end(), back_inserter(allimgpt[k]));
        }

      }
    }

    cv::fisheye::undistortPoints(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[0]), K1, D1, cv::Mat(), K1);
    cv::fisheye::undistortPoints(cv::Mat(allimgpt[1]), cv::Mat(allimgpt[1]), K2, D2, cv::Mat(), K2);
    cv::Mat F = cv::findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
    cv::Mat H1, H2;
    cv::stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imgL.size(), H1, H2, 3);
    P1 = K1;
    P2 = K1;
   
    R1 = K1.inv()*H1*K1;
    R2 = K2.inv()*H2*K2; 
    /*
    P1 = K1
    K1.copyTo(P1.rowRange(0,3).colRange(0,3));
    K2.copyTo(P2.rowRange(0,3).colRange(0,3));
    //P2.at<double>(0,3) = P1.at<double>(0,0)*T[0] / 1000;
    P2.at<double>(0,3) = 27.5; 
    */
#else
     cv::fisheye::stereoRectify(K1, D1, K2, D2, imgL.size(), R, T, R1, R2, P1, P2, 
    Q, CV_CALIB_ZERO_DISPARITY, imgL.size(), 0.0, 1.1);
#endif
      fs1 << "LEFT_R" << R1;
      fs1 << "RIGHT_R" << R2;
       fs1 << "LEFT_P" << P1;
         fs1 << "RIGHT_P" << P2;
      //fs1 << "Q" << Q;
      printf("Done Rectification\n");

    //show the undistorted image
    for (size_t i = 0; i < imageFilenamesL.size(); ++i)
    {

        cv::Mat output;// = Mat(Size(img_height, img_width), CV_8UC3);
        cv::Mat imgL = cv::imread(imageFilenamesL.at(i), -1);
        cv::Mat imgR = cv::imread(imageFilenamesR.at(i), -1);
        cv::Mat newK, map1,map2;
        Mat rviewR(Size(imgR.cols, imgR.rows), CV_8UC1);
        
        int offset = 30;
        int lineNum = imgR.rows / offset;

        // show undistorted right image
        //fisheye::estimateNewCameraMatrixForUndistortRectify(K2, D2, Size(imgR.cols, imgR.rows), R2, newK, 1);
        fisheye::initUndistortRectifyMap(K2, D2, R2, P2, Size(imgR.cols, imgR.rows), CV_16SC2, map1, map2);
        remap(imgR, rviewR, map1, map2, INTER_LINEAR);

        for(int i = 0; i < lineNum; i ++)
        {
            cv::line(rviewR, cv::Point(0,i*offset), cv::Point(imgR.cols,i*offset), cv::Scalar(255,0,0));

        }
                     
        // cv::namedWindow("undistorted right img", 0);
        //cv::imshow("undistorted right img", rviewR);


              // show undistorted left image 
        cv::Mat map3,map4;
        Mat rviewL(Size(imgL.cols, imgL.rows), imgL.type());
        //fisheye::estimateNewCameraMatrixForUndistortRectify(K1, D1, Size(imgL.cols, imgL.rows), R1, newK, 1);
        fisheye::initUndistortRectifyMap(K1, D1, R1, P1, Size(imgL.cols, imgL.rows), CV_16SC2, map3, map4);
        remap(imgL, rviewL, map3, map4, INTER_LINEAR);

 
     
         for(int i = 0; i < lineNum; i ++)
        {
            cv::line(rviewL, cv::Point(0,i*offset), cv::Point(imgL.cols,i*offset), cv::Scalar(255,0,0));

        }
        //cv::namedWindow("undistorted left img", 0);
        //cv::imshow("undistorted left img", rviewL);

        showReclifyImages(rviewL, rviewR);

    }



    return 0;
}
