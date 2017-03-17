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
   cout << " Found corners!" << endl;
    bool found1 = false, found2 = false;

    found1 = findChessboardCorners(imgL, board_size, corners1);
    found2 = findChessboardCorners(imgR, board_size, corners2);
    cout << " Found corners!" << endl;

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
cout << " Found corners!" << endl;
    vector<cv::Point3d> obj;
    for( int i = 0; i < board_height; ++i )
      for( int j = 0; j < board_width; ++j )
        obj.push_back(Point3d(double( (float)j * square_size ), double( (float)i * square_size ), 0));

    if (found1 && found2) {
      cout << " Found corners!" << endl;
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





int main(int argc, char** argv)
{
    cv::Size boardSize;
    float squareSize;
    std::string inputDir;
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
        ("input,i", boost::program_options::value<std::string>(&inputDir)->default_value("images"), "Input directory containing chessboard images")
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
    cout << " Found corners!" << endl;
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

cout << " Found corners!" << endl;
    // look for images in input directory
    std::vector<std::string> imageFilenamesL, imageFilenamesR;
    boost::filesystem::directory_iterator itr;
    for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (!boost::filesystem::is_regular_file(itr->status()))
        {
            continue;
        }

        std::string filename = itr->path().filename().string();

        // check if file extension matches
        if (filename.compare(filename.length() - fileExtension.length(), fileExtension.length(), fileExtension) != 0)
        {
            continue;
        }

        // check if prefix matches
        if (prefixL.empty() || (!prefixL.empty() && (filename.compare(0, prefixL.length(), prefixL) == 0)))
        {
            imageFilenamesL.push_back(itr->path().string());

            if (verbose)
            {
                std::cerr << "# INFO: Adding " << imageFilenamesL.back() << std::endl;
            }
        }
        if (prefixR.empty() || (!prefixR.empty() && (filename.compare(0, prefixR.length(), prefixR) == 0)))
        {
            imageFilenamesR.push_back(itr->path().string());

            if (verbose)
            {
                std::cerr << "# INFO: Adding " << imageFilenamesR.back() << std::endl;
            }
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

    for (size_t i = 0; i < imageFilenamesL.size(); ++i)
    {
        std::string filenameL = boost::filesystem::path(imageFilenamesL.at(i)).filename().string();
        std::string filenameR = boost::filesystem::path(imageFilenamesR.at(i)).filename().string();

        if (filenameL.compare(prefixL.length(),
                              filenameL.size() - prefixL.length(),
                              filenameR,
                              prefixR.length(),
                              filenameR.size() - prefixR.length()) != 0)
        {
            matchImages = false;

            if (verbose)
            {
                std::cerr << "# ERROR: Filenames do not match: "
                          << imageFilenamesL.at(i) << " " << imageFilenamesR.at(i) << std::endl;
            }
        }
    }

    if (!matchImages)
    {
        return 1;
    }

    if (verbose)
    {
        std::cerr << "# INFO: # images: " << imageFilenamesL.size() << std::endl;
    }

cout << " Found corners!" << endl;
    // load images and detect corner points and the corresponding 3d points
    load_image_points(boardSize.width, boardSize.height, squareSize, imageFilenamesL, imageFilenamesR);
cout << " Found corners!" << endl;
    cv::FileStorage fsl(leftcalib_file, cv::FileStorage::READ);
    cv::FileStorage fsr(rightcalib_file, cv::FileStorage::READ);

      printf("Starting Calibration\n");
      cv::Mat K1, K2, R;
      cv::Vec3d T;
      cv::Mat D1, D2;
      int flag = 0;
      //flag |= CV_CALIB_USE_INTRINSIC_GUESS;

      //flag |= cv::fisheye::CALIB_FIX_INTRINSIC; 

      fsl["K"] >> K1;
      fsr["K"] >> K2;
      fsl["D"] >> D1;
      fsr["D"] >> D2;

      cv::fisheye::stereoCalibrate(object_points, left_img_points, right_img_points,
          K1, D1, K2, D2, imgL.size(), R, T, flag,
          cv::TermCriteria(3, 12, 0));

      cv::FileStorage fs1("stereo_camera.yaml", cv::FileStorage::WRITE);
      fs1 << "K1" << Mat(K1);
      fs1 << "K2" << Mat(K2);
      fs1 << "D1" << D1;
      fs1 << "D2" << D2;
      fs1 << "R" << Mat(R);
      fs1 << "T" << T;
      printf("Done Calibration\n");

      printf("Starting Rectification\n");

      cv::Mat R1, R2, P1, P2, Q;
      cv::fisheye::stereoRectify(K1, D1, K2, D2, imgL.size(), R, T, R1, R2, P1, P2, 
    Q, CV_CALIB_ZERO_DISPARITY, imgL.size(), 0.0, 1.1);

      fs1 << "R1" << R1;
      fs1 << "R2" << R2;
      fs1 << "P1" << P1;
      fs1 << "P2" << P2;
      fs1 << "Q" << Q;

      printf("Done Rectification\n");

    //show the undistorted image
    for (size_t i = 0; i < imageFilenamesL.size(); ++i)
    {

        cv::Mat output;// = Mat(Size(img_height, img_width), CV_8UC3);
        cv::Mat imgL = cv::imread(imageFilenamesL.at(i), 0);
        cv::Mat imgR = cv::imread(imageFilenamesR.at(i), 0);
        cv::Mat newK, map1,map2;
        Mat rview(Size(imgL.cols, imgL.rows), imgL.type());
        
        // show undistorted left image 
        fisheye::estimateNewCameraMatrixForUndistortRectify(K1, D1, Size(imgL.cols, imgL.rows), Matx33d::eye(), newK, 1);
        fisheye::initUndistortRectifyMap(K1, D1, Matx33d::eye(), newK, Size(imgL.cols, imgL.rows), CV_16SC2, map1, map2);
        remap(imgL, rview, map1, map2, INTER_LINEAR);
        cv::namedWindow("undistorted left img", 0);
        cv::imshow("undistorted left img", rview);

        // show undistorted right image
        fisheye::estimateNewCameraMatrixForUndistortRectify(K2, D2, Size(imgL.cols, imgL.rows), Matx33d::eye(), newK, 1);
        fisheye::initUndistortRectifyMap(K2, D2, Matx33d::eye(), newK, Size(imgL.cols, imgL.rows), CV_16SC2, map1, map2);
        remap(imgR, rview, map1, map2, INTER_LINEAR);
         cv::namedWindow("undistorted right img", 0);
        cv::imshow("undistorted right img", rview);


        waitKey();
    }



    return 0;
}
