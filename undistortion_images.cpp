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
        ("left,l", boost::program_options::value<std::string>(&leftDir)->default_value("images"), "Input directory containing chessboard images")
        ("right,r", boost::program_options::value<std::string>(&rightDir)->default_value("images"), "Input directory containing chessboard images")
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
      cv::Mat K1, K2, D1, D2, R1, R2, P1, P2, Q;
      cv::FileStorage fs("poc.yaml", cv::FileStorage::READ);
     /* fs["K1"] >> K1;
      fs["K2"] >> K2;
      fs["D1"] >> D1;
      fs["D2"] >> D2;
      fs["R1"] >> R1;
      fs["R2"] >> R2;
      fs["P1"] >> P1;
      fs["P2"] >> P2; */

      fs["LEFT.K"] >> K1;
      fs["RIGHT.K"] >> K2;
      fs["LEFT.D"] >> D1;
      fs["RIGHT.D"] >> D2;
      fs["LEFT.R"] >> R1;
      fs["RIGHT.R"] >> R2;
      fs["LEFT.P"] >> P1;
      fs["RIGHT.p"] >> P2; 

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
        fisheye::estimateNewCameraMatrixForUndistortRectify(K2, D2, Size(imgR.cols, imgR.rows), R2, newK, 1);
        fisheye::initUndistortRectifyMap(K2, D2, R2, newK, Size(imgR.cols, imgR.rows), CV_16SC2, map1, map2);
        remap(imgR, rviewR, map1, map2, INTER_LINEAR);
        for(int i = 0; i < lineNum; i ++)
        {
            cv::line(rviewR, cv::Point(0,i*offset), cv::Point(imgR.cols,i*offset), cv::Scalar(255,0,0));

        }

         //cv::namedWindow("undistorted right img", 0);
        //cv::imshow("undistorted right img", rviewR);


              // show undistorted left image 
        cv::Mat map3,map4;
        Mat rviewL(Size(imgL.cols, imgL.rows), imgL.type());
        fisheye::estimateNewCameraMatrixForUndistortRectify(K1, D1, Size(imgL.cols, imgL.rows), R1, newK, 1);
        fisheye::initUndistortRectifyMap(K1, D1, R1, newK, Size(imgL.cols, imgL.rows), CV_16SC2, map3, map4);
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
