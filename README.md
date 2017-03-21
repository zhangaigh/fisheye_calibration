# fisheye_calibration
This calibration tool is based on fisheye calibraton of OPENCV implementation. Currently, monocular and stereo camera calibration can be supported
# build procedure
mkdir build
cd build
ccmake ../
make

# run demo
 ./fisheye_cali -i ../data/images/ -p img
