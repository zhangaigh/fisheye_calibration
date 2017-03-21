cmake_minimum_required (VERSION 2.8)
project(fisheye_cali)

# find opencv libratry
FIND_PACKAGE(OpenCV REQUIRED)
# find boost library
find_package(Boost 1.40 REQUIRED COMPONENTS filesystem program_options serialization system thread)

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
  add_definitions(-DBOOST_ALL_DYN_LINK)
endif()
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(fisheye_mono_calibrate
	fisheye_mono_camera.cpp
	)
target_link_libraries(fisheye_mono_calibrate
	${OpenCV_LIBS}
	${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
	)


add_executable(fisheye_stereo_calibrate
	fisheye_stereo_camera.cpp
	)
target_link_libraries(fisheye_stereo_calibrate
	${OpenCV_LIBS}
	${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
	)

add_executable(undistortion_images
	undistortion_images.cpp
	)
target_link_libraries(undistortion_images
	${OpenCV_LIBS}
	${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
	)
