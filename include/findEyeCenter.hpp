#include "opencv2/imgproc/imgproc.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <queue>
#include <stdio.h>

using namespace dlib;

// Algorithm Parameters
const int fast_eye_width = 50;
const int weight_blur_size = 5;
const double gradient_threshold = 0.3;

cv::Point findEyeCenter(cv::Mat face, cv::Rect eye, std::string window);

cv::vector<cv::Point> findEyes(cv::Mat frame_gray, cv::Rect face, full_object_detection & shape);