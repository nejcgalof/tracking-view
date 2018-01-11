#include "opencv2/imgproc/imgproc.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>

using namespace dlib;

// Algorithm Parameters
const int fast_eye_width = 50;
const int weight_blur_size = 5;
const bool kEnableWeight = false; // bols dela na false
const float kWeightDivisor = 1.0;
const double gradient_threshold = 0.3; //50
// Postprocessing - ni ok
const bool kEnablePostProcess = false;
const float kPostProcessThreshold = 0.97;

cv::Point findEyeCenter(cv::Mat face, cv::Rect eye);

cv::vector<cv::Point> findEyes(cv::Mat frame_gray, cv::Rect face, full_object_detection & shape);