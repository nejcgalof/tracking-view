#include "opencv2/imgproc/imgproc.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>

using namespace dlib;
cv::Point findEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow);

cv::vector<cv::Point> findEyes(cv::Mat frame_gray, cv::Rect face, full_object_detection & shape);