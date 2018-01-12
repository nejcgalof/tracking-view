#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>

using namespace dlib;

bool eye_close_right(full_object_detection & shape, double eye_closed_threshold=0.2);

bool eye_close_left(full_object_detection & shape, double eye_closed_threshold=0.2);
