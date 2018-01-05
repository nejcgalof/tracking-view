#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "constants.hpp"
#include "findEyeCenter.hpp"

using namespace dlib;
using namespace std;
cv::Mat debugImage;

/**
* @brief Detecting eye EAR with facial landmarks.
*
* Based on the work by Soukupová and Èech in their 2016 paper, Real-Time Eye Blink Detection using Facial Landmarks,
* We can then derive an equation that reflects this relation called the eye aspect ratio (EAR).
* Each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye (as if you were looking at the person),
* And then working clockwise around the remainder of the region (p1-p6):
* EAR = ( ||p2-p6|| + ||p3-p5|| ) / (2*||p1-p4||)
* There is a relation between the width and the height of these coordinates.
* Numbers of landmarks for left eye: 37-42 (counting start with 1).
* Numbers of landmarks for right eye: 43-48 (counting start with 1).
*
* @param[in] landmark_collection Input rcr::LandmarkCollection.
* @return eye aspect for both eyes.
*/
bool eye_close_right(full_object_detection &shape, double eye_closed_threshold = 0.2) {

	// Eye aspect for right eye
	// compute the euclidean distances between the two sets of vertical eye landmarks(x, y) - coordinates
	double right_A = cv::norm(cv::Point2f(shape.part(43)(0), shape.part(43)(1)) - cv::Point2f(shape.part(47)(0), shape.part(47)(1)));
	double right_B = cv::norm(cv::Point2f(shape.part(44)(0), shape.part(44)(1)) - cv::Point2f(shape.part(46)(0), shape.part(46)(1)));
	// compute the euclidean distance between the horizontal eye landmark(x, y) - coordinates
	double right_C = cv::norm(cv::Point2f(shape.part(42)(0), shape.part(42)(1)) - cv::Point2f(shape.part(45)(0), shape.part(45)(1)));
	double right_ear = (right_A + right_B) / (2.0 * right_C);

	if (right_ear < eye_closed_threshold) {
		return true;
	}
	else {
		return false;
	}
}

bool eye_close_left(full_object_detection &shape, double eye_closed_threshold = 0.2) {
	// Eye aspect for left eye
	// compute the euclidean distances between the two sets of vertical eye landmarks(x, y) - coordinates
	double left_A = cv::norm(cv::Point2f(shape.part(37)(0), shape.part(37)(1)) - cv::Point2f(shape.part(41)(0), shape.part(41)(1)));
	double left_B = cv::norm(cv::Point2f(shape.part(38)(0), shape.part(38)(1)) - cv::Point2f(shape.part(40)(0), shape.part(40)(1)));
	// compute the euclidean distance between the horizontal eye landmark(x, y) - coordinates
	double left_C = cv::norm(cv::Point2f(shape.part(36)(0), shape.part(36)(1)) - cv::Point2f(shape.part(39)(0), shape.part(39)(1)));
	double left_ear = (left_A + left_B) / (2.0 * left_C);

	if (left_ear < eye_closed_threshold) {
		return true;
	}
	else {
		return false;
	}
}

cv::Rect dlibRectangleToOpenCV(dlib::rectangle r) {
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

cv::vector<cv::Point> findEyes(cv::Mat frame_gray, cv::Rect face, full_object_detection &shape) {
	cv::vector<cv::Point> pupils;
	cv::vector<cv::Point> pointListLeftEye;
	for (int i = 36; i <= 41; i++) {
		cv::circle(frame_gray, cv::Point(shape.part(i)(0), shape.part(i)(1)), 2, cv::Scalar(255));
		pointListLeftEye.push_back(cv::Point(shape.part(i)(0), shape.part(i)(1)));
	}
	cv::vector<cv::Point> pointListRightEye;
	for (int i = 42; i <= 47; i++) {
		cv::circle(frame_gray, cv::Point(shape.part(i)(0), shape.part(i)(1)), 2, cv::Scalar(255));
		pointListRightEye.push_back(cv::Point(shape.part(i)(0), shape.part(i)(1)));
	}
	cv::Rect leftEyeRegion = cv::boundingRect(pointListLeftEye);
	cv::Rect rightEyeRegion = cv::boundingRect(pointListRightEye);
	cv::rectangle(frame_gray, leftEyeRegion, cv::Scalar(255));
	cv::rectangle(frame_gray, rightEyeRegion, cv::Scalar(255));
	//-- Find Eye Centers
	cv::Point leftPupil = findEyeCenter(frame_gray, leftEyeRegion, "Left Eye");
	cv::Point rightPupil = findEyeCenter(frame_gray, rightEyeRegion, "Right Eye");
	// change eye centers to face coordinates
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;
	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;
	pupils.push_back(leftPupil);
	pupils.push_back(rightPupil);
	return pupils;
}

int main()
{
	try
	{
		cv::VideoCapture cap(0);
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}

		image_window win;

		// Load face detection and pose estimation models.
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		// Grab and process frames until the main window is closed by the user.
		while (!win.is_closed())
		{
			// Grab a frame
			cv::Mat temp;
			if (!cap.read(temp))
			{
				break;
			}
			cv::flip(temp, temp, 1);
			temp.copyTo(debugImage);
			// Turn OpenCV's Mat into something dlib can deal with.  Note that this just
			// wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
			// long as temp is valid.  Also don't do anything to temp that would cause it
			// to reallocate the memory which stores the image as that will make cimg
			// contain dangling pointers.  This basically means you shouldn't modify temp
			// while using cimg.
			cv_image<bgr_pixel> cimg(temp);

			// Detect faces 
			std::vector<rectangle> faces = detector(cimg);
			// Find the pose of each face.
			std::vector<full_object_detection> shapes;
			cv::Mat frame = toMat(cimg);
			for (unsigned long i = 0; i < faces.size(); ++i) {
				full_object_detection shape = pose_model(cimg, faces[i]);
				shapes.push_back(shape);
				cv::Mat frame_gray;
				cvtColor(frame, frame_gray, CV_BGR2GRAY);
				cv::vector<cv::Point> pupils=findEyes(frame_gray, dlibRectangleToOpenCV(faces[i]), shape);
				if (!eye_close_left(shape)) {
					cv::circle(frame, pupils[0], 3, cv::Scalar(255), CV_FILLED);
				}
				if (!eye_close_right(shape)) {
					cv::circle(frame, pupils[1], 3, cv::Scalar(255), CV_FILLED);
				}
			}
			// Display it all on the screen
			win.clear_overlay();
			cv_image<bgr_pixel> cimg2(frame);
			win.set_image(cimg2);
			win.add_overlay(render_face_detections(shapes));
		}
	}
	catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}