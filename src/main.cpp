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
#include "eyeClosed.hpp"

using namespace dlib;
using namespace std;
cv::Mat debugImage;

cv::Rect dlibRectangleToOpenCV(dlib::rectangle r) {
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
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