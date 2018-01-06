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
#include <Windows.h>

using namespace dlib;
using namespace std;
cv::Mat debugImage;

cv::Rect dlibRectangleToOpenCV(dlib::rectangle r) {
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

inline cv::Point calculateEyeCentroid(const std::vector<cv::Point> eye)
{
	cv::Moments mu = cv::moments(eye);
	cv::Point centroid = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);

	return centroid;
}

void mouseLeftClick()
{
	INPUT Input = { 0 };

	// left down
	Input.type = INPUT_MOUSE;
	Input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
	::SendInput(1, &Input, sizeof(INPUT));

	// left up
	::ZeroMemory(&Input, sizeof(INPUT));
	Input.type = INPUT_MOUSE;
	Input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
	::SendInput(1, &Input, sizeof(INPUT));
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
		bool gamemode = false;
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
			//find biggest face
			rectangle face; 
			for (int i = 0; i < faces.size(); i++) {
				if (faces[i].area() > face.area()) {
					face = faces[i];
				}
			}
			std::vector<full_object_detection> shapes;
			cv::Mat frame = toMat(cimg);
				if (face.area() > 0) {
				full_object_detection shape = pose_model(cimg, face);
				shapes.push_back(shape);
				cv::Mat frame_gray;
				cvtColor(frame, frame_gray, CV_BGR2GRAY);
				cv::vector<cv::Point> pupils = findEyes(frame_gray, dlibRectangleToOpenCV(face), shape);
				std::vector<cv::Point> eye_left = { cv::Point(shape.part(36)(0), shape.part(36)(1)),cv::Point(shape.part(37)(0), shape.part(37)(1)),cv::Point(shape.part(38)(0), shape.part(38)(1)),cv::Point(shape.part(39)(0), shape.part(39)(1)),cv::Point(shape.part(40)(0), shape.part(40)(1)),cv::Point(shape.part(41)(0), shape.part(41)(1)) };
				cv::Point left_eye_center=calculateEyeCentroid(eye_left);
				std::vector<cv::Point> eye_right = { cv::Point(shape.part(42)(0), shape.part(42)(1)),cv::Point(shape.part(43)(0), shape.part(43)(1)),cv::Point(shape.part(44)(0), shape.part(44)(1)),cv::Point(shape.part(45)(0), shape.part(45)(1)),cv::Point(shape.part(46)(0), shape.part(46)(1)),cv::Point(shape.part(47)(0), shape.part(47)(1)) };
				cv::Point right_eye_center = calculateEyeCentroid(eye_right);
				if (!eye_close_left(shape)) {
					cv::drawMarker(frame, left_eye_center, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 10, 1);
					cv::circle(frame, pupils[0], 4, cv::Scalar(255), CV_FILLED);
				}
				if (!eye_close_right(shape)) {
					cv::drawMarker(frame, right_eye_center, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 10, 1);
					cv::circle(frame, pupils[1], 4, cv::Scalar(255), CV_FILLED);
				}
				if (!eye_close_left(shape) && !eye_close_right(shape) && gamemode) {
					POINT p;
					if (GetCursorPos(&p))
					{
						int changex = ((pupils[0].x - left_eye_center.x) + (pupils[1].x - right_eye_center.x) / 2);
						int changey = ((pupils[0].y - left_eye_center.y) + (pupils[1].y - right_eye_center.y) / 2);
						if (changex < 5 && changex > -5) {
							changex = 0;
						}
						if (changey < 2 && changey > -2) {
							changey = 0;
						}
						else {
							changey *= 2;
						}
						int posx = p.x + changex;
						std::cout << posx << endl;
						SetCursorPos(p.x + changex, p.y);
					}
				}
				else if(gamemode){
					mouseLeftClick();
				}
			}
			// Display it all on the screen
			win.clear_overlay();
			cv_image<bgr_pixel> cimg2(frame);
			win.set_image(cimg2);
			win.add_overlay(render_face_detections(shapes));
			if (GetAsyncKeyState('G') & 0x8000)
			{
				gamemode = !gamemode;
				cout << "game mode";
				Sleep(100);
			}
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