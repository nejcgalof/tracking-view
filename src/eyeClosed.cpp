#include "..\include\eyeClosed.hpp"

/**
* Based on the work by Soukupová and Èech in their 2016 paper, Real-Time Eye Blink Detection using Facial Landmarks,
* We can then derive an equation that reflects this relation called the eye aspect ratio (EAR).
* Each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye (as if you were looking at the person),
* And then working clockwise around the remainder of the region (p1-p6):
* EAR = ( ||p2-p6|| + ||p3-p5|| ) / (2*||p1-p4||)
* There is a relation between the width and the height of these coordinates.
* Numbers of landmarks for right eye: 43-48 (counting start with 1).
*/
bool eye_close_right(full_object_detection &shape, double eye_closed_threshold) {
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

/**
* Based on the work by Soukupová and Èech in their 2016 paper, Real-Time Eye Blink Detection using Facial Landmarks,
* We can then derive an equation that reflects this relation called the eye aspect ratio (EAR).
* Each eye is represented by 6 (x, y)-coordinates, starting at the left-corner of the eye (as if you were looking at the person),
* And then working clockwise around the remainder of the region (p1-p6):
* EAR = ( ||p2-p6|| + ||p3-p5|| ) / (2*||p1-p4||)
* There is a relation between the width and the height of these coordinates.
* Numbers of landmarks for left eye: 37-42 (counting start with 1).
*/
bool eye_close_left(full_object_detection &shape, double eye_closed_threshold) {
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