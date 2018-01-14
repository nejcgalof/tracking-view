#include "..\include\findEyeCenter.hpp"

cv::Mat matrix_magnitude(cv::Mat &matX, cv::Mat &matY) {
	cv::Mat magnitude(matX.rows, matX.cols, CV_64F);
	for (int y = 0; y < matX.rows; ++y) {
		double *Xr = matX.ptr<double>(y);
		double *Yr = matY.ptr<double>(y);
		double *Mr = magnitude.ptr<double>(y);
		for (int x = 0; x < matX.cols; ++x) {
			double gX = Xr[x];
			double gY = Yr[x];
			double magnitude = sqrt((gX * gX) + (gY * gY));
			Mr[x] = magnitude;
		}
	}
	return magnitude;
}

double compute_dynamic_threshold(cv::Mat &mat, double std_dev_factor) {
	cv::Scalar mean_magnitude_gradient, std_magnitude_gradient;
	cv::meanStdDev(mat, mean_magnitude_gradient, std_magnitude_gradient);
	double std_dev = std_magnitude_gradient[0] / sqrt(mat.rows*mat.cols);
	return std_dev_factor * std_dev + mean_magnitude_gradient[0];
}

// Resize eye ROI to bigger fix width. Keep ratio on cols and rows. If eye to far or to near camera. Keep always same speed.
void scale_to_fix_size(const cv::Mat &src,cv::Mat &dst) {
  cv::resize(src, dst, cv::Size(fast_eye_width,(((float)fast_eye_width)/src.cols) * src.rows));
}

// Because we scale eye ROI, point location is not correct. We must "unscaled".
cv::Point unscale_point(cv::Point p, cv::Rect origSize) {
	float ratio = (((float)fast_eye_width) / origSize.width);
	int x = round(p.x / ratio);
	int y = round(p.y / ratio);
	return cv::Point(x, y);
}

// Matlab code: [x(2)-x(1) (x(3:end)-x(1:end-2))/2 x(end)-x(end-1)] translated to cpp
// Gradient algorithm
cv::Mat compute_X_gradient(const cv::Mat &mat) {
  cv::Mat out(mat.rows,mat.cols,CV_64F);
  
  for (int y = 0; y < mat.rows; ++y) {
    const uchar *Mr = mat.ptr<uchar>(y);
    double *Or = out.ptr<double>(y);
    
    Or[0] = Mr[1] - Mr[0];
    for (int x = 1; x < mat.cols - 1; ++x) {
      Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
    }
    Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
  }
  
  return out;
}

void test_possible_center(int x, int y, double gx, double gy, cv::Mat &out) {
  // For all possible centers
  for (int cy = 0; cy < out.rows; ++cy) {
    double *Or = out.ptr<double>(cy);
    for (int cx = 0; cx < out.cols; ++cx) {
      if (x == cx && y == cy) {
        continue;
      }
      // Create a vector from the possible center to the gradient origin
      double dx = x - cx;
      double dy = y - cy;
      // Normalize d
      double magnitude = sqrt((dx * dx) + (dy * dy));
      dx = dx / magnitude;
      dy = dy / magnitude;
      double dotProduct = dx*gx + dy*gy;
	  // Negative values change to 0
      dotProduct = std::max(0.0,dotProduct);
	  Or[cx] += dotProduct * dotProduct;
    }
  }
}

cv::Point find_eye_pupil(cv::Mat face, cv::Rect eye, std::string window) {
  cv::Mat eye_ROI_unscaled = face(eye);
  cv::Mat eye_ROI;
  scale_to_fix_size(eye_ROI_unscaled, eye_ROI);

  // Find the gradient
  cv::Mat gradientX = compute_X_gradient(eye_ROI);
  cv::Mat gradientY = compute_X_gradient(eye_ROI.t()).t(); // transpose for y gradient - turn back for right direction
  //imshow(window+"gradientX",gradientX);
  //imshow(window+"gradientY",gradientY);
  // Normalize and threshold the gradient
  // Compute all the magnitudes
  cv::Mat mags = matrix_magnitude(gradientX, gradientY);
  //imshow(window+"mags", mags);
  // Compute the threshold
  double gradient_thresh = compute_dynamic_threshold(mags, gradient_threshold);
  // Normalize
  for (int y = 0; y < eye_ROI.rows; ++y) {
	  double *Xr = gradientX.ptr<double>(y);
	  double *Yr = gradientY.ptr<double>(y);
	  double *Mr = mags.ptr<double>(y);
	  for (int x = 0; x < eye_ROI.cols; ++x) {
		  double gX = Xr[x];
		  double gY = Yr[x];
		  double magnitude = Mr[x];
		  if (magnitude > gradient_thresh) {
			  Xr[x] = gX / magnitude;
			  Yr[x] = gY / magnitude;
		  }
		  else {
			  Xr[x] = 0.0;
			  Yr[x] = 0.0;
		  }
	  }
  }
  //imshow(window+"after_gradientX",gradientX);
  //imshow(window+"after_gradient", gradientY);
  // Run the algorithm

  cv::Mat out_sum = cv::Mat::zeros(eye_ROI.rows, eye_ROI.cols, CV_64F);
  // For each possible gradient location loops for every possible center
  for (int y = 0; y < eye_ROI.rows; ++y) {
	double *Xr = gradientX.ptr<double>(y);
	double *Yr = gradientY.ptr<double>(y);
	for (int x = 0; x < eye_ROI.cols; ++x) {
		double gX = Xr[x];
		double gY = Yr[x];
		// If not both gradients 0 - possible center
		if (gX == 0.0 && gY == 0.0) {
			continue;
		}
		test_possible_center(x, y, gX, gY, out_sum);
	}
  }

  // Scale all the values down = 1/N
  double numGradients = (eye_ROI.rows*eye_ROI.cols);
  cv::Mat out;
  out_sum.convertTo(out, CV_32F, 1.0/numGradients);

  // Show all possible centers
  cv::Mat dout;
  double  minVal1, maxVal1;
  minMaxLoc(out, &minVal1, &maxVal1);  //find  minimum  and  maximum  intensities
  out.convertTo(dout, CV_8U, 255.0 / (maxVal1 - minVal1), -minVal1);
  cv::applyColorMap(dout, dout, cv::COLORMAP_JET);
  cv::resize(dout, dout, cv::Size(), 8, 8);
  cv::imshow(window+"All_possible", dout);
  cv::waitKey(30);

  // Find the maximum point
  cv::Point maxP;
  double maxVal;
  // Find argMax
  cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
  return unscale_point(maxP,eye);
}

cv::vector<cv::Point> find_eyes(cv::Mat frame_gray, cv::Rect face, full_object_detection &shape) {
	cv::vector<cv::Point> pupils;
	// Put eye landmarks (points) to vector
	cv::vector<cv::Point> pointListLeftEye;
	for (int i = 36; i <= 41; i++) {
		pointListLeftEye.push_back(cv::Point(shape.part(i)(0), shape.part(i)(1)));
	}
	cv::vector<cv::Point> pointListRightEye;
	for (int i = 42; i <= 47; i++) {
		pointListRightEye.push_back(cv::Point(shape.part(i)(0), shape.part(i)(1)));
	}

	// Get bounding rect
	cv::Rect leftEyeRegion = cv::boundingRect(pointListLeftEye);
	cv::Rect rightEyeRegion = cv::boundingRect(pointListRightEye);

	// Find Eye Centers
	cv::Point leftPupil = find_eye_pupil(frame_gray, leftEyeRegion,"left");
	cv::Point rightPupil = find_eye_pupil(frame_gray, rightEyeRegion,"right");

	// Change eye centers to face coordinates
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;
	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;

	// Add to result
	pupils.push_back(leftPupil);
	pupils.push_back(rightPupil);
	return pupils;
}