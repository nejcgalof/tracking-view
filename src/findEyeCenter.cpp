#include "..\include\findEyeCenter.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <queue>
#include <stdio.h>

bool inMat(cv::Point p, int rows, int cols) {
	return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

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

double compute_dynamic_threshold(const cv::Mat &mat, double std_dev_factor) {
	cv::Scalar mean_magnitude_gradient, std_magnitude_gradient;
	cv::meanStdDev(mat, mean_magnitude_gradient, std_magnitude_gradient);
	double std_dev = std_magnitude_gradient[0] / sqrt(mat.rows*mat.cols);
	return std_dev_factor * std_dev + mean_magnitude_gradient[0];
}

// Pre-declarations
cv::Mat floodKillEdges(cv::Mat &mat);

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

void test_possible_center(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out) {
  // for all possible centers
  for (int cy = 0; cy < out.rows; ++cy) {
    double *Or = out.ptr<double>(cy);
    const unsigned char *Wr = weight.ptr<unsigned char>(cy);
    for (int cx = 0; cx < out.cols; ++cx) {
      if (x == cx && y == cy) {
        continue;
      }
      // create a vector from the possible center to the gradient origin
      double dx = x - cx;
      double dy = y - cy;
      // normalize d
      double magnitude = sqrt((dx * dx) + (dy * dy));
      dx = dx / magnitude;
      dy = dy / magnitude;
      double dotProduct = dx*gx + dy*gy;
      dotProduct = std::max(0.0,dotProduct);
      // square and multiply by the weight
      if (kEnableWeight) {
        Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
      } else {
        Or[cx] += dotProduct * dotProduct;
      }
    }
  }
}

cv::Point findEyeCenter(cv::Mat face, cv::Rect eye) {
  cv::Mat eye_ROI_unscaled = face(eye);
  cv::Mat eye_ROI;
  scale_to_fix_size(eye_ROI_unscaled, eye_ROI);

  // Find the gradient
  cv::Mat gradientX = compute_X_gradient(eye_ROI);
  cv::Mat gradientY = compute_X_gradient(eye_ROI.t()).t(); // transpose for y gradient - turn back for right direction
  imshow("gradientX",gradientX);
  imshow("gradientY",gradientY);
  // Normalize and threshold the gradient
  // Compute all the magnitudes
  cv::Mat mags = matrix_magnitude(gradientX, gradientY);
  imshow("mags1", mags);
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
  imshow("polX",gradientX);
  imshow("polY", gradientY);
  //std::cin.get();

  // Create a blurred and inverted image for weighting
  cv::Mat weight;
  GaussianBlur( eye_ROI, weight, cv::Size(weight_blur_size, weight_blur_size), 0, 0 );
  for (int y = 0; y < weight.rows; ++y) {
    unsigned char *row = weight.ptr<unsigned char>(y);
    for (int x = 0; x < weight.cols; ++x) {
      row[x] = (255 - row[x]);
    }
  }
  imshow("weighted",weight);
  cv::waitKey(39);

  // Run the algorithm

  cv::Mat out_sum = cv::Mat::zeros(eye_ROI.rows, eye_ROI.cols, CV_64F);
  // For each possible gradient location loops for every possible center
  for (int y = 0; y < weight.rows; ++y) {
	double *Xr = gradientX.ptr<double>(y);
	double *Yr = gradientY.ptr<double>(y);
	for (int x = 0; x < weight.cols; ++x) {
		double gX = Xr[x];
		double gY = Yr[x];
		// If not both gradients 0 - possible center
		if (gX == 0.0 && gY == 0.0) {
			continue;
		}
		test_possible_center(x, y, weight, gX, gY, out_sum);
	}
  }

  // scale all the values down, basically averaging them
  double numGradients = (weight.rows*weight.cols);
  cv::Mat out;
  out_sum.convertTo(out, CV_32F,1.0/numGradients);
  //imshow(debugWindow,out);
  //-- Find the maximum point
  cv::Point maxP;
  double maxVal;
  cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
  //-- Flood fill the edges
  if(kEnablePostProcess) {
    cv::Mat floodClone;
    //double floodThresh = computeDynamicThreshold(out, 1.5);
    double floodThresh = maxVal * kPostProcessThreshold;
    cv::threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);
    cv::Mat mask = floodKillEdges(floodClone);
    //imshow(debugWindow + " Mask",mask);
    //imshow(debugWindow,out);
    // redo max
    cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
  }
  return unscale_point(maxP,eye);
}

bool floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat) {
  return inMat(np, mat.rows, mat.cols);
}

// returns a mask
cv::Mat floodKillEdges(cv::Mat &mat) {
  cv::rectangle(mat,cv::Rect(0,0,mat.cols,mat.rows),255);
  
  cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
  std::queue<cv::Point> toDo;
  toDo.push(cv::Point(0,0));
  while (!toDo.empty()) {
    cv::Point p = toDo.front();
    toDo.pop();
    if (mat.at<float>(p) == 0.0f) {
      continue;
    }
    // add in every direction
    cv::Point np(p.x + 1, p.y); // right
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x - 1; np.y = p.y; // left
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y + 1; // down
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y - 1; // up
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    // kill it
    mat.at<float>(p) = 0.0f;
    mask.at<uchar>(p) = 0;
  }
  return mask;
}

cv::vector<cv::Point> findEyes(cv::Mat frame_gray, cv::Rect face, full_object_detection &shape) {
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
	cv::Point leftPupil = findEyeCenter(frame_gray, leftEyeRegion);
	cv::Point rightPupil = findEyeCenter(frame_gray, rightEyeRegion);

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