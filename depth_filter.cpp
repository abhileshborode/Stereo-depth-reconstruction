
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <iostream>
#include <string>
#include <math.h>




using namespace cv;
using namespace cv::ximgproc;
using namespace std;



bool enableVisualization = 0;
// uncomment to visualize results
static const std::string OPENCV_WINDOW = "Image window";


//parameters for stereo matching and filtering
double vis_mult = 5.0;
int wsize = 3;
int max_disp = 16 * 2;
double lambda = 10000.0;
double sigma = 1.0;

/*double vis_mult = 1.0;
int wsize = 3;
int max_disp = 16 * 10;
double lambda = 8000.0;
double sigma = 1.5;*/

int speckleWindowSize = 50;
int speckleRange = 2;


//Some object instatiation that can be done only once
Mat left_for_matcher, right_for_matcher;
Mat left_disp, right_disp;
Mat filtered_disp;
Rect ROI;
Ptr<DisparityWLSFilter> wls_filter;
Mat filtered_disp_vis;


void compute_stereo(Mat& imL, Mat& imR)
{
	Mat Limage(imL);
	//confidence map
	Mat conf_map = Mat(imL.rows, imL.cols, CV_8U);
	conf_map = Scalar(255);

	// downsample images to speed up results

	/*resize(imL, left_for_matcher, Size(), 0.5, 0.5);
	resize(imR, right_for_matcher, Size(), 0.5, 0.5);*/


	resize(imL, left_for_matcher, Size(), 1, 1);
	resize(imR, right_for_matcher, Size(), 1, 1);

	//compute disparity
	Ptr<StereoSGBM> left_matcher = StereoSGBM::create(0, max_disp, wsize);
	left_matcher->setP1(24 * wsize*wsize);
	left_matcher->setP2(96 * wsize*wsize);
	left_matcher->setPreFilterCap(63);
	left_matcher->setUniquenessRatio(10);
	left_matcher->setSpeckleWindowSize(100);
	left_matcher->setSpeckleRange(32);
	left_matcher->setDisp12MaxDiff(1);
	//left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
	left_matcher->setMode(StereoSGBM::MODE_SGBM);

	wls_filter = createDisparityWLSFilter(left_matcher);
	Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

	left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
	right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);

	//filter
	wls_filter->setLambda(lambda);
	wls_filter->setSigmaColor(sigma);
	wls_filter->filter(left_disp, imL, filtered_disp, right_disp);

	//conf_map = wls_filter->getConfidenceMap();
	ROI = wls_filter->getROI();

	//visualization
	getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
	//filtered_disp_vis.convertTo(filtered_disp_vis, CV_8UC1, 255 / (16));
	//filtered_disp_vis.convertTo(filtered_disp_vis, CV_8U,255/32);

	//cv::normalize(filtered_disp_vis,filtered_disp_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
	//applyColorMap(filtered_disp_vis, filtered_disp_vis, COLORMAP_RAINBOW);

	//void cv::filterSpeckles(filtered_disp, 0,4000, 11);


	namedWindow("left", WINDOW_AUTOSIZE);
	namedWindow("right", WINDOW_AUTOSIZE);

	namedWindow("DISP", WINDOW_AUTOSIZE);

	cv::imshow("left", imL);
	cv::imshow("right", imR);
	cv::imshow("DISP", filtered_disp_vis);


	//		viewer.showCloud(cloud_filtered2);
	cv::waitKey();
}




int main(int argc, char **argv) {
	Mat cv_left, cv_right;
	cv_left = imread("INL511.png", IMREAD_COLOR);
	cv_right = imread("INL512.png", IMREAD_COLOR);

	//compute_stereo(cv_left->image, cv_right->image);
	compute_stereo(cv_left, cv_right);

	return 0;
}
