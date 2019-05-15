#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/ccalib/omnidir.hpp"
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

vector< vector< Point3f > > object_points;
vector< vector< Point2f > > imagePoints1, imagePoints2;
vector< Point2f > corners1, corners2;
vector< vector< Point2d > > left_img_points, right_img_points;


Mat img1, img2, gray1, gray2, spl1, spl2;

void load_image_points(int board_width, int board_height, float square_size, int num_imgs,
	char* img_dir, char* leftimg_filename, char* rightimg_filename) {
	Size board_size = Size(board_width, board_height);
	int board_n = board_width * board_height;

	for (int i = 1; i <= num_imgs; i++) {
		char left_img[100], right_img[100];
		sprintf_s(left_img, "%s%s%d.png", img_dir, leftimg_filename, i);
		sprintf_s(right_img, "%s%s%d.png", img_dir, rightimg_filename, i);
		img1 = imread(left_img, CV_LOAD_IMAGE_COLOR);
		img2 = imread(right_img, CV_LOAD_IMAGE_COLOR);
		cv::cvtColor(img1, gray1, CV_BGR2GRAY);
		cv::cvtColor(img2, gray2, CV_BGR2GRAY);

		bool found1 = false, found2 = false;

		found1 = cv::findChessboardCorners(img1, board_size, corners1,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		found2 = cv::findChessboardCorners(img2, board_size, corners2,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (found1)
		{
			cv::cornerSubPix(gray1, corners1, cv::Size(5, 5), cv::Size(-1, -1),
				cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			cv::drawChessboardCorners(gray1, board_size, corners1, found1);
			//imwrite("draw1.jpg", gray1);
		}
		if (found2)
		{
			cv::cornerSubPix(gray2, corners2, cv::Size(5, 5), cv::Size(-1, -1),
				cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			cv::drawChessboardCorners(gray2, board_size, corners2, found2);
			//imwrite("draw2.jpg", gray2);
		}

		vector<cv::Point3f> obj;
		for (int i = 0; i < board_height; ++i)
			for (int j = 0; j < board_width; ++j)
				obj.push_back(Point3d(float((float)j * square_size), float((float)i * square_size), 0));

		if (found1 && found2) {
			cout << i << ". Found corners!" << endl;
			imagePoints1.push_back(corners1);
			imagePoints2.push_back(corners2);
			object_points.push_back(obj);
		}
	}
	for (int i = 0; i < imagePoints1.size(); i++) {
		vector< Point2d > v1, v2;
		for (int j = 0; j < imagePoints1[i].size(); j++) {
			v1.push_back(Point2d((double)imagePoints1[i][j].x, (double)imagePoints1[i][j].y));
			v2.push_back(Point2d((double)imagePoints2[i][j].x, (double)imagePoints2[i][j].y));
		}
		left_img_points.push_back(v1);
		right_img_points.push_back(v2);
	}


}

int main(int argc, char const *argv[])
{
	int board_width, board_height, num_imgs;
	float square_size;
	char* img_dir;
	char* leftimg_filename;
	char* rightimg_filename;
	char* out_file;

	char* img_dir1;
	char* leftimg_filename1;
	char* rightimg_filename1;



	board_width = 9;
	board_height = 6;
	num_imgs = 27;
	square_size = 0.025f;
	img_dir = "../AA/";
	leftimg_filename = "QL";
	rightimg_filename = "QR";
	out_file = "cam_stereo17.yml";
	char* ll;
	char* rr;

	char left[100], right[100], DO[100], RECL[100], RECR[100];

	/*	ll = "QL1.png";
	rr = "QR1.png";*/

	img_dir1 = "../ST/";
	leftimg_filename1 = "OL";
	rightimg_filename1 = "OR";
	int it;
	char* rightimg_filename2 = "DO";
	char* rightimg_filename3 = "RECL";
	char* rightimg_filename4 = "RECR";



	char* img_dir2 = "../DD/";



	//std::vector<cv::Mat> objectPoints, imagePoints11, imagePoints22;
	cv::Size a = img1.size();


	load_image_points(board_width, board_height, square_size, num_imgs, img_dir, leftimg_filename, rightimg_filename);




	cv::Size imgSize1, imgSize2;
	imgSize1 = img1.size();
	imgSize2 = img2.size();

	cv::Mat K1, K2, xi1, xi2, D1, D2;
	Mat idx;
	int flags = 0;
	vector<Vec3d> rvecs, tvecs;
	Vec3d rvec, tvec;
	double _xi1, _xi2, rms;
	cv::TermCriteria critia(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 0.0001);
	rms = cv::omnidir::stereoCalibrate(object_points, imagePoints1, imagePoints2, imgSize1, imgSize2, K1, xi1, D1, K2, xi2, D2, rvec, tvec, rvecs, tvecs, flags, critia, idx);
/*	_xi1 = xi1.at<double>(0);
	_xi2 = xi2.at<double>(0);
	flags = cv::omnidir::RECTIFY_LONGLATI;
	Mat undistorted1, undistorted2;
	cv::Size new_size;
	new_size = img1.size();
	Matx33f Knew;
	Knew = Matx33f(new_size.width / 3.1415, 0, 0,
		0, new_size.height / 3.1415, 0,
		0, 0, 1);
	Mat I1 = imread("RECL2.png", IMREAD_COLOR);
	Mat I2 = imread("RECR2.png", IMREAD_COLOR);
	cv::omnidir::undistortImage(I1, undistorted1, K1, D1, -xi1, flags, Knew, new_size);
	cv::omnidir::undistortImage(I2, undistorted2, -K2, D2, xi2, flags, Knew, new_size);


	imwrite("tr1.png", undistorted1);
	imwrite("tr2.png", undistorted2);*/
	
	cv::Size imgSize = img1.size();
	int numDisparities = 16 * 5;
	int SADWindowSize = 5;
	cv::Mat disMap;
	int flag = cv::omnidir::RECTIFY_LONGLATI;
	int pointType = omnidir::XYZRGB;
	// the range of theta is (0, pi) and the range of phi is (0, pi)
	cv::Matx33d KNew(imgSize.width / 3.1415, 0, 0, 0, imgSize.height / 3.1415, 0, 0, 0, 1);
	Mat imageRec1, imageRec2, pointCloud;

	for (it = 739; it < 749 ; it++)
	{


		sprintf_s(left, "%s%s%d.png", img_dir1, leftimg_filename1, it);
		sprintf_s(right, "%s%s%d.png", img_dir1, rightimg_filename1, it);

		Mat I1 = imread(left, IMREAD_COLOR);
		Mat I2 = imread(right, IMREAD_COLOR);

		cv::omnidir::stereoReconstruct(I1, I2, K1, D1, xi1, K2, D2, xi2, rvec, tvec, flag, numDisparities, SADWindowSize, disMap, imageRec1, imageRec2, imgSize, KNew, pointCloud);
		normalize(disMap, disMap, 0.1, 255, CV_MINMAX, CV_8U);




		sprintf_s(DO, "%s%s%d.png", img_dir1, rightimg_filename2, it);
		sprintf_s(RECL, "%s%s%d.png", img_dir1, rightimg_filename3, it);
		sprintf_s(RECR, "%s%s%d.png", img_dir1, rightimg_filename4, it);





		imwrite(RECL, imageRec1);
		imwrite(RECR, imageRec2);
		imwrite(DO, disMap);
		printf("DD");

		/*int k = left_img_points.size();
		printf("Starting Calibration\n");
		cv::Matx33d K1, K2, R;
		cv::Vec3d T;
		cv::Vec4d D1, D2;
		int flag = 0;
		flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
		flag |= cv::fisheye::CALIB_CHECK_COND;
		flag |= cv::fisheye::CALIB_FIX_SKEW;
		//flag |= cv::fisheye::CALIB_FIX_K2;
		//flag |= cv::fisheye::CALIB_FIX_K3;
		//flag |= cv::fisheye::CALIB_FIX_K4;
		double rms = cv::fisheye::stereoCalibrate(object_points, left_img_points, right_img_points,
		K1, D1, K2, D2, img1.size(), R, T, flag,
		cv::TermCriteria(3, 12, 0));*/

		/*cv::FileStorage fs1(out_file, cv::FileStorage::WRITE);
		fs1 << "K1" << Mat(K1);
		fs1 << "K2" << Mat(K2);
		fs1 << "D1" << D1;
		fs1 << "D2" << D2;
		//fs1 << "R" << Mat(R);
		//fs1 << "T" << T;*/
		printf("Done Calibration\n");

		printf("Starting Rectification\n");

		/*	cv::Mat R1, R2, P1, P2, Q;
		cv::fisheye::stereoRectify(K1, D1, K2, D2, img1.size(), R, T, R1, R2, P1, P2,
		Q, CV_CALIB_ZERO_DISPARITY, img1.size(), 0.0, 1.1);

		fs1 << "R1" << R1;
		fs1 << "R2" << R2;
		fs1 << "P1" << P1;
		fs1 << "P2" << P2;
		fs1 << "Q" << Q;*/

		printf("Done Rectification\n");
	}
	/*int o = object_points.size();
	int im = imagePoints1.size();
	//vector <Mat> rvecs, tvecs;
	//cv::fisheye::calibrate(object_points, imagePoints1, img1.size(), K1, D1, rvecs, tvecs, flag, cv::TermCriteria(3, 12, 0));

	Mat rvecs, tvecs;
	Mat rvecs2, tvecs2;

	Mat r1, t1;
	Mat r12, t12;

	//vector<Mat> r1, t1;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	double alpha;
	vector<Point2d> imageP;
	vector<Point2d> imageP2;

	bool useExtrinsicGuess;
	int flags;
	//solvePnP(object_points, left_img_points, K1, D1, rvecs, tvecs);


	double jacobian;
	vector<float> perViewErrors;
	vector<float> perViewErrors2;


	perViewErrors.resize(object_points.size());
	perViewErrors2.resize(object_points.size());


	for (i = 0; i < (int)object_points.size(); i++)
	{

	//solvePnP(object_points[i], imagePoints1[i], K1, D1, rvecs[i], tvecs[i], useExtrinsicGuess = false, flags = SOLVEPNP_ITERATIVE);
	solvePnP(object_points[i], imagePoints1[i], K1, D1, rvecs, tvecs, useExtrinsicGuess = false, flags = SOLVEPNP_ITERATIVE);
	//cv::solvePnPRansac(Mat(object_points[i]), imagePoints1[i], K1, D1, rvecs, tvecs);
	r1 = rvecs; t1 = tvecs;
	//projectPoints(Mat(object_points[i]), imageP, rvecs[i], tvecs[i], K1, D1);
	//projectPoints(Mat(object_points[i]),  rvecs[i], tvecs[i], K1, D1,imageP);
	projectPoints(Mat(object_points[i]), r1, t1, K1, D1, imageP);
	err = norm(Mat(left_img_points[i]), Mat(imageP), NORM_L2);
	//err = cv::norm(imagePoints1[i] - imageP);
	int n = (int)object_points[i].size();
	perViewErrors[i] = (float)std::sqrt(err*err / n);
	totalErr += err*err;
	totalPoints += n;


	}


	for (i = 0; i < (int)object_points.size(); i++)
	{

	//solvePnP(object_points[i], imagePoints1[i], K1, D1, rvecs[i], tvecs[i], useExtrinsicGuess = false, flags = SOLVEPNP_ITERATIVE);
	solvePnP(object_points[i], imagePoints2[i], K1, D1, rvecs2, tvecs2, useExtrinsicGuess = false, flags = SOLVEPNP_ITERATIVE);
	//cv::solvePnPRansac(Mat(object_points[i]), imagePoints1[i], K1, D1, rvecs, tvecs);
	r12 = rvecs; t12 = tvecs;
	//projectPoints(Mat(object_points[i]), imageP, rvecs[i], tvecs[i], K1, D1);
	//projectPoints(Mat(object_points[i]),  rvecs[i], tvecs[i], K1, D1,imageP);
	projectPoints(Mat(object_points[i]), r12, t12, K1, D1, imageP2);
	err = norm(Mat(right_img_points[i]), Mat(imageP2), NORM_L2);
	//err = cv::norm(imagePoints1[i] - imageP);
	int n = (int)object_points[i].size();
	perViewErrors2[i] = (float)std::sqrt(err*err / n);
	totalErr += err*err;
	totalPoints += n;

	}
	double a = sqrt(totalErr / totalPoints);*/


	//cout << "error :" << rms;
	waitKey();

	//return 0;
}


