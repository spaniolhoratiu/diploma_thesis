// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <limits.h>
#include <queue>
#include <random>
#include <math.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <omp.h>
 

const int WHITE = 255;
const int BLACK = 0;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void changeGrayLevel()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar addedValue;
				int testValue = val + 100;
				if (testValue > 255)
					addedValue = 255;
				else
					addedValue = (uchar) testValue;
				dst.at<uchar>(i, j) = addedValue;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("change gray levels image", dst);
		waitKey();
	}
}

bool isTopLeft(int i, int j, int height, int width)
{
	return (i <= height / 2) && (j <= width / 2) ? true : false;
}

bool isBottomLeft(int i, int j, int height, int width)
{
	return (i > height / 2) && (j <= width / 2) ? true : false;
}

bool isTopRight(int i, int j, int height, int width)
{
	return (i <= height / 2) && (j > width / 2) ? true : false;
}


void fourSquaresColorImage()
{
	Mat_<Vec3b> myImage = Mat_<Vec3b>(600, 600);
	int height = myImage.rows;
	int width = myImage.cols;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (isTopLeft(i, j, height, width)) //top left
			{
				myImage(i, j) = Vec3b(255, 255, 255); // white
			}else
			if (isBottomLeft(i, j, height, width)) //bottom left
			{
				myImage(i, j) = Vec3b(100, 100, 100); // gray
			}else
			if ((i <= height / 2) && (j > width / 2)) // top right
			{
				myImage(i, j) = Vec3b(0, 0, 255); // red
			}else // bottom right
			{
				myImage(i, j) = Vec3b(0, 255, 255); // yellow
			}
		}
	}
	imshow("My squares", myImage);
	waitKey();
}

void inverseMatrix()
{
	float vals[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 10};
	Mat M(3, 3, CV_32FC1, vals); //3 parameter constructor 
	std::cout << "The matrix:"<< std::endl;
	std::cout << M << std::endl;
	std::cout << "The matrix inverse:" << std::endl;
	std::cout << M.inv() << std::endl;
	getchar();
	getchar();
}

void rgb24SplitChannels()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat rMatrix = Mat(height, width, CV_8UC1);
		Mat gMatrix = Mat(height, width, CV_8UC1);
		Mat bMatrix = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b p = src.at<Vec3b>(i, j);
				uchar rValue = p[2];
				uchar gValue = p[1];
				uchar bValue = p[0];
				rMatrix.at<uchar>(i, j) = rValue;
				gMatrix.at<uchar>(i, j) = gValue;
				bMatrix.at<uchar>(i, j) = bValue;
			}
		}

		imshow("input image", src);
		imshow("R", rMatrix);
		imshow("G", gMatrix);
		imshow("B", bMatrix);
		waitKey();
	}
}

void colorToGrayscale()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b p = src.at<Vec3b>(i, j);
				uchar grayColorMedian = (p[0] + p[1] + p[2]) / 3;
				dst.at<uchar>(i, j) = grayColorMedian;
			}
		}

		imshow("input image", src);
		imshow("gray", dst);
		waitKey();
	}
}

void grayscaleToBW()
{
	const int THRESHOLD = 128;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat_<uchar> dst = Mat_<uchar>(height, width);
		
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar currentPixel = src(i, j);
				if (currentPixel > THRESHOLD)
				{
					dst(i, j) = 255;
				}
				else
				{
					dst(i, j) = 0;
				}
			}
		}

		imshow("input image", src);
		imshow("Black and White", dst);
		waitKey();
	}

}

void rgbToHsv()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat_<uchar> hueImage = Mat_<uchar>(height, width);
		Mat_<uchar> saturationImage = Mat_<uchar>(height, width);
		Mat_<uchar> valueImage = Mat_<uchar>(height, width);
		
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b currentPixel = src(i, j);
				uchar redComponentFromColoredImage = currentPixel[2];
				uchar greenComponentFromColoredImage = currentPixel[1];
				uchar blueComponentFromColoredImage = currentPixel[0];
				
				float normalizedR = (float)redComponentFromColoredImage / 255;
				float normalizedG = (float)greenComponentFromColoredImage / 255;
				float normalizedB = (float)blueComponentFromColoredImage / 255;

				float M = max(max(normalizedR, normalizedG), normalizedB);
				float m = min(min(normalizedR, normalizedG), normalizedB);
				float C = M - m;

				// Value
				float V = M;

				// Saturation
				float S;
				if (V != 0)
				{
					S = C / V;
				}
				else
				{
					// Grayscale pixel
					S = 0;
				}

				// Hue	
				float H;
				if (C != 0) // When pixel is not grayscale
				{
					if (M == normalizedR) H = 60 * (normalizedG - normalizedB) / C;
					if (M == normalizedG) H = 120 + 60 * (normalizedB - normalizedR) / C;
					if (M == normalizedB) H = 240 + 60 * (normalizedR - normalizedG) / C;
				}
				else
				{
					// Grayscale pixel
					H = 0;
				}

				if (H < 0)
				{
					H = H + 360;
				}

				float H_norm = H * 255 / 360;
				float S_norm = S * 255;
				float V_norm = V * 255;

				hueImage(i, j) = H_norm;
				saturationImage(i, j) = S_norm;
				valueImage(i, j) = V_norm;
			}
		}

		imshow("input image", src);
		imshow("Hue", hueImage);
		imshow("Saturation", saturationImage);
		imshow("Value", valueImage);
		waitKey();
	}
}


void histogram()
{
	char fname[MAX_PATH];
	const int G_MAX = 255;
	int hg[256] = { 0 };
	float p[256] = { 0 };

	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hg[src(i,j)]++;
			}
		}

		for (int i = 0; i < 256; i++)
		{
			p[i] = (float) hg[i] / (height * width);
			printf("p[%d] = %f\n", i, p[i]);
		}

		showHistogram("Histogram", hg, 256, 256);		
	}
}


bool maxInInterval(float p[], int i, int WH)
{
	for (int j = i - WH; j <= i + WH; j++)
	{
		if (p[j] > p[i])
			return false;
	}
	return true;
}

void multilevelThresholding()
{
	int WH = 5;
	int windowWidth = 2 * WH + 1;
	float TH = 0.0003;
	char fname[MAX_PATH];
	int hg[256] = { 0 };
	float p[256] = { 0 };
	int localMaximal[100];

	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hg[src(i, j)]++;
			}
		}

		for (int i = 0; i < 256; i++)
		{
			p[i] = (float)hg[i] / (height * width);
		}

		int k = 0;
		localMaximal[0] = 0;
		for (int i = WH; i <= 255 - WH; i++)
		{
			float avg = 0;
			for (int j = i - WH; j <= i + WH; j++)
			{
				avg += p[j];
			}
			avg = avg / (2 * WH + 1);

			if ((maxInInterval(p, i, WH)) && (p[i] > TH + avg))
			{
				k++;
				localMaximal[k] = i;
			}
			
		}

		localMaximal[++k] = 255;
		printf("k = %d\n", k);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				for (int index = 0; index < k-1; index++)
				{
					if (src(i, j) >= localMaximal[index] && src(i, j) <= localMaximal[index + 1])
					{
						int difference1 = abs(src(i, j) - localMaximal[index]);
						int difference2 = abs(src(i, j) - localMaximal[index+1]);
						if (difference1 > difference2)
							src(i, j) = localMaximal[index + 1];
						else
							src(i, j) = localMaximal[index];
						break;
					}
				}
			}
		}

		imshow("Multilevel thresholding", src);
	}

}

void floydSteinbergDithering()
{
	int WH = 5;
	int windowWidth = 2 * WH + 1;
	float TH = 0.0003;
	char fname[MAX_PATH];
	int hg[256] = { 0 };
	float p[256] = { 0 };
	int localMaximal[100];

	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hg[src(i, j)]++;
			}
		}

		for (int i = 0; i < 256; i++)
		{
			p[i] = (float)hg[i] / (height * width);
		}

		int k = 0;
		localMaximal[0] = 0;
		for (int i = WH; i <= 255 - WH; i++)
		{
			float avg = 0;
			for (int j = i - WH; j <= i + WH; j++)
			{
				avg += p[j];
			}
			avg = avg / (2 * WH + 1);

			if ((maxInInterval(p, i, WH)) && (p[i] > TH + avg))
			{
				k++;
				localMaximal[k] = i;
			}

		}

		localMaximal[++k] = 255;
		k++;

		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				int oldPixel = src(i, j);
				int newPixel;
				for (int index = 0; index < k - 1; index++)
				{
					if (oldPixel >= localMaximal[index] && oldPixel <= localMaximal[index + 1])
					{
						int difference1 = abs(oldPixel - localMaximal[index]);
						int difference2 = abs(oldPixel - localMaximal[index + 1]);
						if (difference1 > difference2)
						{
							newPixel = localMaximal[index + 1];
						}
						else
						{
							newPixel = localMaximal[index];
						}
						break;
					}
				}
				src(i,j) = newPixel;
				
				float error = oldPixel - newPixel;

				src(i, j + 1) = max(0, min((src(i, j + 1) + 7 * error / 16.0), 255));
				src(i + 1, j - 1) = max(0, min((src(i + 1, j - 1) + 3 * error / 16.0), 255));
				src(i + 1, j) = max(0, min((src(i + 1, j) + 5 * error / 16.0), 255));
				src(i + 1, j + 1) = max(0, min((src(i + 1, j + 1) + error / 16.0), 255));
			}

		}

		imshow("Floyd-Steinberg Dithering", src);
	}
}


void geometricPropertiesCallback(int event, int x, int y, int flags, void* param)
{
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDBLCLK)
	{
		Vec3b color = (*src).at<Vec3b>(y, x);
		int area = 0;
		int height = (*src).rows;
		int width = (*src).cols;

		// Compute area
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if ((*src).at<Vec3b>(i, j) == color)
				{
					area++;
				}
			}
		}


		// Compute center of mass
		int centerMassRow = 0, centerMassColumn = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if ((*src).at<Vec3b>(i, j) == color)
				{
					//	Correct mathemathical order
					//centerMassRow += j;
					//centerMassColumn += i;
					
					// Correct because OpenCV is weird
					centerMassRow += i;
					centerMassColumn += j;
				}
			}
		}
		centerMassRow = ((1.0f / area) * centerMassRow);
		centerMassColumn = ((1.0f / area) * centerMassColumn);

		// Compute axis of elongation
		float nominator = 0;
		float denom1 = 0;
		float denom2 = 0;
		float denominator = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if ((*src).at<Vec3b>(i, j) == color)
				{
					nominator += (i - centerMassRow) * (j - centerMassColumn);
					denom1 += (j - centerMassColumn) * (j - centerMassColumn);
					denom2 += (i - centerMassRow) * (i - centerMassRow);
				}
			}
		}
		nominator *= 2;
		denominator = denom1 - denom2;
		float phiAngle = atan2(nominator, denominator) / 2;
		//If negative get into positive
		if (phiAngle < 0)
		{
			phiAngle += PI;
		}
		// To transform into degrees
		float phiAngleInDegrees = phiAngle * 180 / PI; 
		

		// Compute perimeter
		float perimeter = 0;
		Mat contour(height, width, CV_8UC3, CV_RGB(255, 255, 255));
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if ((*src).at<Vec3b>(i, j) == color)
				{
					Vec3b top = (*src).at<Vec3b>(i - 1, j);
					Vec3b down = (*src).at<Vec3b>(i + 1, j);
					Vec3b left = (*src).at<Vec3b>(i, j - 1);
					Vec3b right = (*src).at<Vec3b>(i, j + 1);
					Vec3b topLeft = (*src).at<Vec3b>(i - 1, j - 1);
					Vec3b topRight = (*src).at<Vec3b>(i - 1, j + 1);
					Vec3b downLeft = (*src).at<Vec3b>(i + 1, j - 1);
					Vec3b downRight = (*src).at<Vec3b>(i + 1, j + 1);

					if (top != color || down != color || left != color || right != color || topLeft != color || topRight != color || downLeft != color || downRight != color)
					{
						perimeter++;
						contour.at<Vec3b>(i, j) = (*src).at<Vec3b>(i, j);
					}
				}
			}
		}
		perimeter *= PI / 4.0f;
		//Add center of mass in countour projection
		contour.at<Vec3b>(centerMassRow, centerMassColumn) = color;
		


		// Compute thinness ration(circularity)
		float thinnessRatio = 4 * PI * (((float)area) / (perimeter * perimeter));

		// Compute aspect ratio
		int maxRow = INT_MIN;
		int maxColumn = INT_MIN;
		int minRow = INT_MAX;
		int minColumn = INT_MAX;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if ((*src).at<Vec3b>(i, j) == color)
				{
					maxRow = (i > maxRow) ? i : maxRow;
					minRow = (i < minRow) ? i : minRow;
					maxColumn = (j > maxColumn) ? j : maxColumn;
					minColumn = (j < minColumn) ? j : minColumn;
				}
			}
		}
		float aspectRatio = (float)(maxColumn - minColumn + 1) / (maxRow - minRow + 1);

		//Add axis of elongation in contour projection
		int rowA = centerMassRow + tan(phiAngle) * (minColumn - centerMassColumn);
		int rowB = centerMassRow + tan(phiAngle) * (maxColumn - centerMassColumn);
		int columnA = minColumn;
		int columnB = maxColumn;
		Point A(columnA, rowA);
		Point B(columnB, rowB);
		line(contour, A, B, (0,0,0), 2);

		// Compute projections of binary objects
		Mat horizontalProjection(height, width, CV_8UC3, CV_RGB(255, 255, 255));
		Mat verticalProjection(height, width, CV_8UC3, CV_RGB(255, 255, 255));
		for (int i = 0; i < height; i++)
		{
			int index = 0;
			for (int j = 0; j < width; j++)
			{
				if ((*src).at<Vec3b>(i, j) == color)
				{
					horizontalProjection.at<Vec3b>(i, index) = color;
					index++;
				}
			}
		}
		
		for (int j = 0; j < width; j++)
		{
			int index = 0;
			for (int i = 0; i < height; i++)
			{
				if ((*src).at<Vec3b>(i, j) == color)
				{
					verticalProjection.at<Vec3b>(index, j) = color;
					index++;
				}
			}
		}

		printf("Area = %d\n", area);
		printf("Center of mass:\n\tRow:%d\n\tColumn:%d\n", centerMassRow, centerMassColumn);
		printf("Axis of elongation Phi angle in degrees: %f\n", phiAngleInDegrees);
		printf("Perimeter: %f\n", perimeter);
		printf("Thinness ratio: %f\n", thinnessRatio);
		printf("Aspect ratio: %f\n", aspectRatio);

		imshow("Contour", contour);
		imshow("Horizontal Projection", horizontalProjection);
		imshow("Vertical Projection", verticalProjection);
		printf("\n");
	}
}


void geometricProperties()
{
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_COLOR);

		namedWindow("Geometric Properties", 1);

		//set the callback function for any mouse event
		setMouseCallback("Geometric Properties", geometricPropertiesCallback, &src);

		imshow("Geometric Properties", src);

		waitKey(0);
	}
}


void colorObjects(Mat labels)
{
	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);
	int height = labels.rows;
	int width = labels.cols;

	std::vector<int> labelsVector;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			labelsVector.push_back(labels.at<int>(i, j));
		}
	}

	sort(labelsVector.begin(), labelsVector.end());
	labelsVector.erase(unique(labelsVector.begin(), labelsVector.end()), labelsVector.end());

	Mat dst(height, width, CV_8UC3);

	for (std::vector<int>::size_type k = 1; k != labelsVector.size(); k++)
	{
		int r = d(gen);
		int g = d(gen);
		int b = d(gen);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labels.at<int>(i, j) == k)
				{
					dst.at<Vec3b>(i, j)[2] = r;
					dst.at<Vec3b>(i, j)[1] = g;
					dst.at<Vec3b>(i, j)[0] = b;
				}
				else if (labels.at<int>(i, j) == 0)
				{
					dst.at<Vec3b>(i, j)[2] = 255;
					dst.at<Vec3b>(i, j)[1] = 255;
					dst.at<Vec3b>(i, j)[0] = 255;
				}
			}
		}
	}

	imshow("Labeled", dst);
}

void colorObjectsAux(Mat labels, std::vector<int> newLabels)
{
	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);
	int height = labels.rows;
	int width = labels.cols;

	Mat dst(height, width, CV_8UC3);

	for (std::vector<int>::size_type k = 1; k < newLabels.size(); k++)
	{
		int r = d(gen);
		int g = d(gen);
		int b = d(gen);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labels.at<int>(i, j) == k)
				{
					dst.at<Vec3b>(i, j)[2] = r;
					dst.at<Vec3b>(i, j)[1] = g;
					dst.at<Vec3b>(i, j)[0] = b;
				}
				else if (labels.at<int>(i, j) == 0)
				{
					dst.at<Vec3b>(i, j)[2] = 255;
					dst.at<Vec3b>(i, j)[1] = 255;
					dst.at<Vec3b>(i, j)[0] = 255;
				}
			}
		}
	}

	imshow("Destination", dst);
}


void labelBinaryImages()
{
	char fname[MAX_PATH];
	Mat src;
	int di[8] = { -1, 0, 1, 0, -1, 1, -1, 1 };
	int dj[8] = { 0, -1, 0, 1, 1, -1, -1, 1 };
	uchar neighbors[8];
	int height, width;
	

	while (openFileDlg(fname))
	{
		int label = 0;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		height = src.rows;
		width = src.cols;
		Mat labels = Mat::zeros(height, width, CV_32SC1);

		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if ((src.at<uchar>(i, j) == 0) && (labels.at<int>(i, j) == 0))
				{
					label++;
					std::queue<Point2i> Q;
					labels.at<int>(i, j) = label;
					Q.push({ i,j });
					while (!Q.empty())
					{
						Point2i q = Q.front();
						Q.pop();
						
						for (int k = 0; k < 8; k++)
						{
							if (q.x + di[k] <= height - 1 && q.x + di[k] >= 0 && q.y + dj[k] >= 0 && q.y + dj[k] <= width - 1)
							{
								neighbors[k] = src.at<uchar>(q.x + di[k], q.y + dj[k]);
								//uchar neighborX = q.x + di[k];
								//uchar neighborY = q.y + dj[k];

								if ((neighbors[k] == 0) && (labels.at<int>(q.x + di[k], q.y + dj[k]) == 0))
								{
									labels.at<int>(q.x + di[k], q.y + dj[k]) = label;
									Q.push({ q.x + di[k], q.y + dj[k] });
								}
							}

						}
					}
				}
			}
		}

		
		colorObjects(labels);
		imshow("Source", src);
	}
}

int minOf(std::vector<int> numbers)
{
	int min = numbers.at(0);
	for (int i = 0; i < numbers.size(); i++)
	{
		if (numbers.at(i) < min)
		{
			min = numbers.at(i);
		}
	}

	return min;
}


void labelBinaryImagesWithTwoPass()
{
	char fname[MAX_PATH];
	Mat src;
	int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	uchar neighbors[8];
	int height, width;
	int label = 0;

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		height = src.rows;
		width = src.cols;
		Mat labels = Mat::zeros(height, width, CV_32SC1);
		std::vector<std::vector<int>> edges;
	
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0)
				{
					std::vector<int> L;
					for (int k = 0; k < 4; k++)
					{
						//neighbors[k] = src.at<uchar>(i + di[k],j + dj[k]);
						if (labels.at<int>(i + di[k], j + dj[k]) > 0)
						{
							L.push_back(labels.at<int>(i + di[k], j + dj[k]));
						}
					}
					if (L.size() == 0)
					{
						label++;
						edges.resize(label + 1);
						labels.at<int>(i, j) = label;
					} else
					{
						int x = minOf(L);
						labels.at<int>(i, j) = x;
						for (int index0 = 0; index0 < L.size(); index0++)
						{
							if (L.at(index0) != x)
							{
								edges[x].push_back(L.at(index0));
								edges[L.at(index0)].push_back(x);
							}
						}
					}

				}
			}
		}
	
		int newLabel = 0;
		std::vector<int> newLabels;
		for (int i = 0; i < label + 1; i++)
		{
			newLabels.push_back(0);
		}

		for (int i = 1; i < label; i++)
		{
			if (newLabels.at(i) == 0)
			{
				newLabel++;
				std::queue<int> Q;
				newLabels.at(i) = newLabel;
				Q.push(i);
				while (!Q.empty())
				{
					int x = Q.front();
					Q.pop();
					for (int k = 0; k < edges[x].size(); k++)
					{
						if (newLabels.at(edges[x].at(k)) == 0)
						{
							newLabels.at(edges[x].at(k)) = newLabel;
							Q.push(edges[x].at(k));
						}
					}
				}
			}
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				labels.at<int>(i, j) = newLabels.at(labels.at<int>(i, j));
			}
		}


		colorObjects(labels);
		imshow("Source", src);
	}

	
}


void borderTracingAlgorithm()
{
	char fname[MAX_PATH];
	const int OBJECT_PIXEL = 0;
	Mat src;
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC1, RGB(255, 255, 255));
		int dir;

		std::vector<Point2i> allPixels;

		dir = 7; // 8 connectivity

		bool firstPixelFound = false;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == OBJECT_PIXEL)
				{
					dst.at<uchar>(i, j) = OBJECT_PIXEL;
					firstPixelFound = true;
					allPixels.push_back(Point2i(i, j));
					break;
				}

			}
		
			if (firstPixelFound)
			{
				break;
			}
		}

		Point2i pixel = allPixels.at(0);

		while (1)
		{
			int startingIndex;
			if (dir % 2 == 0)
			{
				dir = (dir + 7) % 8;
			}
			else
			{
				dir = (dir + 6) % 8;
			}

			while (src.at<uchar>(pixel.x + di[dir], pixel.y + dj[dir]) != OBJECT_PIXEL)
			{
				dir = (dir + 1) % 8;
			}

			pixel.x += di[dir];
			pixel.y += dj[dir];
			allPixels.push_back(pixel);
			dst.at<uchar>(pixel.x, pixel.y) = OBJECT_PIXEL;


			if (allPixels.size() >= 4)
				if (allPixels[0] == allPixels[allPixels.size() - 2] && allPixels[1] == allPixels[allPixels.size() - 1])
					break;

		}


		std::vector<int> chainCode;
		for (int i = 0; i < allPixels.size() - 1; i++)
		{
			Point2i currentPixel = allPixels.at(i);
			Point2i nextPixel = allPixels.at(i + 1);
			for (int j = 0; j < 8; j++)
			{
				if (currentPixel.x + di[j] == nextPixel.x && currentPixel.y + dj[j] == nextPixel.y)
				{
					chainCode.push_back(j);
				}
			}
		}

		// Pop the 2 last values because they are the same as the first 2
		chainCode.pop_back();
		chainCode.pop_back();

		printf("\nChain code: ");
		for (int i = 0; i < chainCode.size(); i++)
		{
			printf("%d ", chainCode.at(i));
		}
		printf("\n");

		std::vector<int> derivativeChainCode;
		
		int lastDerivativeChainCodeNumber;
		int lastChainCodeNumber = chainCode.at(chainCode.size() - 1);
		int firstChainCodeNumber = chainCode.at(0);

		if (lastChainCodeNumber <= firstChainCodeNumber)
			lastDerivativeChainCodeNumber = firstChainCodeNumber - lastChainCodeNumber;
		else
			lastDerivativeChainCodeNumber = 8 - abs(firstChainCodeNumber - lastChainCodeNumber);
	
		for (int i = 0; i < chainCode.size() - 1; i++)
		{
			int currentChainNumber = chainCode.at(i);
			int nextChainNumber = chainCode.at(i + 1);
			if (currentChainNumber <= nextChainNumber)
				derivativeChainCode.push_back(nextChainNumber - currentChainNumber);
			else
				derivativeChainCode.push_back(8 - abs(nextChainNumber - currentChainNumber));
		}

		derivativeChainCode.push_back(lastDerivativeChainCodeNumber);

		printf("Derivative Chain Code: ");
		for (int i = 0; i < derivativeChainCode.size(); i++)
		{
			printf("%d ", derivativeChainCode.at(i));
		}
		printf("\n");

		printf("Chain size: %d\n Derivative chain size: %d\n", chainCode.size(), derivativeChainCode.size());


		imshow("Source", src);
		imshow("Destination", dst);
	}
}


void reconstructFromFile()
{
	char fname[MAX_PATH];
	const int OBJECT_PIXEL = 0;
	Mat image;
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	while (openFileDlg(fname))
	{

		image = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = image.rows;
		int width = image.cols;
		
		FILE* filePointer = fopen("reconstruct.txt", "r");

		int readX = -1, readY = -1;
		fscanf(filePointer, "%d %d\n", &readX, &readY);
		Point2i pixel(readX, readY);

		int lengthOfChain = -1;
		fscanf(filePointer, "%d\n", &lengthOfChain);
		std::vector<int> chainCode;
		for (int i = 0; i < lengthOfChain; i++)
		{
			int currentElem = -1;
			fscanf(filePointer, "%d ", &currentElem);
			chainCode.push_back(currentElem);
		}

		image.at<uchar>(readX, readY) = OBJECT_PIXEL;
		for (int i = 0; i < lengthOfChain; i++)
		{
			for (int k = 0; k < 8; k++)
			{
				if (chainCode.at(i) == k)
				{
					pixel.x += di[k];
					pixel.y += dj[k];
					image.at<uchar>(pixel.x, pixel.y) = OBJECT_PIXEL;
					break;
				}
			}
		}


		imshow("Dst", image);
	}
}

Mat dilateNTimes(Mat src, int n)
{
	int height = src.rows;
	int width = src.cols;
	Mat newSource = src.clone();
	
	Mat dst(height, width, CV_8UC1, RGB(255, 255, 255));
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	const int OBJECT_PIXEL = 0;

	for (int nTimes = 0; nTimes < n; nTimes++)
	{
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (newSource.at<uchar>(i, j) == OBJECT_PIXEL)
				{
					dst.at<uchar>(i, j) = OBJECT_PIXEL;
					for (int k = 0; k < 8; k++)
					{
						dst.at<uchar>(i + di[k], j + dj[k]) = OBJECT_PIXEL;
					}
				}
			}
		}

		newSource = dst.clone();
	}

	return dst;
}





void dilate()
{
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		int n = 0;
		printf("Dilate N times.\nn = ");
		scanf("%d", &n);
		Mat dst = dilateNTimes(src, n);

		imshow("Dilate", dst);
		imshow("Source", src);		
	}

}

Mat erodeNTimes(Mat src, int n)
{
	int height = src.rows;
	int width = src.cols;
	Mat newSource = src.clone();

	Mat dst(height, width, CV_8UC1, RGB(255, 255, 255));
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	const int OBJECT_PIXEL = 0;
	const int BACKGROUND_PIXEL = 255;
	bool allObjectPixels = false;
	bool oneBGPixelFound = false;

	for (int nTimes = 0; nTimes < n; nTimes++)
	{
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (newSource.at<uchar>(i, j) == OBJECT_PIXEL)
				{
					for (int k = 0; k < 8; k++)
					{
						if (newSource.at<uchar>(i + di[k], j + dj[k]) == BACKGROUND_PIXEL)
						{
							oneBGPixelFound = true;
							break;
						}
					}

					if (!oneBGPixelFound)
					{
						dst.at<uchar>(i, j) = OBJECT_PIXEL;
					}
					else
					{
						dst.at<uchar>(i, j) = BACKGROUND_PIXEL;
					}
				}
				else
				{
					dst.at<uchar>(i, j) = BACKGROUND_PIXEL;
				}

				oneBGPixelFound = false;
			}
		}


		newSource = dst.clone();
	}

	return dst;
}


void erode()
{
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		int n = 0;
		printf("Erode N times.\nn = ");
		scanf("%d", &n);
		Mat dst = erodeNTimes(src, n);

		imshow("Erode", dst);
		imshow("Source", src);
	}
}

void open1()
{
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat eroded = erodeNTimes(src, 1);
		Mat dilated = dilateNTimes(eroded, 1);

		imshow("Open", dilated);
		imshow("Source", src);
	}
}

void close1()
{
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dilated = dilateNTimes(src, 1);
		Mat eroded = erodeNTimes(dilated, 1);

		imshow("Close", eroded);
		imshow("Source", src);
	}

}


void boundaryExtraction()
{
	char fname[MAX_PATH];
	Mat src;
	const int OBJECT_PIXEL = 0;

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat eroded = erodeNTimes(src, 1);
		Mat boundaryExtraction(height,width, CV_8UC1, RGB(255, 255, 255));

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) != eroded.at<uchar>(i, j))
				{
					boundaryExtraction.at<uchar>(i, j) = OBJECT_PIXEL;
				}
			}
		}

		imshow("Boundary Extraction", boundaryExtraction);
		imshow("Source", src);
	}
}

Mat intersect(Mat a, Mat b)
{
	int height = a.rows;
	int width = a.cols;
	Mat dst(height, width, CV_8UC1, RGB(255, 255, 255));

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (a.at<uchar>(i, j) == 0 && b.at<uchar>(i, j) == 0)
				dst.at<uchar>(i, j) = 0;
		}
	}

	return dst;
}

Mat complement(Mat a)
{
	int height = a.rows;
	int width = a.cols;
	Mat dst(height, width, CV_8UC1, RGB(255,255,255));
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (a.at<uchar>(i, j) == 0)
				dst.at<uchar>(i, j) = 255;
			else if (a.at<uchar>(i, j) == 255)
				dst.at<uchar>(i, j) = 0;
		}
	}
	return dst;
}

bool areSame(Mat a, Mat b)
{
	int height = a.rows;
	int width = a.cols;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (a.at<uchar>(i, j) != b.at<uchar>(i, j))
				return false;
		}
	}

	return true;
}


Mat reunion(Mat a, Mat b)
{
	int height = a.rows;
	int width = a.cols;
	Mat dst = a.clone();
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (b.at<uchar>(i, j) == 0)
				dst.at<uchar>(i, j) = 0;
		}
	}

	return dst;
}


void regionFilling()
{
	char fname[MAX_PATH];
	Mat src;
	const int OBJECT_PIXEL = 0;

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat matrixXOfKMinusOne(height, width, CV_8UC1, RGB(255, 255, 255));
		matrixXOfKMinusOne.at<uchar>(height / 2, width / 2) = OBJECT_PIXEL;

		Mat matrixXOfK(height, width, CV_8UC1, RGB(255, 255, 255));		
		int k = 0;
		while (1)
		{
			k++;
			Mat dilatedMatrixXOfKMinusOne = dilateNTimes(matrixXOfKMinusOne, 1);
			Mat complementOfSetA = complement(src);
			matrixXOfK = intersect(dilatedMatrixXOfKMinusOne, complementOfSetA);

			if (areSame(matrixXOfK, matrixXOfKMinusOne))
				break;

			matrixXOfKMinusOne = matrixXOfK.clone();
		}

		Mat reunitedWithSrc = reunion(matrixXOfK, src);
		
		printf("k = %d\n", k);
		imshow("Region filled", reunitedWithSrc);
		imshow("src", src);
	}
}

Mat computeOpenedImage(Mat a)
{
	Mat aClone = a.clone();
	Mat eroded = erodeNTimes(aClone, 1);
	Mat dilated = dilateNTimes(eroded, 1);
	return dilated;
}

Mat computeClosedImage(Mat a)
{
	Mat aClone = a.clone();
	Mat dilated = dilateNTimes(aClone, 1);
	Mat eroded = erodeNTimes(dilated, 1);
	return eroded;
}

Mat labelImage(Mat labels)
{
	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);
	int height = labels.rows;
	int width = labels.cols;

	std::vector<int> labelsVector;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			labelsVector.push_back(labels.at<int>(i, j));
		}
	}

	sort(labelsVector.begin(), labelsVector.end());
	labelsVector.erase(unique(labelsVector.begin(), labelsVector.end()), labelsVector.end());

	Mat dst(height, width, CV_8UC3);

	for (std::vector<int>::size_type k = 1; k != labelsVector.size(); k++)
	{
		int r = d(gen);
		int g = d(gen);
		int b = d(gen);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labels.at<int>(i, j) == k)
				{
					dst.at<Vec3b>(i, j)[2] = r;
					dst.at<Vec3b>(i, j)[1] = g;
					dst.at<Vec3b>(i, j)[0] = b;
				}
				else if (labels.at<int>(i, j) == 0)
				{
					dst.at<Vec3b>(i, j)[2] = 255;
					dst.at<Vec3b>(i, j)[1] = 255;
					dst.at<Vec3b>(i, j)[0] = 255;
				}
			}
		}
	}

	return dst;
}

Mat computeLabeledImage(Mat labels)
{
	srand(time(NULL));


	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);

	int height = labels.rows;
	int width = labels.cols;

	std::vector<int> labelsVector;
	labelsVector.push_back(labels.at<int>(0, 0));

	bool differentFromAllValues;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			differentFromAllValues = true;
			for (int k = 0; k < labelsVector.size(); k++)
			{
				if (labels.at<int>(i, j) == labelsVector.at(k))
				{
					differentFromAllValues = false;
					break;
				}
			}
			if (differentFromAllValues)
				labelsVector.push_back(labels.at<int>(i, j));	
		}
	}

	labelsVector.erase(labelsVector.begin());

	Mat dst(height, width, CV_8UC3);

	for (int k = 0; k < labelsVector.size(); k++)
	{
		int r = d(gen);
		int g = d(gen);
		int b = d(gen);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labels.at<int>(i, j) == labelsVector.at(k))
				{
					dst.at<Vec3b>(i, j)[2] = r;
					dst.at<Vec3b>(i, j)[1] = g;
					dst.at<Vec3b>(i, j)[0] = b;
				}
				else if (labels.at<int>(i, j) == 0)
				{
					dst.at<Vec3b>(i, j)[2] = 0;
					dst.at<Vec3b>(i, j)[1] = 0;
					dst.at<Vec3b>(i, j)[0] = 0;
				}
			}
		}
	}

	return dst;
}



std::vector<Vec3b> computeObjectsColorsWhiteBackground(Mat src)
{
	int height = src.rows;
	int width = src.cols;
	const int BACKGROUND_PIXEL = 255;

	std::vector<Vec3b> availableColors;
	availableColors.push_back(Vec3b(255, 255, 255));
	bool differentFromAllColors;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			differentFromAllColors = true;
			for (int k = 0; k < availableColors.size(); k++)
			{
				if (src.at<Vec3b>(i, j) == availableColors.at(k))
				{
					differentFromAllColors = false;
					break;
				}
			}

			if (differentFromAllColors)
				availableColors.push_back(src.at<Vec3b>(i, j));
		}
	}

	// Delete the white color
	availableColors.erase(availableColors.begin());

	return availableColors;
}

std::vector<Vec3b> computeObjectsColorsBlackBackground(Mat src)
{
	int height = src.rows;
	int width = src.cols;
	const int BACKGROUND_PIXEL = 255;

	std::vector<Vec3b> availableColors;
	availableColors.push_back(Vec3b(0, 0, 0));
	bool differentFromAllColors;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			differentFromAllColors = true;
			for (int k = 0; k < availableColors.size(); k++)
			{
				if (src.at<Vec3b>(i, j) == availableColors.at(k))
				{
					differentFromAllColors = false;
					break;
				}
			}

			if (differentFromAllColors)
				availableColors.push_back(src.at<Vec3b>(i, j));
		}
	}

	// Delete the black color
	availableColors.erase(availableColors.begin());

	return availableColors;
}



Mat removeSomeObjects(Mat labeled, std::vector<Vec3b> colorsVector)
{
	std::vector<Vec3b> acceptedObjectsColors;
	int height = labeled.rows;
	int width = labeled.cols;

	for (int k = 0; k < colorsVector.size(); k++)
	{
		Vec3b currentColor = colorsVector.at(k);
		
		// Compute area
		int area = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labeled.at<Vec3b>(i, j) == currentColor)
				{
					area++;
				}
			}
		}

		// Compute center of mass
		int centerMassRow = 0, centerMassColumn = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labeled.at<Vec3b>(i, j) == currentColor)
				{
					centerMassRow += i;
					centerMassColumn += j;
				}
			}
		}
		centerMassRow = ((1.0f / area) * centerMassRow);
		centerMassColumn = ((1.0f / area) * centerMassColumn);

		// Compute axis of elongation
		float nominator = 0;
		float denom1 = 0;
		float denom2 = 0;
		float denominator = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labeled.at<Vec3b>(i, j) == currentColor)
				{
					nominator += (i - centerMassRow) * (j - centerMassColumn);
					denom1 += (j - centerMassColumn) * (j - centerMassColumn);
					denom2 += (i - centerMassRow) * (i - centerMassRow);
				}
			}
		}
		nominator *= 2;
		denominator = denom1 - denom2;
		float phiAngle = atan2(nominator, denominator) / 2;
		//If negative get into positive
		if (phiAngle < 0)
		{
			phiAngle += PI;
		}
		// To transform into degrees
		float phiAngleInDegrees = phiAngle * 180 / PI;



		// Compute aspect ratio
		int maxRow = INT_MIN;
		int maxColumn = INT_MIN;
		int minRow = INT_MAX;
		int minColumn = INT_MAX;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labeled.at<Vec3b>(i, j) == currentColor)
				{
					maxRow = (i > maxRow) ? i : maxRow;
					minRow = (i < minRow) ? i : minRow;
					maxColumn = (j > maxColumn) ? j : maxColumn;
					minColumn = (j < minColumn) ? j : minColumn;
				}
			}
		}
		float aspectRatio = (float)(maxColumn - minColumn + 1) / (maxRow - minRow + 1);

		int currentObjectHeight = maxRow - minRow;
		int currentObjectWidth = maxColumn - minColumn;

		
		printf("\n");
		printf("Phi angle: %f\n", phiAngleInDegrees);
		printf("Height = %d\n", currentObjectHeight);
		printf("Width = %d\n", currentObjectWidth);
		printf("Aspert ratio = %f\n", aspectRatio);
		
		//Final decision if
		//Add only items based on some criteria
		//if (aspectRatio >= -1000.0 && aspectRatio <= 1000.0 && phiAngleInDegrees <= 50.0 && phiAngleInDegrees >= -50.0)
		//if((currentObjectHeight < 500 && currentObjectWidth < 1500) && (currentObjectHeight > 0 && currentObjectWidth > 0))
		//if ((phiAngleInDegrees >= 175.00 || phiAngleInDegrees <= 5.0) && currentObjectHeight > height / 8 && currentObjectWidth > width / 8 && currentObjectWidth > currentObjectHeight && aspectRatio >= 3 && aspectRatio <= 4)
		
		if ((phiAngleInDegrees >= 175.00 || phiAngleInDegrees <= 5.0) && aspectRatio >= 3 && aspectRatio <= 4)
		{
			acceptedObjectsColors.push_back(currentColor);
		}

	}

	Mat result = labeled.clone();
	bool foundAcceptableColor;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			foundAcceptableColor = false;
			for (int k = 0; k < acceptedObjectsColors.size(); k++)
			{
				if (result.at<Vec3b>(i, j) == acceptedObjectsColors.at(k))
				{
					foundAcceptableColor = true;
					break;
				}
			}

			if (!foundAcceptableColor)
				result.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
		}
	}

	return result;
}

Mat highlightPossiblePlates(Mat srcColor, Mat coloredPlates, std::vector<Vec3b> allPossiblePlatesColors)
{
	int height = coloredPlates.rows;
	int width = coloredPlates.cols;

	for (int k = 0; k < allPossiblePlatesColors.size(); k++)
	{
		Vec3b currentColor = allPossiblePlatesColors.at(k);

		// Compute aspect ratio
		int maxRow = INT_MIN;
		int maxColumn = INT_MIN;
		int minRow = INT_MAX;
		int minColumn = INT_MAX;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (coloredPlates.at<Vec3b>(i, j) == currentColor)
				{
					maxRow = (i > maxRow) ? i : maxRow;
					minRow = (i < minRow) ? i : minRow;
					maxColumn = (j > maxColumn) ? j : maxColumn;
					minColumn = (j < minColumn) ? j : minColumn;
				}
			}
		}

		Point topLeft(minColumn, minRow);
		Point bottomLeft(minColumn, maxRow);
		Point topRight(maxColumn, minRow);
		Point bottomRight(maxColumn, maxRow);


		line(srcColor, topLeft, topRight, Scalar(0, 0, 255), 3);
		line(srcColor, topLeft, bottomLeft, Scalar(0, 0, 255), 3);
		line(srcColor, bottomLeft, bottomRight, Scalar(0, 0, 255), 3);
		line(srcColor, bottomRight, topRight, Scalar(0, 0, 255), 3);
	}

	return srcColor;
}



void project()
{
	char fname[MAX_PATH];
	Mat src;
	Mat srcColor;
	int di[8] = { -1, 0, 1, 0, -1, 1, -1, 1 };
	int dj[8] = { 0, -1, 0, 1, 1, -1, -1, 1 };
	uchar neighbors[8];

	while (openFileDlg(fname))
	{
		srcColor = imread(fname, CV_LOAD_IMAGE_COLOR);
		imshow("Source", srcColor);

		// Grayscale Image
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat grayScaleImage = src;
		imshow("Grayscale", grayScaleImage);

		// B&W Image
		const int THRESHOLD = 128;
		const int BLACK = 0;
		const int WHITE = 255;
		Mat blackAndWhiteImage(height, width, CV_8UC1, RGB(255, 255, 255));
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (grayScaleImage.at<uchar>(i, j) > THRESHOLD)
				{
					blackAndWhiteImage.at<uchar>(i, j) = WHITE;
				}
				else
				{
					blackAndWhiteImage.at<uchar>(i, j) = BLACK;
				}
			}
		}

		char buffer[50];
		sprintf(buffer, "B&W Image w/ Threshold %d", THRESHOLD);
		imshow(buffer, blackAndWhiteImage);

		// Edge detection
		Mat cannyImage(height, width, CV_8UC1);
		Canny(blackAndWhiteImage, cannyImage, 100, 200);
		imshow("Canny", cannyImage);

		// Open
		//Mat openedImage(height, width, CV_8UC1);
		//openedImage = computeOpenedImage(complement(cannyImage));
		//openedImage = complement(openedImage);
		//imshow("Open", openedImage);

		//Dilate
		Mat dilatedImage(height, width, CV_8UC1);
		dilatedImage = dilateNTimes(complement(cannyImage), 6);
		dilatedImage = complement(dilatedImage);
		imshow("Dilated", dilatedImage);

		//Connected-component labeling
		int label = 0;
		Mat labels = Mat::zeros(height, width, CV_32SC1);
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if ((dilatedImage.at<uchar>(i, j) == 255) && (labels.at<int>(i, j) == 0))
				{
					label++;
					std::queue<Point2i> Q;
					labels.at<int>(i, j) = label;
					Q.push({ i,j });
					while (!Q.empty())
					{
						Point2i q = Q.front();
						Q.pop();

						for (int k = 0; k < 8; k++)
						{
							if (q.x + di[k] <= height - 1 && q.x + di[k] >= 0 && q.y + dj[k] >= 0 && q.y + dj[k] <= width - 1)
							{
								neighbors[k] = dilatedImage.at<uchar>(q.x + di[k], q.y + dj[k]);

								if ((neighbors[k] == 255) && (labels.at<int>(q.x + di[k], q.y + dj[k]) == 0))
								{
									labels.at<int>(q.x + di[k], q.y + dj[k]) = label;
									Q.push({ q.x + di[k], q.y + dj[k] });
								}
							}
						}
					}
				}
			}
		}

		// Color labeled
		Mat labeledImage = computeLabeledImage(labels);
		imshow("Labeled", labeledImage);
		std::vector<Vec3b> colorsOfObjects = computeObjectsColorsWhiteBackground(labeledImage);
		printf("Nb of objects before filtering: %d\n", colorsOfObjects.size());

		// Filter
		Mat removed = removeSomeObjects(labeledImage, colorsOfObjects);
		imshow("Filtered", removed);

		std::vector<Vec3b> colorsOfObjectsAfterFiltering = computeObjectsColorsWhiteBackground(removed);
		printf("Nb of objects before filtering: %d\n", colorsOfObjectsAfterFiltering.size());

		// Highlight
		Mat highlighted = srcColor.clone();
		highlighted = highlightPossiblePlates(highlighted, removed, colorsOfObjectsAfterFiltering);
		imshow("Highlighted", highlighted);
	}

}

void statisticalProperties()
{
	char fname[MAX_PATH];
	Mat src;
	int di[8] = { -1, 0, 1, 0, -1, 1, -1, 1 };
	int dj[8] = { 0, -1, 0, 1, 1, -1, -1, 1 };
	uchar neighbors[8];
	const int OBJECT_PIXEL = 0;
	const int BACKGROUND_PIXEL = 255;

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Source", src);
		int height = src.rows;
		int width = src.cols;

		int maxIntensity = src.at<uchar>(0, 0);
		int minIntensity = src.at<uchar>(0, 0);
		

		// Mean
		int I = 0;
		int M = height * width;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				I += src.at<uchar>(i, j);
				if (src.at<uchar>(i, j) > maxIntensity)
					maxIntensity = src.at<uchar>(i, j);
				if (src.at<uchar>(i, j) <= minIntensity)
					minIntensity = src.at<uchar>(i, j);
			}
		}

		float mean = (float)I / M;
		printf("Mean = %f\n", mean);

		// Standard deviation
		float sum = 0.0f;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				float toPower = (src.at<uchar>(i, j) - mean) * (src.at<uchar>(i, j) - mean);
				sum += toPower;
			}
		}

		float standardDeviation = sqrt(sum / (float)M);
		printf("Standard deviation = %f\n", standardDeviation);


		// Histograms
		int hg[256] = { 0 };

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hg[src.at<uchar>(i, j)]++;
			}
		}

		
		float cumulativeP[256] = { 0 };
		for (int i = 0; i < 256; i++)
		{
			cumulativeP[i] = (float)hg[i] / (height * width);
		}

		for (int i = 1; i < 256; i++)
		{
			cumulativeP[i] = cumulativeP[i - 1] + ((float)hg[i] / (height * width));
		}

		showHistogram("Histogram", hg, 256, 256);


		// 
		float threshold = ((float)minIntensity + maxIntensity) / 2;
		while(1)
		{
			float N1 = 0.0f;
			float N2 = 0.0f;
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					if (src.at<uchar>(i, j) >= minIntensity && src.at<uchar>(i, j) <= threshold)
					{
						N1 += hg[src.at<uchar>(i, j)];
					}
					else if (src.at<uchar>(i, j) > threshold && src.at<uchar>(i, j) <= maxIntensity)
					{
						N2 += hg[src.at<uchar>(i, j)];
					}
				}
			}

			float meanG1 = 0.0f;
			float meanG2 = 0.0f;
			for(int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					if (src.at<uchar>(i,j) >= minIntensity && src.at<uchar>(i, j) <= threshold)
					{
						meanG1 += src.at<uchar>(i, j) * hg[src.at<uchar>(i, j)];
					}
					else if (src.at<uchar>(i, j) > threshold && src.at<uchar>(i, j) <= maxIntensity)
					{
						meanG2 += src.at<uchar>(i, j) * hg[src.at<uchar>(i, j)];
					}
				}	
			}

			meanG1 = meanG1 / N1;
			meanG2 = meanG2 / N2;

			float oldThreshold = threshold;
			threshold = (meanG1 + meanG2) / 2.0f;
			
			if (abs((float)threshold - (float)oldThreshold) < 0.1)
				break;
		}

		printf("Min intensity = %d\n", minIntensity);
		printf("Max intensity = %d\n", maxIntensity);
		printf("Threshold = %f\n", threshold);
		Mat thresholdedImage(height, width, CV_8UC1, RGB(255, 255, 255));
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) <= threshold)
					thresholdedImage.at<uchar>(i, j) = OBJECT_PIXEL;
			}
		}

		imshow("Thresholded Image", thresholdedImage);
	

		// Analytical histogram transformation functions 
		int gOutMax = 255;
		int gOutMin = 0;
		int gInMin = hg[0];
		int gInMax = hg[0];
		for (int i = 0; i < 256; i++)
		{
			if (hg[i] != 0)
			{
				gInMin = i;
				break;
			}
		}

		for (int i = 255; i >= 0; i--)
		{
			if (hg[i] != 0)
			{
				gInMax = i;
				break;
			}
		}

		// Stretch/shrinking
		int gOut[256] = { 0 };
		Mat stretchShrinkImage(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				float aux = (float)gOutMin + (float)(src.at<uchar>(i,j) - gInMin) * ((float)(gOutMax - gOutMin) / (float)(gInMax - gInMin));
				if (aux < 0)
					aux = 0;
				else if (aux > 255)
					aux = 255;		
				stretchShrinkImage.at<uchar>(i, j) = aux;
			}
		}
		imshow("Stretch/Shrink image", stretchShrinkImage);
		
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				gOut[stretchShrinkImage.at<uchar>(i, j)]++;
			}
		}
		showHistogram("Stretched/shrinking", gOut, 256, 256);


		// Gamma correction
		float gamma = 2.0f;
		int gammaHistogram[256] = { 0 };
		Mat gammaImage(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{

				float aux = 255 * ((float)pow((float)src.at<uchar>(i,j) / (float)255, gamma));
				if (aux < 0)
					aux = 0;
				else if (aux > 255)
					aux = 255;
				gammaImage.at<uchar>(i, j) = aux;
			}
		}
		imshow("Gamma image", gammaImage);
		
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				gammaHistogram[gammaImage.at<uchar>(i, j)]++;
			}
		}
		showHistogram("Gamma correction", gammaHistogram, 256, 256);


		// Change brightness
		float offset = 50.0f;
		int brightnessHistogram[256] = { 0 };
		Mat brightnessImage(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				float aux = src.at<uchar>(i,j) + offset;
				if (aux < 0)
					aux = 0;
				else if (aux > 255)
					aux = 255;
				brightnessImage.at<uchar>(i, j) = aux;
			}
		}
		imshow("Brightness image", brightnessImage);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				brightnessHistogram[brightnessImage.at<uchar>(i, j)]++;
			}
		}
		showHistogram("Brightness histogram", brightnessHistogram, 256, 256);


		// Histogram equalization
		Mat histogramEqualizationImage(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				histogramEqualizationImage.at<uchar>(i, j) = (float)255  * cumulativeP[src.at<uchar>(i, j)];
			}
		}

		int histogramEqualization[256] = { 0 };
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				histogramEqualization[histogramEqualizationImage.at<uchar>(i, j)]++;
			}
		}

		imshow("Histogram equalization image", histogramEqualizationImage);
		showHistogram("Histogram Equalization", histogramEqualization, 256, 256);
	}

}

Mat spatialGeneralFilter(Mat sentMatrix, Mat filter)
{
	Mat src = sentMatrix.clone();
	int height = src.rows;
	int width = src.cols;
	
	Mat dst(height, width, CV_8UC1, RGB(255, 255, 255));
	
	double sumOfFilter = 0.0f;
	boolean onlyPositives = true;
	for (int u = 0; u < filter.rows; u++)
	{
		for (int v = 0; v < filter.cols; v++)
		{
			sumOfFilter += filter.at<double>(u, v);
			if (filter.at<double>(u, v) < 0)
				onlyPositives = false;
		}
	}


	double k = (filter.rows - 1) / 2;
	for (int i = k; i < height - k; i++)
	{
		for (int j = k; j < width - k; j++)
		{
			double sum = 0;
			for (int u = 0; u < filter.rows; u++)
			{
				for (int v = 0; v < filter.cols; v++)
				{
					sum += filter.at<double>(u, v) * src.at<uchar>(i + u - k, j + v - k);
				}
			}


			if (onlyPositives)
			{
				sum /= sumOfFilter;
			}
			else
			{
				if (sum > 255)
					sum = 255;
				if (sum < 0)
					sum = 0;
			}
			

			dst.at<uchar>(i, j) = sum;
		}
	}

	return dst;
}


void generalFilter()
{
	char fname[MAX_PATH];
	Mat src;


	while (openFileDlg(fname))
	{

		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Source", src);

		double meanFilter3x3Data[9] = { (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9 };
		double meanFilter5x5Data[25] = { (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9,
										(double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9,
										(double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9, (double)1 / 9 };
		double gaussianFilterData[9] = { (double)1 / 16, (double)2 / 16, (double)1 / 16, (double)2 / 16, (double)4 / 16, (double)2 / 16, (double)1 / 16, (double)2 / 16, (double)1 / 16 };
		
		double highPassFilterData[9] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
		double laplaceFilterData[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1};

		Mat_<double> filter(3, 3, gaussianFilterData);

		Mat dst = spatialGeneralFilter(src, filter);
		imshow("Dest", dst);	
	}
}

void centering_transform(Mat img) 
{  
	//expects floating point image  
	for (int i = 0; i < img.rows; i++)
	{   
		for (int j = 0; j < img.cols; j++)
		{   
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);   
		}  
	} 
}

Mat logMatrix(Mat magnitude)
{
	Mat src = magnitude.clone();
	int height = src.rows;
	int width = src.cols;
	int max = log(1 + src.at<float>(0, 0));

	Mat_<float> dst(height, width);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float value = log(1 + src.at<float>(i, j));
			dst.at<float>(i, j) = value;
			if (value > max)
				max = value;
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst.at<float>(i, j) /= max;
		}
	}

	return dst;
}


Mat generic_frequency_domain_filter(Mat src, int type)
{
	const int LOW_PASS = 0;
	const int HIGH_PASS = 1;
	const int GAUSSIAN_LOW = 2;
	const int GAUSSIAN_HIGH = 3;

	//convert input image to float image
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	//centering transformation
	centering_transform(srcf);

	//perform forward transform with complex image output 
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	//split into real and imaginary channels 
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I)) 

	//calculate magnitude and phase in floating point images mag and phi
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);

	//display the phase and magnitude images here  // ...... 

	Mat logarithmMagnitude = logMatrix(mag);
	imshow("Log Magnitude", logarithmMagnitude);


	int height = channels[0].rows;
	int width = channels[0].cols;
	float R = 30;
	
	if(type == LOW_PASS)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				float result = (height / 2 - i) * (height / 2 - i) + (width / 2 - j) * (width / 2 - j);
				if (result > R * R)
				{
					channels[0].at<float>(i, j) = 0;
					channels[1].at<float>(i, j) = 0;
				}
			}
		}
	}
	else if(type == HIGH_PASS)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				float result = (height / 2 - i) * (height / 2 - i) + (width / 2 - j) * (width / 2 - j);
				if (result <= R * R)
				{
					channels[0].at<float>(i, j) = 0;
					channels[1].at<float>(i, j) = 0;
				}
			}
		}
	}
	else if (type == GAUSSIAN_LOW)
	{
		float A = 20.0f;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				float numerator = (height / 2 - i)*(height / 2 - i) + (width / 2 - j)*(width / 2 - j);
				numerator *= -1;
				float denominator = A * A;
				float eulerExponent = numerator / denominator;
				float realResult = channels[0].at<float>(i,j) * exp(eulerExponent);
				float imaginaryResult = channels[1].at<float>(i, j) * exp(eulerExponent);

				channels[0].at<float>(i, j) = realResult;
				channels[1].at<float>(i, j) = imaginaryResult;
			}
		}
	}
	else if (type == GAUSSIAN_HIGH)
	{
		float A = 20.0f;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				float numerator = (height / 2 - i)*(height / 2 - i) + (width / 2 - j)*(width / 2 - j);
				numerator *= -1;
				float denominator = A * A;
				float eulerExponent = numerator / denominator;
				float realResult = channels[0].at<float>(i, j) * (1 - exp(eulerExponent));
				float imaginaryResult = channels[1].at<float>(i, j) * (1 - exp(eulerExponent));

				channels[0].at<float>(i, j) = realResult;
				channels[1].at<float>(i, j) = imaginaryResult;
			}
		}
	}


	//perform inverse transform and put results in dstf
	Mat dst, dstf; 
	merge(channels, 2, fourier); 
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE); 

	//inverse centering transformation
	centering_transform(dstf); 

	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255,  NORM_MINMAX, CV_8UC1); 

	//Note: normalizing distorts the resut while enhancing the image display in the range [0,255].
	//For exact results (see Practical work 3) the normalization should be replaced with convertion:
	//dstf.convertTo(dst, CV_8UC1); 

	return dst;
}

void frequencyDomain()
{
	char fname[MAX_PATH];
	Mat src;

	const int LOW_PASS = 0;
	const int HIGH_PASS = 1;
	const int GAUSSIAN_LOW = 2;
	const int GAUSSIAN_HIGH = 3;

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Source", src);

		Mat dst = generic_frequency_domain_filter(src, GAUSSIAN_HIGH);
		imshow("Destination", dst);
	}
}

int compare(const void* a, const void* b)
{
	int int_a = *((int*)a);
	int int_b = *((int*)b);

	if (int_a == int_b) return 0;
	else if (int_a < int_b) return -1;
	else return 1;
}


void medianFilter()
{
	char fname[MAX_PATH];
	Mat src;
	int w = -1;
	printf("w = ");
	scanf("%d", &w);
	
	int offset = w / 2;
	int sizeOfFilter = w * w;
	int middleElementPosition = sizeOfFilter / 2;

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount();

		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Source", src);
		int height = src.rows;
		int width = src.cols;
		int neighborsValues[100];

		Mat dst(height, width, CV_8UC1, RGB(255,255,255));
		for (int i = offset; i < height - offset - 1; i++)
		{
			for (int j = offset; j < width - offset - 1; j++)
			{
				int neighborPosition = 0;
				for (int u = i - offset; u <= i + offset; u++)
				{
					for (int v = j - offset; v <= j + offset; v++)
					{
						neighborsValues[neighborPosition] = src.at<uchar>(u, v);
						neighborPosition++;
					}
				}

				qsort(neighborsValues, sizeOfFilter, sizeof(int), compare);

				dst.at<uchar>(i, j) = neighborsValues[middleElementPosition];
			}
		}
		
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("Destination", dst);
	}
}


void twoDGaussianFilter()
{
	char fname[MAX_PATH];
	Mat src;
	int w = -1;
	printf("w = ");
	scanf("%d", &w);

	int offset = w / 2;
	int sizeOfFilter = w * w;
	int middleElementPosition = sizeOfFilter / 2;
	float sigma = (float)w / 6;

	// Create filter
	Mat filter(w, w, CV_32FC1);
	float firstTerm = (float)1 / (2 * PI * pow(sigma, 2));
	float denominator = (float)2 * pow(sigma, 2);
	float sumOfFilterElements = 0.0f;
	for (int x = 0; x < w; x++)
	{
		for (int y = 0; y < w; y++)
		{
			float nominator = pow((x - offset), 2) + pow((y - offset), 2);
			nominator *= (float)-1;
			
			float result = firstTerm;
			result *= exp(nominator / denominator);

			filter.at<float>(x, y) = result;
			sumOfFilterElements += result;
		}
	}

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount();

		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Source", src);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC1, RGB(255, 255, 255));
		for (int i = offset; i < height - offset - 1; i++)
		{
			for (int j = offset; j < width - offset - 1; j++)
			{
				float result = 0;
				for (int x = 0; x < w; x++)
				{
					for (int y = 0; y < w; y++)
					{
						result += src.at<uchar>(i - offset + x, j - offset + y) * filter.at<float>(x, y);
					}
				}
				
				result /= sumOfFilterElements;

				dst.at<uchar>(i, j) = result;
			}
		}

		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("Destination", dst);
	}
}

void oneDGaussianFilters()
{
	char fname[MAX_PATH];
	Mat src;
	int w = -1;
	printf("w = ");
	scanf("%d", &w);

	int offset = w / 2;
	int sizeOfFilter = w * w;
	int middleElementPosition = sizeOfFilter / 2;
	float sigma = (float)w / 6;

	// Create filter
	Mat filter(w, w, CV_32FC1);
	float firstTerm = (float)1 / (sqrt(2 * PI) * pow(sigma, 2));
	float denominator = (float)2 * pow(sigma, 2);
	float xFilterSum = 0.0f;
	float yFilterSum = 0.0f;

	// G(y) filter
	for (int y = 0; y < w; y++)
	{
		float nominator = pow(y - offset, 2);
		nominator *= (float)-1;

		float result = firstTerm;
		result *= exp(nominator / denominator);

		filter.at<float>(offset, y) = result;
		yFilterSum += result;
	}

	// G(x) filter
	for (int x = 0; x < w; x++)
	{
		float nominator = pow(x - offset, 2);
		nominator *= (float)-1;

		float result = firstTerm;
		result *= exp(nominator / denominator);

		filter.at<float>(x, offset) = result;
		xFilterSum += result;
	}

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount();

		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Source", src);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC1, RGB(255, 255, 255));
		Mat aux(height, width, CV_8UC1, RGB(255, 255, 255));


		for (int i = offset; i < height - offset - 1; i++)
		{
			for (int j = offset; j < width - offset - 1; j++)
			{
				float result = 0;

				// G(y)
				for (int y = 0; y < w; y++)
				{
					result += src.at<uchar>(i, j - offset + y) * filter.at<float>(offset, y);
				}

				result /= yFilterSum;

				aux.at<uchar>(i, j) = result;
			}
		}

		for (int i = offset; i < height - offset - 1; i++)
		{
			for (int j = offset; j < width - offset - 1; j++)
			{
				float result = 0;

				// G(x)
				for (int x = 0; x < w; x++)
				{
					result += aux.at<uchar>(i - offset + x, j) * filter.at<float>(x, offset);
				}
				
				result /= xFilterSum;

				dst.at<uchar>(i, j) = result;
			}
		}

		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("Destination", dst);
	}
}

Mat convoluteImageWithFilter(Mat sentMatrix, Mat filter)
{
	Mat src = sentMatrix.clone();
	int height = src.rows;
	int width = src.cols;

	Mat dst(height, width, CV_32FC1, RGB(255, 255, 255));

	int k = filter.rows / 2;
	for (int i = k; i < height - k; i++)
	{
		for (int j = k; j < width - k; j++)
		{
			float sum = 0;
			for (int u = 0; u < filter.rows; u++)
			{
				for (int v = 0; v < filter.cols; v++)
				{
					sum += filter.at<double>(u, v) * src.at<uchar>(i + u - k, j + v - k);
				}
			}

			dst.at<float>(i, j) = sum;
		}
	}

	return dst;
}



void cannyEdgeDetection()
{
	char fname[MAX_PATH];
	Mat src;


	while (openFileDlg(fname))
	{

		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		imshow("Source", src);


		// Gaussian Filter
		int w = -1;
		printf("w = ");
		scanf("%d", &w);

		int offset = w / 2;
		int sizeOfFilter = w * w;
		int middleElementPosition = sizeOfFilter / 2;
		float sigma = (float)w / 6;

		Mat gaussianFilter(w, w, CV_32FC1);
		float firstTerm = (float)1 / (2 * PI * pow(sigma, 2));
		float denominator = (float)2 * pow(sigma, 2);
		float sumOfFilterElements = 0.0f;
		for (int x = 0; x < w; x++)
		{
			for (int y = 0; y < w; y++)
			{
				float nominator = pow((x - offset), 2) + pow((y - offset), 2);
				nominator *= (float)-1;

				float result = firstTerm;
				result *= exp(nominator / denominator);

				gaussianFilter.at<float>(x, y) = result;
				sumOfFilterElements += result;
			}
		}

		Mat gaussianImage(height, width, CV_8UC1, RGB(255, 255, 255));
		for (int i = offset; i < height - offset; i++)
		{
			for (int j = offset; j < width - offset; j++)
			{
				float result = 0;
				for (int x = 0; x < w; x++)
				{
					for (int y = 0; y < w; y++)
					{
						result += src.at<uchar>(i - offset + x, j - offset + y) * gaussianFilter.at<float>(x, y);
					}
				}

				result /= sumOfFilterElements;

				gaussianImage.at<uchar>(i, j) = result;
			}
		}

		imshow("Gaussian filter", gaussianImage);


		double xPrewittFilter[9] = { -1, 0, 1,
								-1, 0, 1,
								-1, 0, 1 };

		double yPrewittFilter[9] = {1, 1, 1,
									0, 0, 0,
									-1, -1, -1 };

		double xSobelFilter[9] = { -1, 0, 1,
									-2, 0, 2,
									-1, 0, 1 };
		
		double ySobelFilter[9] = { 1, 2, 1,
									0, 0, 0,
									-1, -2, -1 };

		double xRobertFilter[4] = { 1, 0,
									0, -1 };

		double yRobertFilter[4] = { 0, -1,
									1, 0 };

		printf("Type of filter:\n");
		printf("0. Prewitt Filter\n");
		printf("1. Sobel Filter\n");
		printf("2. Robert Filter\n");

		int filterType = -1;
		scanf("%d", &filterType);

		Mat xGradient, yGradient;

		if (filterType == 0)
		{
			Mat_<double> xFilter(3, 3, xPrewittFilter);
			Mat_<double> yFilter(3, 3, yPrewittFilter);

			xGradient = convoluteImageWithFilter(gaussianImage, xFilter);
			yGradient = convoluteImageWithFilter(gaussianImage, yFilter);
		}
		else if (filterType == 1)
		{
			Mat_<double> xFilter(3, 3, xSobelFilter);
			Mat_<double> yFilter(3, 3, ySobelFilter);

			xGradient = convoluteImageWithFilter(gaussianImage, xFilter);
			yGradient = convoluteImageWithFilter(gaussianImage, yFilter);
		}
		else 
		{
			Mat_<double> xFilter(2, 2, xRobertFilter);
			Mat_<double> yFilter(2, 2, yRobertFilter);

			xGradient = convoluteImageWithFilter(gaussianImage, xFilter);
			yGradient = convoluteImageWithFilter(gaussianImage, yFilter);
		}

		imshow("X Gradient", xGradient);
		imshow("Y Gradient", yGradient);
	
		Mat gradientMagnitude(height, width, CV_32FC1);
		Mat showGradientMagnitude(height, width, CV_8UC1);
		Mat direction(height, width, CV_32FC1);
		// Magnitude
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				float magResult = (float)sqrt((float) pow(xGradient.at<float>(i, j), 2) + (float) pow(yGradient.at<float>(i, j), 2));
				magResult /= (4 * sqrt(2));
				gradientMagnitude.at<float>(i, j) = magResult;
				showGradientMagnitude.at<uchar>(i, j) = magResult;

				float dirResult = atan2(yGradient.at<float>(i, j), xGradient.at<float>(i, j));
				direction.at<float>(i, j) = dirResult;
			}
		}

		imshow("Magnitude", showGradientMagnitude);
		

		Mat supressedGradientMagnitude(height, width, CV_32FC1);
		Mat showSupressedGradientMagnitude(height, width, CV_8UC1);

		
		for (int i = offset; i < height - offset; i++)
		{
			for (int j = offset; j < width - offset; j++)
			{
				float phiAngle = direction.at<float>(i, j);
				float G = gradientMagnitude.at<float>(i, j);
				float GN1, GN2;

				// 0
				if ((phiAngle >= -CV_PI/8 && phiAngle <= CV_PI / 8) || phiAngle >= 7 * CV_PI / 8 || phiAngle <= -7 * CV_PI / 8)
				{
					GN1 = gradientMagnitude.at<float>(i, j + 1);
					GN2 = gradientMagnitude.at<float>(i, j - 1);
				}
				// 1
				else if((phiAngle >= CV_PI/8 && phiAngle <= 3*CV_PI/8) || (phiAngle >= -7*CV_PI/8 && phiAngle <= -5*CV_PI/8))
				{
					GN1 = gradientMagnitude.at<float>(i - 1, j + 1);
					GN2 = gradientMagnitude.at<float>(i + 1, j - 1);
				}
				// 2
				else if ((phiAngle >= 3*CV_PI/8 && phiAngle <= 5*CV_PI/8) || (phiAngle >= -5*CV_PI/8 && phiAngle <= -3*CV_PI/8))
				{
					GN1 = gradientMagnitude.at<float>(i - 1, j);
					GN2 = gradientMagnitude.at<float>(i + 1, j);
				}
				// 3
				else
				{
					GN1 = gradientMagnitude.at<float>(i - 1, j - 1);
					GN2 = gradientMagnitude.at<float>(i + 1, j + 1);
				}

				if (G >= GN1 && G >= GN2)
					supressedGradientMagnitude.at<float>(i, j) = G;
				else
					supressedGradientMagnitude.at<float>(i, j) = 0;
				showSupressedGradientMagnitude.at<uchar>(i,j) = supressedGradientMagnitude.at<float>(i, j);
			}

			imshow("Non-maxima suppression", showSupressedGradientMagnitude);
		}


		// Adaptive thresholding
		int histogram[256] = { 0 };
		float p = 0.1f;
		for (int i = offset; i < height - offset; i++)
		{
			for (int j = offset; j < width - offset; j++)
			{
				int valueOfGradient = showSupressedGradientMagnitude.at<uchar>(i, j);
				histogram[valueOfGradient]++;
			}
		}

		int numberNonEdgePixels = (1-p) * ((height - 2) * (width - 2) - histogram[0]);

		int histogramSum = 0;
		float thresholdHigh = -1;
		bool thresholdFound = false;
		for (int i = 1; i < 255; i++)
		{
			histogramSum += histogram[i];
			if (thresholdFound == false && histogramSum >= numberNonEdgePixels)
			{
				thresholdHigh = i;
				thresholdFound = true;
			}
		}

		printf("height-2 * width-2 = %d\n", (height - 2) * (width - 2));
		printf("histgoram sum = %d\n", histogramSum);

		float thresholdLow = 0.4f * thresholdHigh;

		printf("numberNonEdgePixels = %d\n", numberNonEdgePixels);
		printf("Threshold low = %f\n", thresholdLow);
		printf("Threshold high = %f\n", thresholdHigh);

		const int NON_EDGE = 0;
		const int WEAK_EDGE = 128;
		const int STRONG_EDGE = 255;


		Mat thresholdedMagnitude(height, width, CV_8UC1);
		for (int i = offset; i < height - offset; i++)
		{
			for (int j = offset; j < width - offset; j++)
			{
				int valueOfCurrentPixel = showSupressedGradientMagnitude.at<uchar>(i, j);
				if (valueOfCurrentPixel < thresholdLow)
					thresholdedMagnitude.at<uchar>(i, j) = NON_EDGE;
				else if (valueOfCurrentPixel >= thresholdLow && valueOfCurrentPixel < thresholdHigh)
					thresholdedMagnitude.at<uchar>(i, j) = WEAK_EDGE;
				else
					thresholdedMagnitude.at<uchar>(i, j) = STRONG_EDGE;
			}
		}

		imshow("Thresholded", thresholdedMagnitude);

		// Edge linking
		int di[8] = { -1, 0, 1, 0, -1, 1, -1, 1 };
		int dj[8] = { 0, -1, 0, 1, 1, -1, -1, 1 };
		uchar neighbors[8];
		for (int i = offset; i < height - offset; i++)
		{
			for (int j = offset; j < width - offset; j++)
			{
				if (thresholdedMagnitude.at<uchar>(i, j) == STRONG_EDGE)
				{
					std::queue<Point2i> Q;

					Q.push({ i,j });
					while (!Q.empty())
					{
						Point2i q = Q.front();
						Q.pop();

						for (int k = 0; k < 8; k++)
						{
							if (q.x + di[k] <= height && q.x + di[k] >= 0 && q.y + dj[k] >= 0 && q.y + dj[k] <= width)
							{
								neighbors[k] = thresholdedMagnitude.at<uchar>(q.x + di[k], q.y + dj[k]);
								//uchar neighborX = q.x + di[k];
								//uchar neighborY = q.y + dj[k];

								if (neighbors[k] == WEAK_EDGE)
								{
									thresholdedMagnitude.at<uchar>(q.x + di[k], q.y + dj[k]) = STRONG_EDGE;
									Q.push({ q.x + di[k], q.y + dj[k] });
								}

							}

						}
					}
				}
			}
		}

		for (int i = offset; i < height - offset; i++)
		{
			for (int j = offset; j < width - offset; j++)
			{
				if (thresholdedMagnitude.at<uchar>(i, j) == WEAK_EDGE)
				{
					thresholdedMagnitude.at<uchar>(i, j) = NON_EDGE;
				}
			}
		}

		imshow("Edge-linked", thresholdedMagnitude);



	}
}

void leastMeanSquares_model1()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int height;
		int width;
		
		// Open file		
		FILE* f = fopen(fname, "r");
		// Number of points
		float nbOfPoints;
		fscanf(f, "%f", &nbOfPoints);

		float currPointX = 0, currPointY = 0;
		float xArray[600], yArray[600];
		float xMin = 1000, yMin = 1000, xMax = 0, yMax = 0;
		for (int i = 0; i < nbOfPoints; i++)
		{
			fscanf(f, "%f %f", &currPointX, &currPointY);
			xArray[i] = currPointX;
			yArray[i] = currPointY;
			if (currPointX < xMin)
				xMin = currPointX;
			if (currPointY < yMin)
				yMin = currPointY;
			if (currPointX > xMax)
				xMax = currPointX;
			if (currPointY > yMax)
				yMax = currPointY;
		}

		yMax -= yMin;
		xMax -= xMin;
		height = yMax + 1;
		width = xMax + 1;

		Mat img(height, width, CV_8UC3, Scalar(255, 255, 255));
		// Scalare
		for (int i = 0; i < nbOfPoints; i++)
		{
			xArray[i] = xArray[i] - xMin;
			yArray[i] = yArray[i] - yMin;

			img.at<Vec3b>(yArray[i], xArray[i]) = Vec3b(0, 0, 0);

			printf("%f %f\n", xArray[i], yArray[i]);
		}

		float nominatorSum1 = 0, nominatorSum2 = 0, nominatorSum3 = 0;
		float denominatorSum1 = 0, denominatorSum2 = 0;
		float theta0Sum1 = 0, theta0Sum2 = 0;
		
		for (int i = 0; i < nbOfPoints; i++)
		{
			nominatorSum1 += xArray[i] * yArray[i];
			nominatorSum2 += xArray[i];
			nominatorSum3 += yArray[i];
			denominatorSum1 += xArray[i] * xArray[i];
			denominatorSum2 += xArray[i];

			theta0Sum1 += yArray[i];
			theta0Sum2 += xArray[i];
		}
		
		float theta1 = (nbOfPoints * nominatorSum1 - nominatorSum2 * nominatorSum3) / (nbOfPoints * denominatorSum1 - (denominatorSum2 * denominatorSum2));
		float theta0 = 1 / nbOfPoints * (theta0Sum1 - theta1 * theta0Sum2);
		printf("Theta 0= %f\n", theta0);
		printf("Theta 1= %f\n", theta1);

		Point aPoint(0, theta0);
		Point bPoint(width, theta1*width + theta0);
		line(img, aPoint, bPoint, Scalar(0, 0, 255), 2);


		imshow("My Window", img);
		waitKey(0);
		fclose(f);
	}
}

void leastMeanSquares_model2()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int height;
		int width;

		// Open file		
		FILE* f = fopen(fname, "r");
		// Number of points
		float nbOfPoints;
		fscanf(f, "%f", &nbOfPoints);

		float currPointX = 0, currPointY = 0;
		float xArray[600], yArray[600];
		float xMin = 1000, yMin = 1000, xMax = 0, yMax = 0;
		for (int i = 0; i < nbOfPoints; i++)
		{
			fscanf(f, "%f %f", &currPointX, &currPointY);
			xArray[i] = currPointX;
			yArray[i] = currPointY;
			if (currPointX < xMin)
				xMin = currPointX;
			if (currPointY < yMin)
				yMin = currPointY;
			if (currPointX > xMax)
				xMax = currPointX;
			if (currPointY > yMax)
				yMax = currPointY;
		}

		yMax -= yMin;
		xMax -= xMin;
		height = yMax + 1;
		width = xMax + 1;

		Mat img(height, width, CV_8UC3, Scalar(255, 255, 255));
		// Scalare
		for (int i = 0; i < nbOfPoints; i++)
		{
			xArray[i] = xArray[i] - xMin;
			yArray[i] = yArray[i] - yMin;

			img.at<Vec3b>(yArray[i], xArray[i]) = Vec3b(0, 0, 0);

			printf("%f %f\n", xArray[i], yArray[i]);
		}

		float xySum = 0, xSum = 0, ySum = 0, ySquaredMinusXSquaredSum = 0;
		for (int i = 0; i < nbOfPoints; i++)
		{
			xySum += xArray[i] * yArray[i];
			xSum += xArray[i];
			ySum += yArray[i];
			ySquaredMinusXSquaredSum += yArray[i] * yArray[i] - xArray[i] * xArray[i];
		}

		float beta = (-1 / 2.0) * atan2(2 * xySum - (2 / nbOfPoints) * xSum * ySum, ySquaredMinusXSquaredSum + ((1 / nbOfPoints) * (xSum * xSum)) - ((1 / nbOfPoints) * (ySum * ySum)));
		float ro = (1 / nbOfPoints) * (cos(beta) * xSum + sin(beta) * ySum);

		printf("beta = %f\n", beta);
		printf("ro = %f\n", ro);

		if (abs(beta) > 0.5)
		{
			Point aPoint(0, ro / sin(beta));
			Point bPoint(width, ((ro - width * cos(beta)) / sin(beta)));
			line(img, aPoint, bPoint, Scalar(0, 255, 0), 3);
		}
		else
		{
			Point aPoint(ro/cos(beta), 0);
			Point bPoint(((ro - height * sin(beta)) / cos(beta)), height);
			line(img, aPoint, bPoint, Scalar(0, 255, 0), 3);
		}

		imshow("My Window", img);
		waitKey(0);
		fclose(f);
	}
}

void ransac()
{
	char fname[MAX_PATH];
	Mat src;
	srand(time(NULL));
	while (openFileDlg(fname))
	{
		int xCoordinates[1000];
		int yCoordinates[1000];
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0;
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					xCoordinates[k] = j;
					yCoordinates[k] = i;
					k++;
				}
			}
		}

		float t = 10;
		float q = 0.3;
		float p = 0.99;
		float s = 2;
		float N = log(1 - p) / log(1 - pow(q, s));
		int iterations = 0;
		int selectedPoint1X, selectedPoint2X, selectedPoint1Y, selectedPoint2Y;
		int aParamFinal, bParamFinal, cParamFinal;
		int maxNbOfInliers = 0;
		
		while (1)
		{	
			int index1, index2;
			while (1)
			{
				index1 = rand() % k;
				index2 = rand() % k;
				if (index1 != index2)
					break;
			}

			selectedPoint1X = xCoordinates[index1];
			selectedPoint1Y = yCoordinates[index1];
			selectedPoint2X = xCoordinates[index2];
			selectedPoint2Y = yCoordinates[index2];

			int aParam = selectedPoint1Y - selectedPoint2Y;
			int bParam = selectedPoint2X - selectedPoint1X;
			int cParam = selectedPoint1X * selectedPoint2Y - selectedPoint2X * selectedPoint1Y;

			int nbOfInliers = 0;
			for (int i = 0; i < k; i++)
			{
					float nominator = abs(aParam * xCoordinates[i] + bParam * yCoordinates[i] + cParam);
					float denominator = sqrt(pow(aParam, 2) + pow(bParam, 2));
					float distance = nominator / denominator;
					if (distance <= t)
						nbOfInliers++;
			}


			if (nbOfInliers > maxNbOfInliers)
			{
				aParamFinal = aParam;
				bParamFinal = bParam;
				cParamFinal = cParam;
				maxNbOfInliers = nbOfInliers;
			}

			iterations++;

			if (iterations > N || maxNbOfInliers > q * k)
				break;
		}

		line(src, Point(selectedPoint1X, selectedPoint1Y), Point(selectedPoint2X, selectedPoint2Y), Scalar(0, 0, 255));
		imshow("input", src);
	}
}

void houghTransform()
{
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{

		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		int D = sqrt(pow(height, 2) + pow(width, 2)); // Diagonal of image

		const int maxNumberOfPoints = 1000;
		int xCoordinates[maxNumberOfPoints], yCoordinates[maxNumberOfPoints];
		int nbOfPoints = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 255)
				{
					xCoordinates[nbOfPoints] = j;
					yCoordinates[nbOfPoints] = i;
					nbOfPoints++;
				}
			}
		}

		Mat houghSpace(D + 1, 360, CV_32SC1); //matrix with int values
		houghSpace.setTo(0);

		int deltaTheta = 1;
		int deltaRo = 1;

		// Voting
		for (int i = 0; i < nbOfPoints; i++)
		{
			int ro;
			for (int theta = 0; theta < 360; theta++)
			{
				ro = xCoordinates[i] * cos(theta * CV_PI / 180) + yCoordinates[i] * sin(theta * CV_PI / 180);
	
				if (ro > 0 && ro < D)
				{
					houghSpace.at<int>(ro, theta)++;
				}
			}
		}

		int maxHough = houghSpace.at<int>(0, 0);
		for (int i = 1; i < D + 1; i++)
		{
			for (int j = 0; j < 360; j++)
			{
				if (houghSpace.at<int>(i, j) > maxHough)
					maxHough = houghSpace.at<int>(i, j);
			}
		}

		Mat houghImg;
		houghSpace.convertTo(houghImg, CV_8UC1, 255.0f / maxHough);
		imshow("Voting", houghImg);

		
		struct peak {
			int theta, ro, hval;
			bool operator < (const peak& o) const {
				return hval > o.hval;
			}
		};

		std::vector<peak> peaks;
		int w = 5;
		int middlePositionOfStructuringElement = w / 2;
		int houghSpaceRows = houghSpace.rows;
		int houghSpaceCols = houghSpace.cols;

		for (int i = 0 + middlePositionOfStructuringElement; i < houghSpaceRows - middlePositionOfStructuringElement; i++)
		{
			for (int j = 0 + middlePositionOfStructuringElement; j < houghSpaceCols - middlePositionOfStructuringElement; j++)
			{
				bool isMaximalInVicinity = true;
				int currentElem = houghSpace.at<int>(i, j);
				
				for (int a = i - middlePositionOfStructuringElement; a <= i + middlePositionOfStructuringElement; a++)
				{
					for (int b = j - middlePositionOfStructuringElement; b <= j + middlePositionOfStructuringElement; b++)
					{
						if (houghSpace.at<int>(a, b) > currentElem )
						{
							isMaximalInVicinity = false;
						}
					}
				}

				if(isMaximalInVicinity)
				{
					peak newPeak;
					newPeak.theta = j;
					newPeak.ro = i;
					newPeak.hval = currentElem;
					peaks.push_back(newPeak);
				}
			}
		}

		std::sort(peaks.begin(), peaks.end());

		Mat finalImage;
		cvtColor(src, finalImage, CV_GRAY2RGB);
		
		int nbOfLines = 10;
		for (int i = 0; i < nbOfLines; i++)
		{
			Point p1(0, (int)((double)peaks.at(i).ro / sin(peaks.at(i).theta * CV_PI / 180)));
			Point p2(finalImage.cols, (int)(((double)peaks.at(i).ro - finalImage.cols * cos(peaks.at(i).theta * CV_PI / 180)) / (sin(peaks.at(i).theta * CV_PI / 180))));
			line(finalImage, p1, p2, Scalar(0, 0, 255), 2);

			printf("Peak #%d: Theta: %d	Ro:%d	Hval: %d \n", i + 1, peaks.at(i).theta, peaks.at(i).ro, peaks.at(i).hval);
		}


		imshow("Result", finalImage);
	}
}

void distanceTransformPatternMatching()
{
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
	
		int wD = 3;
		int wH = 2;
		int wV = 2;

		int di[8] = { -1,-1,-1,0,0,1,1,1 };
		int dj[8] = { -1,0,1,-1,1,-1,0,1 };
		int weight[8] = { wD,wV,wD,
						  wH,   wH,
						  wD,wV,wD
		};

		Mat distanceTransform = src.clone();

		// Top Left to Bottom Right
		for (int i = 1; i < height; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				int min = distanceTransform.at<uchar>(i, j);
				for (int k = 0; k < 4; k++)
				{
					int neighbour = distanceTransform.at<uchar>(i + di[k], j + dj[k]) + weight[k];
					if (neighbour < min)
						min = neighbour;
				}

				distanceTransform.at<uchar>(i, j) = min;
			}
		}

		// Bottom Right to Top Left
		for (int i = height - 2; i >= 0; i--)
		{
			for (int j = width - 2; j >= 1; j--)
			{	
				int min = distanceTransform.at<uchar>(i, j);
				for (int k = 4; k < 8; k++)
				{
					int neighbour = distanceTransform.at<uchar>(i + di[k], j + dj[k]) + weight[k];
					if (neighbour < min)
						min = neighbour;
				}

				distanceTransform.at<uchar>(i, j) = min;
			}
		}


		imshow("Distance Transform", distanceTransform);
		imshow("Pattern", src);

		char objectFname[MAX_PATH];
		openFileDlg(objectFname);
		Mat object = imread(objectFname, IMREAD_GRAYSCALE);
		imshow("Object", object);

		int objectHeight = object.rows;
		int objectWidth = object.cols;
		int nbOfObjectPixels = 0;
		int distanceSum = 0;
		for (int i = 0; i < objectHeight; i++)
		{
			for (int j = 0; j < objectWidth; j++)
			{
				if (object.at<uchar>(i, j) == 0)
				{
					nbOfObjectPixels++;
					distanceSum += distanceTransform.at<uchar>(i, j);
				}
			}
		}

		float patternMatchingScore = (float)distanceSum / nbOfObjectPixels;
	
		printf("Pattern matching score: %f\n", patternMatchingScore);
	}

}

void statisticalDataAnalysis()
{
	char fname[MAX_PATH];
	char folder[256] = "Images/images_lab5";
	Mat featureMatrix(400, 361, CV_8UC1);
	int inputSize = 400;
	int d = 361;
	FILE *fp;
	fp = fopen("covarianceMatrix.csv", "w+");
	FILE *fpCorrelation;
	fpCorrelation = fopen("correlationMatrix.csv", "w+");

	

	int k = 0;
	int l = 0;

	for (int inputCount = 1; inputCount <= inputSize; inputCount++) 
	{
		sprintf(fname, "%s/face%05d.bmp", folder, inputCount);
		Mat img = imread(fname, 0);
		int height = img.rows;
		int width = img.cols;


		l = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				featureMatrix.at<uchar>(k, l) = img.at<uchar>(i, j);
				l++;
			}
		}

		k++;
	}

	imshow("Feature Matrix", featureMatrix);


	Mat covarianceMatrix(d, d, CV_32F, Scalar(0));
	Mat correlationMatrix(d, d, CV_32F, Scalar(0));

	for (int i = 0; i < d; i++)
	{
		for (int j = 0; j < d; j++)
		{
			float meanI = 0;
			float meanJ = 0;

			float standardDeviationI = 0;
			float standardDeviationJ = 0;

			for (int k = 0; k < inputSize; k++)
			{
				meanI += featureMatrix.at<uchar>(k, i);
				meanJ += featureMatrix.at<uchar>(k, j);
			}
			
			meanI *= (float)1 / inputSize;
			meanJ *= (float)1 / inputSize;

			int sum = 0;
			for(int k = 0; k < inputSize; k++)
			{
				int sumTerm1 = featureMatrix.at<uchar>(k, i) - meanI;
				int sumTerm2 = featureMatrix.at<uchar>(k, j) - meanJ;
				sum += sumTerm1 * sumTerm2;

				standardDeviationI += pow(sumTerm1, 2);
				standardDeviationJ += pow(sumTerm2, 2);
			}
			 
			covarianceMatrix.at<float>(i, j) = ((float)1 / inputSize) * sum;
			fprintf(fp, "%f,", covarianceMatrix.at<float>(i, j));

			standardDeviationI = sqrt(((float)1 / inputSize) * standardDeviationI);
			standardDeviationJ = sqrt(((float)1 / inputSize) * standardDeviationJ);

			correlationMatrix.at<float>(i, j) = covarianceMatrix.at<float>(i, j) / (standardDeviationI * standardDeviationJ);
			fprintf(fpCorrelation, "%f,", correlationMatrix.at<float>(i, j));
		}
		fprintf(fp, "\n");
		fprintf(fpCorrelation, "\n");
	}

	fclose(fp);
	fclose(fpCorrelation);


	Mat eyesCorrelation(256, 256, CV_8UC1, Scalar(255));
	int leftEyeX = 5;
	int leftEyeY = 4;
	int rightEyeX = 5;
	int rightEyeY = 14;
	for (int k = 0; k < inputSize; k++)
	{
		int xFeature = featureMatrix.at<uchar>(k, leftEyeX * 19 + leftEyeY);
		int yFeature = featureMatrix.at<uchar>(k, rightEyeX * 19 + rightEyeY);

		eyesCorrelation.at<uchar>(xFeature, yFeature) = 0;
	}

	printf("Eyes correlation coefficient: %f\n", correlationMatrix.at<float>(leftEyeX * 19 + leftEyeY, rightEyeX * 19 + rightEyeY));


	Mat cheeksCorrelation(256, 256, CV_8UC1, Scalar(255));
	int leftCheekX = 10;
	int leftCheekY = 3;
	int rightCheekX = 9;
	int rightCheekY = 15;
	for (int k = 0; k < inputSize; k++)
	{
		int xFeature = featureMatrix.at<uchar>(k, leftCheekX * 19 + leftCheekY);
		int yFeature = featureMatrix.at<uchar>(k, rightCheekX * 19 + rightCheekY);

		cheeksCorrelation.at<uchar>(xFeature, yFeature) = 0;
	}

	printf("Cheeks correlation coefficient: %f\n", correlationMatrix.at<float>(leftCheekX * 19 + leftCheekY, rightCheekX * 19 + rightCheekY));

	
	Mat leftEyeBottomLeftCornerCorrelation(256, 256, CV_8UC1, Scalar(255));
	int bottomLeftCornerX = 18;
	int bottomLeftCornerY = 0;
	for (int k = 0; k < inputSize; k++)
	{
		int xFeature = featureMatrix.at<uchar>(k, leftEyeX * 19 + leftEyeY);
		int yFeature = featureMatrix.at<uchar>(k, bottomLeftCornerX * 19 + bottomLeftCornerY);

		leftEyeBottomLeftCornerCorrelation.at<uchar>(xFeature, yFeature) = 0;
	}
	printf("Left Eye Left Bottom Corner correlation coefficient: %f\n", correlationMatrix.at<float>(leftEyeX * 19 + leftEyeY, bottomLeftCornerX * 19 + bottomLeftCornerY));

	imshow("Eyes Correlation", eyesCorrelation);
	imshow("Cheeks Correlation", cheeksCorrelation);
	imshow("Left Eye-Bottom Left Corner Correlation", leftEyeBottomLeftCornerCorrelation);
	waitKey();
}

void principalComponentAnalysis()
{


	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int height;
		int width;

		// Open file		
		FILE* f = fopen(fname, "r");
		
		double n;
		double d;
		fscanf(f, "%lf %lf", &n, &d);

		//int n = 1000;
		//int d = 7;

		Mat feat(n, d, CV_64FC1); //feat.at<double>(i,j)

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < d; j++)
			{
				double currValue;
				fscanf(f, "%lf ", &currValue);
				feat.at<double>(i, j) = currValue;
			}
		}

		double mean[10] = { 0 };
		for (int j = 0; j < feat.cols; j++)
		{
			double sum = 0;
			for (int i = 0; i < feat.rows; i++)
			{
				sum += feat.at<double>(i, j);
			}

			sum /= n;
			mean[j] = sum;
		}
		

		Mat X(n, d, CV_64FC1);
		for (int i = 0; i < X.rows; i++)
		{
			for (int j = 0; j < X.cols; j++)
			{
				X.at<double>(i, j) = feat.at<double>(i, j) - mean[j];
			}
		}

		// Covariance matrix after means are subtracted
		Mat C = X.t()*X / (n - 1);

		// Eigenvalue decomposition
		Mat Lambda, Q;
		eigen(C, Lambda, Q);
		Q = Q.t();

		printf("Lambda values: ");
		for (int i = 0; i < Lambda.rows; i++)
		{
			printf("%lf ", Lambda.at<double>(i, 0));
		}

		// k = numberOfDimensionsTaken
		const int k = 2;


		Mat qOneK(d, k, CV_64FC1);
		for (int j = 0; j < qOneK.cols; j++)
		{
			for (int i = 0; i < qOneK.rows; i++)
			{
				qOneK.at<double>(i, j) = Q.at<double>(i, j);
			}
		}

		Mat XCoeff = X * qOneK;
		double max[k] = { 0, 0 };
		double min[k] = { 1000000, 1000000 };

		for (int i = 0; i < XCoeff.rows; i++)
		{
			for (int j = 0; j < XCoeff.cols; j++)
			{
				if (XCoeff.at<double>(i, j) > max[j])
				{
					max[j] = XCoeff.at<double>(i, j);
				}
				if (XCoeff.at<double>(i, j) < min[j])
				{
					min[j] = XCoeff.at<double>(i, j);
				}
			}
		}

		double XCoeffShowRows = max[1] - min[1] + 1;
		double XCoeffShowCols = max[0] - min[0] + 1;
		
		Mat XCoeffShow(XCoeffShowRows, XCoeffShowCols, CV_8UC1, Scalar(255));
		for (int i = 0; i < XCoeff.rows; i++)
		{
			XCoeffShow.at<uchar>(XCoeff.at<double>(i,1) - min[1], XCoeff.at<double>(i, 0) - min[0]) = 0;	
		}

		imshow("Representation of the two features", XCoeffShow);


		Mat XTildaReconstruction = XCoeff * qOneK.t();
		double reconstructionSum = 0;
		for (int i = 0; i < X.rows; i++)
		{
			for (int j = 0; j < X.cols; j++)
			{
				reconstructionSum += (abs(XTildaReconstruction.at<double>(i, j) - X.at<double>(i, j)));
			}
		}


		double mad = reconstructionSum / (n * d);
		printf("\nMAD = %lf\n", mad);
	}
}

void kMeansClustering()
{
	char fname[MAX_PATH];
	Mat src;
	srand(time(NULL));

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		int numberOfFeatures = 2; // a.k.a. d

		const int numberOfClusters = 5; // a.k.a. K
							   
		int numberOfPoints = 0;
		const int NUMBER_OF_POINTS_OCHIOMETRIC = 10000;
		int iPoints[NUMBER_OF_POINTS_OCHIOMETRIC];
		int jPoints[NUMBER_OF_POINTS_OCHIOMETRIC];
		int clusterToPoint[NUMBER_OF_POINTS_OCHIOMETRIC];

		// memorize points into array(s)
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					iPoints[numberOfPoints] = i;
					jPoints[numberOfPoints] = j;
					numberOfPoints++;
				}
			}
		}
		
		int iClusterCenter[numberOfClusters];
		int	jClusterCenter[numberOfClusters];

		for (int i = 0; i < numberOfClusters; i++)
		{
			int randNumber = rand() % numberOfPoints;
			iClusterCenter[i] = iPoints[randNumber];
			jClusterCenter[i] = jPoints[randNumber];
		}

		while(1)
		{

			for (int i = 0; i < NUMBER_OF_POINTS_OCHIOMETRIC; i++)
			{
				clusterToPoint[i] = 0;
			}

			int iClusterCenterSum[numberOfClusters];
			int	jClusterCenterSum[numberOfClusters];
			int nbOfPointsInCluster[numberOfClusters];
			for (int i = 0; i < numberOfClusters; i++)
			{
				iClusterCenterSum[i] = 0;
				jClusterCenterSum[i] = 0;
				nbOfPointsInCluster[i] = 0;
			}
		
			// assign each point to cluster
			for (int i = 0; i < numberOfPoints; i++)
			{
				int distancesToClusters[numberOfClusters];
				for (int j = 0; j < numberOfClusters; j++)
				{
					int feature1Sum = pow(iPoints[i] - iClusterCenter[j], 2);
					int feature2Sum = pow(jPoints[i] - jClusterCenter[j], 2);
					int distance = sqrt(feature1Sum + feature2Sum);
					distancesToClusters[j] = distance;
				}

				int minDistanceToCluster = distancesToClusters[0];
				clusterToPoint[i] = 0;

				for (int j = 1; j < numberOfClusters; j++)
				{
					if (distancesToClusters[j] < minDistanceToCluster)
					{
						minDistanceToCluster = distancesToClusters[j];
						clusterToPoint[i] = j;
					}
				}

				int chosenCluster = clusterToPoint[i];
				iClusterCenterSum[chosenCluster] += iPoints[i];
				jClusterCenterSum[chosenCluster] += jPoints[i];
				nbOfPointsInCluster[chosenCluster]++;
			}

			

			int newIClusterCenter[numberOfClusters];
			int newJClusterCenter[numberOfClusters];
			for (int i = 0; i < numberOfClusters; i++)
			{
				newIClusterCenter[i] = iClusterCenterSum[i] / nbOfPointsInCluster[i];
				newJClusterCenter[i] = jClusterCenterSum[i] / nbOfPointsInCluster[i];
			}
		
			boolean changeDetected = false;
			for (int i = 0; i < numberOfClusters; i++)
			{
				if (iClusterCenter[i] != newIClusterCenter[i] || jClusterCenter[i] != newJClusterCenter[i])
				{
					changeDetected = true;
				}

				iClusterCenter[i] = newIClusterCenter[i];
				jClusterCenter[i] = newJClusterCenter[i];
			}

			if (changeDetected == false)
			{
				break;
			}
				
		}

		Mat colorImage(src.rows, src.cols, CV_8UC3, Scalar(255,255,255));
		
		Vec3b colors[numberOfClusters];
		for (int i = 0; i < numberOfClusters; i++)
		{
			unsigned char randomRed = rand() % 255;
			unsigned char randomBlue = rand() % 255;
			unsigned char randomGreen = rand() % 255;
			colors[i] = { randomBlue, randomGreen, randomRed };
		}
			
		int currentPointIndex = 0;
		for (int i = 0; i < colorImage.rows; i++)
		{
			for (int j = 0; j < colorImage.cols; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					colorImage.at<Vec3b>(i, j) = colors[clusterToPoint[currentPointIndex]];
					currentPointIndex++;
				}
			}
		}

		imshow("Colored clusters", colorImage);

		Mat voronoiTesselation(src.rows, src.cols, CV_8UC3);
		for (int i = 0; i < voronoiTesselation.rows; i++)
		{
			for (int j = 0; j < voronoiTesselation.cols; j++)
			{
				int distancesToClusters[numberOfClusters];
				for (int k = 0; k < numberOfClusters; k++)
				{
					int feature1Sum = pow(i - iClusterCenter[k], 2);
					int feature2Sum = pow(j - jClusterCenter[k], 2);
					int distance = sqrt(feature1Sum + feature2Sum);
					distancesToClusters[k] = distance;
				}

				int minDistanceToCluster = distancesToClusters[0];
				int chosenCluster = 0;
				for (int k = 1; k < numberOfClusters; k++)
				{
					if (distancesToClusters[k] < minDistanceToCluster)
					{
						minDistanceToCluster = distancesToClusters[k];
						chosenCluster = k;
					}
				}

				voronoiTesselation.at<Vec3b>(i, j) = colors[chosenCluster];
			}
		}
		imshow("Voronoi Tesselation", voronoiTesselation);
	}
}

void kNearestNeighboursClassifier()
{

	const int nrclasses = 6;
	char classes[nrclasses][10] =
	{ "beach", "city", "desert", "forest", "landscape", "snow" };
	
	const int numberOfBins = 8;
	const int binSize = 256 / numberOfBins;

	const int nrinst = 672;
	int feature_dim = 3 * numberOfBins; // d

	Mat X(0, feature_dim, CV_32FC1);
	Mat y(0, 1, CV_8UC1);

	char fname[256];
	int fileNr;
	// Train
	for (int k = 0; k < nrclasses; k++)
	{
		fileNr = 0;
		while (1) {
			sprintf(fname, "Images/images_KNN/train/%s/%06d.jpeg", classes[k], fileNr++);
			Mat img = imread(fname, IMREAD_COLOR);
			if (img.cols == 0) break;
			//calculate the histogram in hist
			Mat fv(1, feature_dim, CV_32FC1);
			int histRed[numberOfBins] = { 0 };
			int histGreen[numberOfBins] = { 0 };
			int histBlue[numberOfBins] = { 0 };
			
			for (int i = 0; i < img.rows; i++)
			{
				for (int j = 0; j < img.cols; j++)
				{
					Vec3b currentPixel = img.at<Vec3b>(i, j);
					int blue = currentPixel[0];
					int green = currentPixel[1];
					int red = currentPixel[2];

					histRed[red / binSize]++;
					histGreen[green / binSize]++;
					histBlue[blue / binSize]++;
				}
			}

			int counter = 0;

			for (int j = 0; j < numberOfBins; j++)
			{
				fv.at<float>(0, counter) = histRed[j];
				counter++;
			}

			for (int j = 0; j < numberOfBins; j++)
			{
				fv.at<float>(0, counter) = histGreen[j];
				counter++;
			}

			for (int j = 0; j < numberOfBins; j++)
			{
				fv.at<float>(0, counter) = histBlue[j];
				counter++;
			}

			X.push_back(fv);
			Mat classForCurrent(1, 1, CV_8UC1);
			classForCurrent.at<uchar>(0, 0) = k;
			y.push_back(classForCurrent);
	
		}
	}


	struct element {
	double distance, classNumber;
	bool operator < (const element& o) const {
		return distance < o.distance;
		}
	};

	std::vector<element> elements;
	std::sort(elements.begin(), elements.end());

	Mat confusionMatrix(nrclasses, nrclasses, CV_32FC1, Scalar(0));

	// Test
	for (int k = 0; k < nrclasses; k++)
	{
		fileNr = 0;
		while (1) {
			sprintf(fname, "Images/images_KNN/test/%s/%06d.jpeg", classes[k], fileNr++);
			Mat img = imread(fname, IMREAD_COLOR);
			if (img.cols == 0) break;
			//calculate the histogram in hist
			Mat fv(1, feature_dim, CV_32FC1);
			int histRed[numberOfBins] = { 0 };
			int histGreen[numberOfBins] = { 0 };
			int histBlue[numberOfBins] = { 0 };

			for (int i = 0; i < img.rows; i++)
			{
				for (int j = 0; j < img.cols; j++)
				{
					Vec3b currentPixel = img.at<Vec3b>(i, j);
					int blue = currentPixel[0];
					int green = currentPixel[1];
					int red = currentPixel[2];

					histRed[red / binSize]++;
					histGreen[green / binSize]++;
					histBlue[blue / binSize]++;
				}
			}

			int counter = 0;

			for (int j = 0; j < numberOfBins; j++)
			{
				fv.at<float>(0, counter) = histRed[j];
				counter++;
			}

			for (int j = 0; j < numberOfBins; j++)
			{
				fv.at<float>(0, counter) = histGreen[j];
				counter++;
			}

			for (int j = 0; j < numberOfBins; j++)
			{
				fv.at<float>(0, counter) = histBlue[j];
				counter++;
			}

			std::vector<element> d;


			for (int j = 0; j < nrinst; j++)
			{
				double distance = 0;
				for (int count = 0; count < feature_dim; count++)
				{
					distance += pow(fv.at<float>(0, count) - X.at<float>(j, count), 2);
				}

				element newElement;
				newElement.distance = distance;
				newElement.classNumber = y.at<uchar>(j, 0);
				
				d.push_back(newElement);
			}

			sort(d.begin(), d.end());


			const int K = 5; // CLOSEST K NEIGHBORUS!!!
			int testHist[nrclasses] = { 0 };
			for (int j = 0; j < K; j++)
			{
				int classOfJNeighbour = d[j].classNumber;
				testHist[classOfJNeighbour]++;
			}

			int selectedClass = 0;
			int max = testHist[0];
			for (int j = 1; j < nrclasses; j++)
			{
				if (testHist[j] > max)
				{
					max = testHist[j];
					selectedClass = j;
				}
			}
			
			confusionMatrix.at<float>(k, selectedClass)++;

		}
	}

	int correctPredictions = 0;
	int wrongPredictions = 0;
	for (int i = 0; i < nrclasses; i++)
	{
		for (int j = 0; j < nrclasses; j++)
		{
			if (i == j)
			{
				correctPredictions += confusionMatrix.at<float>(i,j);
			}
			else
			{
				wrongPredictions += confusionMatrix.at<float>(i, j);
			}
		}
	}
	
	float accuracy = (float)correctPredictions / (correctPredictions + wrongPredictions);
	
	printf("Confusion matrix: \n");
	for (int i = 0; i < nrclasses; i++)
	{
		for (int j = 0; j < nrclasses; j++)
		{
			printf("%d ", (int) confusionMatrix.at<float>(i, j));
		}
		printf("\n");
	}
	
	printf("\nAccuracy: %f\n", accuracy);
	

	getchar();
	getchar();
	
}

void naiveBayesianClassifier()
{
	srand(time(NULL));

	const int NUMBER_OF_CLASSES = 10;
	const int NUMBER_OF_IMAGES_IN_EACH_CLASS = 100;
	
	const int IMAGE_ROW_SIZE = 28;
	const int IMAGE_COLUMN_SIZE = 28;
	
	const int XRows = NUMBER_OF_CLASSES * NUMBER_OF_IMAGES_IN_EACH_CLASS;
	const int XColumns = IMAGE_ROW_SIZE * IMAGE_COLUMN_SIZE; // 28 x 28each image is 28x28

	const int BINARIZATION_THRESHOLD = 127;
	
	Mat X(0, XColumns, CV_8UC1);
	Mat y(0, 1, CV_8UC1);
	Mat priors(NUMBER_OF_CLASSES, 1, CV_64FC1);
	Mat likelihood(NUMBER_OF_CLASSES, XColumns, CV_64FC1);


	char fname[256];	
	int totalImages = 0;
	int imagesInClass[NUMBER_OF_CLASSES] = { 0 };
	
	// Train
	for (int currentClass = 0; currentClass < NUMBER_OF_CLASSES; currentClass++)
	{
		int currentImageNumber = 0;
		// TODO: while(1) when reading all images in each class
		while (currentImageNumber < NUMBER_OF_IMAGES_IN_EACH_CLASS) { 
			sprintf(fname, "Images/images_Bayes/train/%d/%06d.png", currentClass, currentImageNumber);
			Mat img = imread(fname, IMREAD_GRAYSCALE);
			if (img.cols == 0) break;
			//process img

			// Binarization 
			for (int i = 0; i < img.rows; i++)
			{
				for (int j = 0; j < img.cols; j++)
				{
					if (img.at<uchar>(i, j) > BINARIZATION_THRESHOLD)
					{
						img.at<uchar>(i, j) = 255;
					}
					else
					{
						img.at<uchar>(i, j) = 0;
					}
				}
			}

			//
			Mat fv(1, XColumns, CV_8UC1);
			int fvCounter = 0;
			for (int i = 0; i < img.rows; i++)
			{
				for (int j = 0; j < img.cols; j++)
				{
					fv.at<uchar>(0, fvCounter) = img.at<uchar>(i,j);
					fvCounter++;
				}
			}

			X.push_back(fv);
			Mat yRowForCurrentImage(1, 1, CV_8UC1);
			yRowForCurrentImage.at<uchar>(0, 0) = currentClass;
			y.push_back(yRowForCurrentImage);

			currentImageNumber++;
			totalImages++;
		}

		imagesInClass[currentClass] = currentImageNumber;
	}
	

	// Priors
	for (int currentClass = 0; currentClass < NUMBER_OF_CLASSES; currentClass++)
	{
		double priorForCurrentClass = ((double) imagesInClass[currentClass]) / totalImages;
		priors.at<double>(currentClass, 0) = priorForCurrentClass;
	}


	// Likelihood
	for (int i = 0; i < likelihood.rows; i++)
	{
		for (int j = 0; j < likelihood.cols; j++)
		{
			int currentClass = i;
			int currentFeature = j;
			
			int countFeature = 0;

			for (int k = 0; k < XRows; k++)
			{
				if (y.at<uchar>(k, 0) == currentClass)
				{
					if (X.at<uchar>(k, j) == 255)
					{
						countFeature++;
					}
				}
			}

			likelihood.at<double>(i, j) = ((double)countFeature + 1) / (imagesInClass[currentClass] + NUMBER_OF_CLASSES);
		}
	}

	printf("\nTraining complete.");
	for (int i = 0; i < priors.rows; i++)
	{
		printf("\nPriori probability for digit %d = %lf", i, priors.at<double>(i, 0));
	}
	
	
	

	// Testing
	char fname2[MAX_PATH];
	Mat testImage;
	while (openFileDlg(fname2))
	{
		testImage = imread(fname2, IMREAD_GRAYSCALE);
		
		for (int i = 0; i < testImage.rows; i++)
		{
			for (int j = 0; j < testImage.cols; j++)
			{
				if (testImage.at<uchar>(i, j) > BINARIZATION_THRESHOLD)
				{
					testImage.at<uchar>(i, j) = 255;
				}
				else
				{
					testImage.at<uchar>(i, j) = 0;
				}
			}
		}

		double classification[NUMBER_OF_CLASSES] = { 0 };

		for (int i = 0; i < NUMBER_OF_CLASSES; i++)
		{
			double logOfCurrentClass = log(priors.at<double>(i, 0));
			double sum = 0;

			for (int j = 0; j < IMAGE_ROW_SIZE * IMAGE_COLUMN_SIZE; j++)
			{

				if (testImage.at<uchar>(j / IMAGE_COLUMN_SIZE, j % IMAGE_COLUMN_SIZE) == 255)
				{
					// j = row * column_size + column_size
					sum += log(likelihood.at<double>(i, j));
				}
				else
				{
					sum += log(1 - likelihood.at<double>(i, j));
				}
			}

			classification[i] = logOfCurrentClass + sum;
		}

		double maxProbability = classification[0];
		int selectedClass = 0;
		printf("\nProbability of 0: %lf\n", classification[0]);
		for (int i = 1; i < NUMBER_OF_CLASSES; i++)
		{
			printf("Probability of %d: %lf\n", i, classification[i]);
			if (classification[i] > maxProbability)
			{
				maxProbability = classification[i];
				selectedClass = i;
			}
		}

		printf("\n Selected class: %d\n", selectedClass);

	}

}

void linearClassifier()
{
	char fname[MAX_PATH];
	Mat img;

	while (openFileDlg(fname))
	{
		img = imread(fname, IMREAD_COLOR);
		Mat X(0, 2, CV_64FC1);
		Mat y(0, 1, CV_64FC1);

		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				Vec3b currentPixel = img.at<Vec3b>(i, j);

				if (currentPixel[0] == 255 && currentPixel[1] == 0 && currentPixel[2] == 0) // If current pixel is blue
				{
					Mat fv(1, 2, CV_64FC1);

					fv.at<double>(0, 0) = j;
					fv.at<double>(0, 1) = i;
					X.push_back(fv);

					Mat yRowForCurrentImage(1, 1, CV_64FC1);
					yRowForCurrentImage.at<double>(0, 0) = -1;
					y.push_back(yRowForCurrentImage);
					
				}
				else if (currentPixel[0] == 0 && currentPixel[1] == 0 && currentPixel[2] == 255) // If current pixel is red
				{
					Mat fv(1, 2, CV_64FC1);
					fv.at<double>(0, 0) = j;
					fv.at<double>(0, 1) = i;
					X.push_back(fv);

					Mat yRowForCurrentImage(1, 1, CV_64FC1);
					yRowForCurrentImage.at<double>(0, 0) = 1;
					y.push_back(yRowForCurrentImage);
				}
				
			}
		}	


		 //Online perceptron
		const int d = 3;
		double learningRate = pow(10, -4);
		double w[] = { 0.1, 0.1, 0.1 };
		double ELimit = pow(10, -5);
		double maxIterations = pow(10, 5);
		
		// g(x) = w[0], w[1] * x1 + w[2] * x2 = wTransposed * x


		for (int currentIteration = 0; currentIteration < maxIterations; currentIteration++)
		{
			double E = 0;
			for (int i = 0; i < X.rows; i++)
			{
				double z = 0;
				
				z += w[0];
				for (int j = 1; j < d; j++)
				{
					z += w[j] * X.at<double>(i, j - 1);					
				}

				if (z * y.at<double>(i, 0) <= 0)
				{
					E++;

					w[0] += learningRate * y.at<double>(i, 0);
					for (int k = 1; k < d; k++)
					{
						w[k] += learningRate * y.at<double>(i, 0) * X.at<double>(i, k - 1);
					}
				}
			}


			E /= X.rows;
			printf("Iteration: %d, E: %lf\n", currentIteration, E);

			if (E < ELimit)
			{
				printf("\n\n Break and finish at %d iteration\n\n", currentIteration);
				break;
			}
		}

		Point2d point1(0, -w[0] / w[2]);
		Point2d point2(img.cols, (-w[0] - w[1] * img.cols) / w[2]);
		line(img, point1, point2, Scalar(0, 255, 0), 2);
		
		
		imshow("Perceptron Classification", img);

	}
}



struct weaklearner {
	int feature_i;
	int threshold;
	int class_label;
	double error;
	int classify(Mat X) {
		if (X.at<double>(feature_i) < threshold)
			return class_label;
		else
			return class_label * -1;
	}
};

weaklearner findWeakLearner(Mat X, Mat y, Mat w, int height, int width)
{
	weaklearner best_h = { 0, 0, -1, 1.0 };
	double best_err = INT_MAX;
	for (int j = 0; j < X.cols; j++)
	{
		int thresholdLimit;
		if (j == 0)
		{
			thresholdLimit = width;
		}
		else if(j == 1)
		{
			thresholdLimit = height;
		}

		for (int threshold = 0; threshold < thresholdLimit; threshold++)
		{
			for (int class_label = -1; class_label <= 1; class_label += 2)
			{
				double e = 0;
				for (int i = 0; i < X.rows; i++)
				{
					double zi;
					if (X.at<double>(i, j) < threshold)
					{
						zi = class_label;
					} else {
						zi = -1 * class_label;
					}
				
					if (zi * y.at<double>(i, 0) < 0)
					{
						e += w.at<double>(i, 0);
					}
				}

				if (e < best_err)
				{
					best_err = e;
					
					best_h.feature_i = j;
					best_h.threshold = threshold;
					best_h.class_label = class_label;
					best_h.error = e;
				}


			}
		}
	}

	return best_h;
}

const int GlobalT = 13;

struct classifier {
	int T = GlobalT;
	double alphas[GlobalT];
	weaklearner hs[GlobalT];
	int classify(Mat X) {
		double currentSum = 0;
		for (int i = 0; i < GlobalT; i++)
		{
			currentSum += alphas[i] * hs[i].classify(X);
		}
		if (currentSum < 0)
		{
			return -1;
		}
		else
		{
			return 1;
		}
	}
};


void adaptiveBoosting()
{
	char fname[MAX_PATH];
	Mat img;

	while (openFileDlg(fname))
	{
		img = imread(fname, IMREAD_COLOR);
		Mat X(0, 2, CV_64FC1);
		Mat y(0, 1, CV_64FC1);
		Mat weights(0, 1, CV_64FC1);
		
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				Vec3b currentPixel = img.at<Vec3b>(i, j);

				if (currentPixel[0] == 255 && currentPixel[1] == 0 && currentPixel[2] == 0) // If current pixel is blue
				{
					Mat fv(1, 2, CV_64FC1);

					fv.at<double>(0, 0) = j;
					fv.at<double>(0, 1) = i;
					X.push_back(fv);

					Mat yRowForCurrentImage(1, 1, CV_64FC1);
					yRowForCurrentImage.at<double>(0, 0) = -1;
					y.push_back(yRowForCurrentImage);

				}
				else if (currentPixel[0] == 0 && currentPixel[1] == 0 && currentPixel[2] == 255) // If current pixel is red
				{
					Mat fv(1, 2, CV_64FC1);
					fv.at<double>(0, 0) = j;
					fv.at<double>(0, 1) = i;
					X.push_back(fv);

					Mat yRowForCurrentImage(1, 1, CV_64FC1);
					yRowForCurrentImage.at<double>(0, 0) = 1;
					y.push_back(yRowForCurrentImage);
				}

			}
		}

		Mat initialWeights(1, 1, CV_64FC1);
		initialWeights.at<double>(0, 0) = 1.0 / X.rows;
		
		for (int i = 0; i < X.rows; i++)
		{
			weights.push_back(initialWeights);
		}

		// Classifier
		classifier classifier;
		for (int threshold = 0; threshold < GlobalT; threshold++)
		{
			weaklearner ht = findWeakLearner(X, y, weights, img.rows, img.cols);
			classifier.hs[threshold] = ht;

			double alpha = 0.5 * log((1 - ht.error) / ht.error);
			classifier.alphas[threshold] = alpha;
			
			double s = 0;
			for (int i = 0; i < X.rows; i++)
			{
				weights.at<double>(i, 0) *= exp(-alpha * y.at<double>(i, 0) * ht.classify(X.row(i)));
				s += weights.at<double>(i, 0);
			}

			for (int i = 0; i < X.rows; i++)
			{
				weights.at<double>(i, 0) /= s;
			}
		}


		Mat coloredImage(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				Vec3b currentPixel = img.at<Vec3b>(i, j);
				Vec3b color1(128, 128, 0);
				Vec3b color2(0, 128, 128);

				Mat aux(1, 2, CV_64FC1);
				aux.at<double>(0, 0) = j;
				aux.at<double>(0, 1) = i;

				if (currentPixel[0] == 255 && currentPixel[1] == 255 && currentPixel[2] == 255) // If current pixel is white
				{
					if (classifier.classify(aux) == -1)
					{
						coloredImage.at<Vec3b>(i, j) = color1;
					}
					else if(classifier.classify(aux) == 1)
					{
						coloredImage.at<Vec3b>(i, j) = color2;
					}
				}
				else
				{
					coloredImage.at<Vec3b>(i, j) = currentPixel;
				}
			}
		}

		imshow("Colored Image", coloredImage);
	}
}

Mat thresholdImage(Mat src, int thresholdValue)
{
	const int WHITE = 255;

	Mat thresholdedImage(src.rows, src.cols, CV_8UC1, Scalar(0, 0, 0));
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) > thresholdValue)
			{
				thresholdedImage.at<uchar>(i, j) = WHITE;
			}
		}
	}

	return thresholdedImage;
}

struct CenterOfMassInformation {
	Mat image;
	std::vector<Point> points;
	Mat areasMat; // CV_32SC1 - int
};

CenterOfMassInformation computeCentersOfMass(Mat labeled, std::vector<Vec3b> colorsVector, const int AREA_THRESHOLD)
{
	int height = labeled.rows;
	int width = labeled.cols;
	
	Mat centerOfMassImage(labeled.rows, labeled.cols, CV_8UC1, Scalar(0,0,0));
	std::vector<Point> centersOfMass;
	Mat areas = Mat::zeros(labeled.rows, labeled.cols, CV_32SC1); // int

	for (int k = 0; k < colorsVector.size(); k++)
	{
		Vec3b currentColor = colorsVector.at(k);

		// Compute area
		int area = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labeled.at<Vec3b>(i, j) == currentColor)
				{
					area++;
				}
			}
		}

		if (area > AREA_THRESHOLD)
		{
			// Compute center of mass
			int centerMassRow = 0, centerMassColumn = 0;
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					if (labeled.at<Vec3b>(i, j) == currentColor)
					{
						centerMassRow += i;
						centerMassColumn += j;
					}
				}
			}
			centerMassRow = ((1.0f / area) * centerMassRow);
			centerMassColumn = ((1.0f / area) * centerMassColumn);

			centersOfMass.push_back(Point(centerMassColumn, centerMassRow));
			areas.at<int>(centerMassRow, centerMassColumn) = area;
			centerOfMassImage.at<uchar>(centerMassRow, centerMassColumn) = WHITE;
		}
	}

	CenterOfMassInformation info;
	info.image = centerOfMassImage;
	info.points = centersOfMass;
	info.areasMat = areas;
	return info;
}


Mat computeLabelsMatrixBFS(Mat src)
{
	int di[8] = { -1, 0, 1, 0, -1, 1, -1, 1 };
	int dj[8] = { 0, -1, 0, 1, 1, -1, -1, 1 };
	uchar neighbors[8];

	int label = 0;
	Mat labels = Mat::zeros(src.rows, src.cols, CV_32SC1);
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			if ((src.at<uchar>(i, j) == 255) && (labels.at<int>(i, j) == 0))
			{
				label++;
				std::queue<Point2i> Q;
				labels.at<int>(i, j) = label;
				Q.push({ i,j });
				while (!Q.empty())
				{
					Point2i q = Q.front();
					Q.pop();

					for (int k = 0; k < 8; k++)
					{
						if (q.x + di[k] <= src.rows - 1 && q.x + di[k] >= 0 && q.y + dj[k] >= 0 && q.y + dj[k] <= src.cols - 1)
						{
							neighbors[k] = src.at<uchar>(q.x + di[k], q.y + dj[k]);

							if ((neighbors[k] == 255) && (labels.at<int>(q.x + di[k], q.y + dj[k]) == 0))
							{
								labels.at<int>(q.x + di[k], q.y + dj[k]) = label;
								Q.push({ q.x + di[k], q.y + dj[k] });
							}
						}
					}
				}
			}
		}
	}

	return labels;
}

std::vector<std::vector<Point>> inputCombinations;
std::vector<std::vector<Point>> constellationCombinations;
std::vector<Point> combination;

struct Distance {
	Point p1;
	Point p2;
	double value;

	Distance(Point point1, Point point2, double v) {
		p1 = point1;
		p2 = point2;
		value = v;
	}

	void swapPoints() {
		Point aux = p1;
		p1 = p2;
		p2 = aux;
	}
	
	bool operator <(const Distance& a) const
	{
		return value < a.value;
	}
};

struct Triangle {
	std::vector<Point> points; // Initially, no order. After computing distance, a certain order is established for all triangles.
	std::vector<Distance> distances;	// Hold distance + points of distance
	void computeDistances()
	{
		double d1Value = sqrt(pow((points[0].x - points[1].x), 2) + pow((points[0].y - points[1].y), 2));
		double d2Value = sqrt(pow((points[0].x - points[2].x), 2) + pow((points[0].y - points[2].y), 2));
		double d3Value = sqrt(pow((points[1].x - points[2].x), 2) + pow((points[1].y - points[2].y), 2));
		
		double minDistance = min(min(d1Value, d2Value), d3Value);
		
		d1Value /= minDistance;
		d2Value /= minDistance;
		d3Value /= minDistance;

		Distance d1(points[0], points[1], d1Value);
		Distance d2(points[0], points[2], d2Value);
		Distance d3(points[1], points[2], d3Value);
	
		distances.push_back(d1);
		distances.push_back(d2);
		distances.push_back(d3);

		sort(distances.begin(), distances.end());

		// Rearrange points of distances to maintain continuity
		if (distances[0].p1 == distances[1].p1) // ab ac XX -> ba ac XX
		{
			distances[0].swapPoints();
		}
		else if(distances[0].p1 == distances[1].p2) { // ab ca XX -> ba ac XX
			distances[0].swapPoints();
			distances[1].swapPoints();
		}
		else if (distances[0].p2 == distances[1].p1) { // ba ac
			// Nothing
		}
		else if (distances[0].p2 == distances[1].p2) { // ba ca XX -> ba ac XX
			distances[1].swapPoints();
		}

		if (distances[1].p2 == distances[2].p2) // XX ac bc -> XX ac cb
		{
			distances[2].swapPoints();
		}

		// Rearrange points in the points vector as well
		points.clear();
		points.push_back(distances[0].p1);
		points.push_back(distances[1].p1);
		points.push_back(distances[2].p1);
	}
};

void generateCombinationsInput(int offset, int k, std::vector<Point> source) {
	if (k == 0) {
		inputCombinations.push_back(combination);
		return;
	}
	for (int i = offset; i <= source.size() - k; ++i) {
		combination.push_back(source[i]);
		generateCombinationsInput(i + 1, k - 1, source);
		combination.pop_back();
	}
}

void generateCombinationsConstellation(int offset, int k, std::vector<Point> source) {
	if (k == 0) {
		constellationCombinations.push_back(combination);
		return;
	}
	for (int i = offset; i <= source.size() - k; ++i) {
		combination.push_back(source[i]);
		generateCombinationsConstellation(i + 1, k - 1, source);
		combination.pop_back();
	}
}

struct MatchingTrianglePair {
	Triangle inputTriangle;
	Triangle constellationTriangle;
	double difference;
	
	MatchingTrianglePair()
	{

	}

	MatchingTrianglePair(Triangle inputTr, Triangle constellationTr)
	{
		inputTriangle = inputTr;
		constellationTriangle = constellationTr;
		difference = abs(inputTriangle.distances[0].value - constellationTriangle.distances[0].value) +
			abs(inputTriangle.distances[1].value - constellationTriangle.distances[1].value) +
			abs(inputTriangle.distances[2].value - constellationTriangle.distances[2].value);
	}

	bool operator <(const MatchingTrianglePair& a) const
	{
		return difference < a.difference;
	}
};


Mat filterWarpedImageFloat(Mat warpedImage)
{
	for (int i = 0; i < warpedImage.rows; i++)
	{
		for (int j = 0; j < warpedImage.cols; j++)
		{
			if (warpedImage.at<float>(i, j) != WHITE)
			{
				warpedImage.at<float>(i, j) = BLACK;
			}
		}
	}

	return warpedImage;
}

struct MyPointSorter {
	bool operator() (cv::Point pt1, cv::Point pt2) { return (pt1.y < pt2.y); } // Sort based on y value.
} MyPointSorter;

struct PointCompare {
	bool operator() (cv::Point pt1, cv::Point pt2) { return (pt1.x < pt2.x || pt1.y < pt2.y); } // Sort based on y value.
};

struct Point2fCompare {
	bool operator() (cv::Point2f pt1, cv::Point2f pt2) { return (pt1.x < pt2.x || pt1.y < pt2.y); } // Sort based on y value.
};

struct SetOfPointsCompare {
	bool operator() (std::set<Point, PointCompare> set1, std::set<Point, PointCompare> set2) const
	{ 
		return std::lexicographical_compare(set1.begin(), set1.end(),
			set2.begin(), set2.end(), set1.key_comp());
	} 
};

void thesis()
{
	char fname[MAX_PATH];
	Mat src;
	const int BINARIZATION_THRESHOLD = 25;
	const double LUMINOSITY_THRESHOLD = 10.0f;
	const int NUMBER_OF_CONSTELLATIONS = 89;

	const double TRIANGLES_DIFFERENCE_THRESHOLD = 0.01; // Values to test: 0.01, 0.05
	const int AREA_THRESHOLD = 3; // Values to test : 3, 2
	const int POSITION_VARIATION_LOW_BOUND = -5; // Values to test : -2, -3, -5
	const int POSITION_VARIATION_HIGH_BOUND = 5; // Values to test: 2, 3, 5

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		imshow("Source", src);
		
		Mat thresholdedImage = thresholdImage(src, BINARIZATION_THRESHOLD);
		char buffer[50];
		sprintf(buffer, "Thresholded at %d", BINARIZATION_THRESHOLD);
		imshow(buffer, thresholdedImage);

		Mat labels = computeLabelsMatrixBFS(thresholdedImage);
		Mat labeledImage = computeLabeledImage(labels);
		imshow("Labeled", labeledImage);
	
		std::vector<Vec3b> colorsOfObjects = computeObjectsColorsBlackBackground(labeledImage);
		//printf("Number of stars: %d\n", colorsOfObjects.size());

		CenterOfMassInformation centerOfMassInformation = computeCentersOfMass(labeledImage, colorsOfObjects, AREA_THRESHOLD);
		imshow("Centers of mass", centerOfMassInformation.image);
		
		//printf("\nCenters of mass:\n");
		//for (int i = 0; i < centerOfMassInformation.points.size(); i++)
		//{
		//	printf("x=%d y=%d\n", centerOfMassInformation.points[i].x, centerOfMassInformation.points[i].y);
		//}

		
		std::vector<Point> inputPoints = centerOfMassInformation.points;
		inputCombinations.clear();
		generateCombinationsInput(0, 3, inputPoints);
		std::vector<Triangle> inputTriangles;

		for (int i = 0; i < inputCombinations.size(); i++)
		{
			Triangle triangle;
			triangle.points = inputCombinations[i];
			triangle.computeDistances();
			inputTriangles.push_back(triangle);
		}

		printf("\nInput image\nPoints:%d\nTriangles:%d\n",
			inputPoints.size(),
			inputTriangles.size());

		std::vector<std::vector<Point>> srcMatchingPoints; // All sets of matched points found in source image
		std::vector<int> srcMatchingPointsCounter; // Count the number of times the set is matched

		for (int currentConstellationNumber = 0; currentConstellationNumber < NUMBER_OF_CONSTELLATIONS; currentConstellationNumber++)
		{
			
			char fileName[250];
			if (currentConstellationNumber < 10)
				sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation0%d_data.txt", currentConstellationNumber);
			else
				sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation%d_data.txt", currentConstellationNumber);
			
			FILE* fp;
			fp = fopen(fileName, "r");
			if (fp != NULL)
			{
				std::vector<Point> constellationPoints;
				std::vector<Triangle> constellationTriangles;
				std::vector<std::vector<int>> constellationLines;

				// Read constellation file
				int nbOfPoints, nbOfLines, nbOfTriangles;
				fscanf(fp, "%d %d %d", &nbOfPoints, &nbOfLines, &nbOfTriangles);

				printf("\nConstellation %d\nPoints: %d\nLines: %d\nTriangles: %d\n",
					currentConstellationNumber,
					nbOfPoints,
					nbOfLines,
					nbOfTriangles);

				for (int i = 0; i < nbOfPoints; i++)
				{
					int index, x, y;
					fscanf(fp, "%d %d %d", &index, &x, &y); // index not used while reading, I use push_back so it's in order already
					constellationPoints.push_back(Point(x,y));
				}

				for (int i = 0; i < nbOfLines; i++)
				{
					int p1Index, p2Index;
					fscanf(fp, "%d %d", &p1Index, &p2Index);
					std::vector<int> aux;
					aux.push_back(p1Index);
					aux.push_back(p2Index);
					constellationLines.push_back(aux);
				}

				for (int i = 0; i < nbOfTriangles; i++)
				{
					Triangle currentTriangle;
					int x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6;
					double d1, d2, d3;	
					fscanf(fp, "%d %d %d %d %lf %d %d %d %d %lf %d %d %d %d %lf",
						&x1, &y1, &x2, &y2, &d1,
						&x3, &y3, &x4, &y4, &d2,
						&x5, &y5, &x6, &y6, &d3);

					// TODO: adding these points MIGHT be redundant 
					currentTriangle.points.push_back(Point(x1, y1));
					currentTriangle.points.push_back(Point(x3, y3));
					currentTriangle.points.push_back(Point(x5, y5));

					Distance distance1(Point(x1, y1), Point(x2, y2), d1);
					Distance distance2(Point(x3, y3), Point(x4, y4), d2);
					Distance distance3(Point(x5, y5), Point(x6, y6), d3);
				
					currentTriangle.distances.push_back(distance1);
					currentTriangle.distances.push_back(distance2);
					currentTriangle.distances.push_back(distance3);

					constellationTriangles.push_back(currentTriangle);
				}

				fclose(fp);
		
				auto startTime = std::chrono::high_resolution_clock::now();

				// Brute force comparison between source and constellation triangles
				printf("Started triangle computations...\n");
				std::vector<MatchingTrianglePair> allPairs;
				//int LIMIT_NB_TRIANGLES = 10;
				for (int i = 0; i < inputTriangles.size(); i++)
				{
					for (int j = 0; j < constellationTriangles.size(); j++)
					{
						MatchingTrianglePair currentPair(inputTriangles[i], constellationTriangles[j]);
						//if (allPairs.size() >= LIMIT_NB_TRIANGLES) break;
						if (currentPair.difference < TRIANGLES_DIFFERENCE_THRESHOLD)
						{
							allPairs.push_back(currentPair);
						}
					}
				}
				printf("Number of matching triangle pairs under %lf = %d\n", TRIANGLES_DIFFERENCE_THRESHOLD, allPairs.size());
			
				//sort(allPairs.begin(), allPairs.end()); // Trivial; all pairs are tested anyways.

				for (int i = 0; i < allPairs.size(); i++)
				{
					MatchingTrianglePair currentPair = allPairs[i];

					Point2f srcTri[3];
					srcTri[0] = Point2f(currentPair.constellationTriangle.points[0].x, currentPair.constellationTriangle.points[0].y);
					srcTri[1] = Point2f(currentPair.constellationTriangle.points[1].x, currentPair.constellationTriangle.points[1].y);
					srcTri[2] = Point2f(currentPair.constellationTriangle.points[2].x, currentPair.constellationTriangle.points[2].y);
					Point2f dstTri[3];
					dstTri[0] = Point2f(currentPair.inputTriangle.points[0].x, currentPair.inputTriangle.points[0].y);
					dstTri[1] = Point2f(currentPair.inputTriangle.points[1].x, currentPair.inputTriangle.points[1].y);
					dstTri[2] = Point2f(currentPair.inputTriangle.points[2].x, currentPair.inputTriangle.points[2].y);

					Mat warp_mat = getAffineTransform(srcTri, dstTri);
				
					// Warp constellation
					std::vector<Point2f> warpedConstellationPoints;
					for (int j = 0; j < constellationPoints.size(); j++)
					{
						Point2f nonWarpedPoint(constellationPoints[j].x, constellationPoints[j].y);
						Point2f warpedPoint;
						warpedPoint.x = warp_mat.at<double>(0, 0) * nonWarpedPoint.x + warp_mat.at<double>(0, 1) * nonWarpedPoint.y + warp_mat.at<double>(0, 2);
						warpedPoint.y =	warp_mat.at<double>(1, 0) * nonWarpedPoint.x + warp_mat.at<double>(1, 1) * nonWarpedPoint.y + warp_mat.at<double>(1, 2);
						warpedConstellationPoints.push_back(warpedPoint);
					}
				
					std::vector<Point> srcMatchingPointsCurrent; // All points matched in the source image for current iteration
					int srcMatchingPointsCurrentCounter; // Number of times all points matched in the source image for current iteration
					
					std::vector<int> matchedIndexes;
					int matchingStarsNb = 0;
					double meanLuminosity = 0.0f;
					// Compute number of matching stars between warped constellation and source image
					for (int j = 0; j < warpedConstellationPoints.size(); j++)
					{
						Point2f currentPoint = warpedConstellationPoints[j];
						int yOfPoint = (int)(currentPoint.y + 0.5f); // Float -> int : add 0.5 then (int)
						int xOfPoint = (int)(currentPoint.x + 0.5f);
						if (yOfPoint > centerOfMassInformation.image.rows || yOfPoint < 0) break; // Check out of bounds
						if (xOfPoint > centerOfMassInformation.image.cols || xOfPoint < 0) break;

						bool brokeInnerLoop = false;
						for (int index1 = POSITION_VARIATION_LOW_BOUND; index1 <= POSITION_VARIATION_HIGH_BOUND; index1++)
						{
							for (int index2 = POSITION_VARIATION_LOW_BOUND; index2 <= POSITION_VARIATION_HIGH_BOUND; index2++)
							{
								if (yOfPoint + index1 > (centerOfMassInformation.image.rows - 1) // Check out of bounds
									|| yOfPoint + index1 < 0
									|| xOfPoint + index2 > (centerOfMassInformation.image.cols - 1)
									|| xOfPoint + index2 < 0
									)
								{
									brokeInnerLoop = true;
									break;
								}
								if (centerOfMassInformation.image.at<uchar>(yOfPoint + index1, xOfPoint + index2) == WHITE)
								{
									srcMatchingPointsCurrent.push_back(Point(xOfPoint + index2, yOfPoint + index1));
									meanLuminosity += centerOfMassInformation.areasMat.at<int>(yOfPoint + index1, xOfPoint + index2);
									matchingStarsNb++;
									matchedIndexes.push_back(j);
									brokeInnerLoop = true;
									break;
								}
							}

							if (brokeInnerLoop) break;
						}
					
					}
					
					
					//if (matchingStarsNb > (int)(constellationPoints.size() * 0.7f + 0.5))
					if (matchingStarsNb == constellationPoints.size())
					{
						meanLuminosity /= matchingStarsNb;

						sort(srcMatchingPointsCurrent.begin(), srcMatchingPointsCurrent.end(), MyPointSorter);
						bool setAlreadyFound = false;
						for (int j = 0; j < srcMatchingPoints.size(); j++)
						{
							if (srcMatchingPointsCurrent == srcMatchingPoints[j])
							{
								srcMatchingPointsCounter[j]++;
								srcMatchingPointsCurrentCounter = srcMatchingPointsCounter[j];
								setAlreadyFound = true;
								break;
							}
						}

						//if (!setAlreadyFound)
						//{
						//	matchingSrcPoints.push_back(srcMatchingPointsCurrent);
						//	matchingSrcPointsCounter.push_back(1);
						//}

						if (!setAlreadyFound && meanLuminosity > LUMINOSITY_THRESHOLD)
						//if(srcMatchingPointsCurrentCounter > 1)
						{
							
							srcMatchingPoints.push_back(srcMatchingPointsCurrent);
							srcMatchingPointsCounter.push_back(1);
							printf("Pair %i Matching stars = %d, Difference = %lf, Luminosity=%lf", i, matchingStarsNb, currentPair.difference, meanLuminosity);

							Mat inputHighlightedImage = imread(fname, IMREAD_COLOR);
							char constellationImageFileName[250];
							if(currentConstellationNumber < 10)
								sprintf(constellationImageFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations//png//constellation0%d.png", currentConstellationNumber);
							else
								sprintf(constellationImageFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations//png//constellation%d.png", currentConstellationNumber);
							Mat constellationHighlightedImage = imread(constellationImageFileName, IMREAD_COLOR);

							// Drawing matching triangles
							line(inputHighlightedImage, currentPair.inputTriangle.points[0], currentPair.inputTriangle.points[1], Vec3b(255, 0, 0), 1);
							line(inputHighlightedImage, currentPair.inputTriangle.points[0], currentPair.inputTriangle.points[2], Vec3b(0, 255, 0), 1);
							line(inputHighlightedImage, currentPair.inputTriangle.points[1], currentPair.inputTriangle.points[2], Vec3b(0, 0, 255), 1);

							line(constellationHighlightedImage, currentPair.constellationTriangle.points[0], currentPair.constellationTriangle.points[1], Vec3b(255, 0, 0), 1);
							line(constellationHighlightedImage, currentPair.constellationTriangle.points[0], currentPair.constellationTriangle.points[2], Vec3b(0, 255, 0), 1);
							line(constellationHighlightedImage, currentPair.constellationTriangle.points[1], currentPair.constellationTriangle.points[2], Vec3b(0, 0, 255), 1);

							// Drawing points of warped constellation on input image
							for (int j = 0; j < matchedIndexes.size(); j++)
							{
								circle(inputHighlightedImage, warpedConstellationPoints[matchedIndexes[j]], 2, Vec3b(0, 0, 255), 3);
							}

							// Drawing lines of warped constellation on input image
							for (int j = 0; j < constellationLines.size(); j++)
							{
								std::vector<int> currentLine = constellationLines[j];
								if (std::find(matchedIndexes.begin(), matchedIndexes.begin(), currentLine[0]) != matchedIndexes.end() // If this line has both stars found
									&& std::find(matchedIndexes.begin(), matchedIndexes.begin(), currentLine[1]) != matchedIndexes.end())
								{
									line(inputHighlightedImage,
										warpedConstellationPoints[currentLine[0]], warpedConstellationPoints[currentLine[1]], Vec3b(255, 255, 0), 1);
								}

							}


							char buffer4[50];
							sprintf(buffer4, "inputHighlighted %d constellation %d", i, currentConstellationNumber);
							imshow(buffer4, inputHighlightedImage);
							char buffer5[50];
							sprintf(buffer5, "constellationHighlighted %d constellation %d", i, currentConstellationNumber);
							imshow(buffer5, constellationHighlightedImage);

							auto endTime = std::chrono::high_resolution_clock::now();
							auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
							printf(" Execution time:%d \n", duration);
						}
					}
				}
			}
		}
	}
}

void thesis_testOnSingularInputImage_withoutLines()
{
	char fname[MAX_PATH];
	Mat src;
	const int BINARIZATION_THRESHOLD = 25;
	const double LUMINOSITY_THRESHOLD = 10.0f;
	const int NUMBER_OF_CONSTELLATIONS = 89;

	const double TRIANGLES_DIFFERENCE_THRESHOLD = 0.1; // Values to test: 0.01, 0.05
	const int AREA_THRESHOLD = 2; // Values to test : 3, 2
	const int POSITION_VARIATION = 10; // Values to test : 2, 3, 5
	const int TARGET_CONSTELLATION = 3;

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		imshow("Source grayscale", src);

		Mat thresholdedImage = thresholdImage(src, BINARIZATION_THRESHOLD);
		char buffer[50];
		sprintf(buffer, "Thresholded at %d", BINARIZATION_THRESHOLD);
		imshow(buffer, thresholdedImage);

		Mat labels = computeLabelsMatrixBFS(thresholdedImage);
		Mat labeledImage = computeLabeledImage(labels);
		imshow("Labeled", labeledImage);

		std::vector<Vec3b> colorsOfObjects = computeObjectsColorsBlackBackground(labeledImage);
		//printf("Number of stars: %d\n", colorsOfObjects.size());

		CenterOfMassInformation centerOfMassInformation = computeCentersOfMass(labeledImage, colorsOfObjects, AREA_THRESHOLD);
		imshow("Centers of mass", centerOfMassInformation.image);

		//printf("\nCenters of mass:\n");
		//for (int i = 0; i < centerOfMassInformation.points.size(); i++)
		//{
		//	printf("x=%d y=%d\n", centerOfMassInformation.points[i].x, centerOfMassInformation.points[i].y);
		//}


		std::vector<Point> inputPoints = centerOfMassInformation.points;
		inputCombinations.clear();
		generateCombinationsInput(0, 3, inputPoints);
		std::vector<Triangle> inputTriangles;

		for (int i = 0; i < inputCombinations.size(); i++)
		{
			Triangle triangle;
			triangle.points = inputCombinations[i];
			triangle.computeDistances();
			inputTriangles.push_back(triangle);
		}

		printf("\nInput image\nPoints:%d\nTriangles:%d\n",
			inputPoints.size(),
			inputTriangles.size());

		std::vector<std::vector<Point>> srcMatchingPoints; // All sets of matched points found in source image
		std::vector<int> srcMatchingPointsCounter; // Count the number of times the set is matched

		for (int currentConstellationNumber = TARGET_CONSTELLATION; currentConstellationNumber < TARGET_CONSTELLATION+1; currentConstellationNumber++)
		{

			char fileName[250];
			if (currentConstellationNumber < 10)
				sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation0%d_data.txt", currentConstellationNumber);
			else
				sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation%d_data.txt", currentConstellationNumber);

			FILE* fp;
			fp = fopen(fileName, "r");
			if (fp != NULL)
			{
				std::vector<Point> constellationPoints;
				std::vector<Triangle> constellationTriangles;

				// Read constellation file
				int nbOfPoints, nbOfTriangles;
				fscanf(fp, "%d %d", &nbOfPoints, &nbOfTriangles);

				printf("\nConstellation %d\nPoints: %d\nTriangles: %d\n",
					currentConstellationNumber,
					nbOfPoints,
					nbOfTriangles);

				for (int i = 0; i < nbOfPoints; i++)
				{
					int index, x, y;
					fscanf(fp, "%d %d %d", &index, &x, &y); // index not used while reading, I use push_back so it's in order already
					constellationPoints.push_back(Point(x, y));
				}

				for (int i = 0; i < nbOfTriangles; i++)
				{
					Triangle currentTriangle;
					int x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6;
					double d1, d2, d3;
					fscanf(fp, "%d %d %d %d %lf %d %d %d %d %lf %d %d %d %d %lf",
						&x1, &y1, &x2, &y2, &d1,
						&x3, &y3, &x4, &y4, &d2,
						&x5, &y5, &x6, &y6, &d3);

					// TODO: adding these points MIGHT be redundant 
					currentTriangle.points.push_back(Point(x1, y1));
					currentTriangle.points.push_back(Point(x3, y3));
					currentTriangle.points.push_back(Point(x5, y5));

					Distance distance1(Point(x1, y1), Point(x2, y2), d1);
					Distance distance2(Point(x3, y3), Point(x4, y4), d2);
					Distance distance3(Point(x5, y5), Point(x6, y6), d3);

					currentTriangle.distances.push_back(distance1);
					currentTriangle.distances.push_back(distance2);
					currentTriangle.distances.push_back(distance3);

					constellationTriangles.push_back(currentTriangle);
				}
				fclose(fp);

				// Brute force comparison between source and constellation triangles
				printf("Started triangle computations...\n");
				std::vector<MatchingTrianglePair> allPairs;
				//int LIMIT_NB_TRIANGLES = 10;
				for (int i = 0; i < inputTriangles.size(); i++)
				{
					for (int j = 0; j < constellationTriangles.size(); j++)
					{
						MatchingTrianglePair currentPair(inputTriangles[i], constellationTriangles[j]);
						//if (allPairs.size() >= LIMIT_NB_TRIANGLES) break;
						if (currentPair.difference < TRIANGLES_DIFFERENCE_THRESHOLD)
						{
							allPairs.push_back(currentPair);
						}
					}
				}
				printf("Number of matching triangle pairs under %lf = %d\n", TRIANGLES_DIFFERENCE_THRESHOLD, allPairs.size());

				//sort(allPairs.begin(), allPairs.end()); // Trivial; all pairs are tested anyways.

				for (int i = 0; i < allPairs.size(); i++)
				{
					MatchingTrianglePair currentPair = allPairs[i];

					Point2f srcTri[3];
					srcTri[0] = Point2f(currentPair.constellationTriangle.points[0].x, currentPair.constellationTriangle.points[0].y);
					srcTri[1] = Point2f(currentPair.constellationTriangle.points[1].x, currentPair.constellationTriangle.points[1].y);
					srcTri[2] = Point2f(currentPair.constellationTriangle.points[2].x, currentPair.constellationTriangle.points[2].y);
					Point2f dstTri[3];
					dstTri[0] = Point2f(currentPair.inputTriangle.points[0].x, currentPair.inputTriangle.points[0].y);
					dstTri[1] = Point2f(currentPair.inputTriangle.points[1].x, currentPair.inputTriangle.points[1].y);
					dstTri[2] = Point2f(currentPair.inputTriangle.points[2].x, currentPair.inputTriangle.points[2].y);

					Mat warp_mat = getAffineTransform(srcTri, dstTri);

					// Warp constellation
					std::vector<Point2f> warpedConstellationPoints;
					for (int j = 0; j < constellationPoints.size(); j++)
					{
						Point2f nonWarpedPoint(constellationPoints[j].x, constellationPoints[j].y);
						Point2f warpedPoint;
						warpedPoint.x = warp_mat.at<double>(0, 0) * nonWarpedPoint.x + warp_mat.at<double>(0, 1) * nonWarpedPoint.y + warp_mat.at<double>(0, 2);
						warpedPoint.y = warp_mat.at<double>(1, 0) * nonWarpedPoint.x + warp_mat.at<double>(1, 1) * nonWarpedPoint.y + warp_mat.at<double>(1, 2);
						warpedConstellationPoints.push_back(warpedPoint);
					}

					std::vector<Point> srcMatchingPointsCurrent; // All points matched in the source image for current iteration
					int srcMatchingPointsCurrentCounter; // Number of times all points matched in the source image for current iteration

					std::vector<int> matchedIndexes;
					int matchingStarsNb = 0;
					double meanLuminosity = 0.0f;
					// Compute number of matching stars between warped constellation and source image
					for (int j = 0; j < warpedConstellationPoints.size(); j++)
					{
						Point2f currentPoint = warpedConstellationPoints[j];
						int yOfPoint = (int)(currentPoint.y + 0.5f); // Float -> int : add 0.5 then (int)
						int xOfPoint = (int)(currentPoint.x + 0.5f);
						if (yOfPoint > centerOfMassInformation.image.rows || yOfPoint < 0) break; // Check out of bounds
						if (xOfPoint > centerOfMassInformation.image.cols || xOfPoint < 0) break;

						bool brokeInnerLoop = false;
						for (int index1 = ((-1) * POSITION_VARIATION); index1 <= POSITION_VARIATION; index1++)
						{
							for (int index2 = ((-1) * POSITION_VARIATION); index2 <= POSITION_VARIATION; index2++)
							{
								if (yOfPoint + index1 > (centerOfMassInformation.image.rows - 1) // Check out of bounds
									|| yOfPoint + index1 < 0
									|| xOfPoint + index2 >(centerOfMassInformation.image.cols - 1)
									|| xOfPoint + index2 < 0
									)
								{
									brokeInnerLoop = true;
									break;
								}
								if (centerOfMassInformation.image.at<uchar>(yOfPoint + index1, xOfPoint + index2) == WHITE)
								{
									srcMatchingPointsCurrent.push_back(Point(xOfPoint + index2, yOfPoint + index1));
									meanLuminosity += centerOfMassInformation.areasMat.at<int>(yOfPoint + index1, xOfPoint + index2);
									matchingStarsNb++;
									matchedIndexes.push_back(j);
									brokeInnerLoop = true;
									break;
								}
							}

							if (brokeInnerLoop) break;
						}

					}

					if (matchingStarsNb == constellationPoints.size())
					//if (matchingStarsNb > 10)
					{
						meanLuminosity /= matchingStarsNb;

						sort(srcMatchingPointsCurrent.begin(), srcMatchingPointsCurrent.end(), MyPointSorter);
						bool setAlreadyFound = false;
						for (int j = 0; j < srcMatchingPoints.size(); j++)
						{
							if (srcMatchingPointsCurrent == srcMatchingPoints[j])
							{
								srcMatchingPointsCounter[j]++;
								srcMatchingPointsCurrentCounter = srcMatchingPointsCounter[j];
								setAlreadyFound = true;
								break;
							}
						}


						if (!setAlreadyFound && meanLuminosity > LUMINOSITY_THRESHOLD)
						{

							srcMatchingPoints.push_back(srcMatchingPointsCurrent);
							srcMatchingPointsCounter.push_back(1);
							printf("Pair %i Matching stars = %d, Difference = %lf, Luminosity=%lf\n", i, matchingStarsNb, currentPair.difference, meanLuminosity);

							Mat inputHighlightedImage = imread(fname, IMREAD_COLOR);
							char constellationImageFileName[250];
							if (currentConstellationNumber < 10)
								sprintf(constellationImageFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations//png//constellation0%d.png", currentConstellationNumber);
							else
								sprintf(constellationImageFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations//png//constellation%d.png", currentConstellationNumber);
							Mat constellationHighlightedImage = imread(constellationImageFileName, IMREAD_COLOR);

							// Drawing matching triangles
							line(inputHighlightedImage, currentPair.inputTriangle.points[0], currentPair.inputTriangle.points[1], Vec3b(255, 0, 0), 1);
							line(inputHighlightedImage, currentPair.inputTriangle.points[0], currentPair.inputTriangle.points[2], Vec3b(0, 255, 0), 1);
							line(inputHighlightedImage, currentPair.inputTriangle.points[1], currentPair.inputTriangle.points[2], Vec3b(0, 0, 255), 1);

							line(constellationHighlightedImage, currentPair.constellationTriangle.points[0], currentPair.constellationTriangle.points[1], Vec3b(255, 0, 0), 1);
							line(constellationHighlightedImage, currentPair.constellationTriangle.points[0], currentPair.constellationTriangle.points[2], Vec3b(0, 255, 0), 1);
							line(constellationHighlightedImage, currentPair.constellationTriangle.points[1], currentPair.constellationTriangle.points[2], Vec3b(0, 0, 255), 1);

							// Drawing points of warped constellation on input image
							for (int j = 0; j < matchedIndexes.size(); j++)
							{
								circle(inputHighlightedImage, warpedConstellationPoints[matchedIndexes[j]], 2, Vec3b(0, 0, 255), 3);
							}

							char buffer4[50];
							sprintf(buffer4, "inputHighlighted %d constellation %d", i, currentConstellationNumber);
							imshow(buffer4, inputHighlightedImage);
							char buffer5[50];
							sprintf(buffer5, "constellationHighlighted %d constellation %d", i, currentConstellationNumber);
							imshow(buffer5, constellationHighlightedImage);
						}
					}
				}
			}
		}
	}
}


void thesis_evaluation_noLines_onTargetConstellation_onFolder()
{
	char fname[MAX_PATH];
	Mat src;
	const int BINARIZATION_THRESHOLD = 25;
	const double LUMINOSITY_THRESHOLD = 5.5;
	const int NUMBER_OF_CONSTELLATIONS = 89;

	const double TRIANGLES_DIFFERENCE_THRESHOLD = 0.1; // Values to test: 0.01, 0.03, 0.05
	const int AREA_THRESHOLD = 3; // Values to test : 3, 2
	const int POSITION_VARIATION = 10; // Values to test: 2, 3, 5

	// Evaluation
	const int TARGET_CONSTELLATION = 3;
	int NB_IMAGES = 10;
	int totalMatches = 0;
	bool falseDetection = false;

	int detectionsOnImage[10] = { 0 };
	
	for (int imageIndex = 1; imageIndex <= NB_IMAGES; imageIndex++)
	{
		int iterationMatches = 0;
		sprintf(fname, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//evaluation_images//%d//%d.png", TARGET_CONSTELLATION, imageIndex);

		src = imread(fname, IMREAD_GRAYSCALE);
		//char bufferSrc[50];
		//sprintf(bufferSrc, "Src %d", imageIndex);
		//imshow(bufferSrc, src);

		Mat thresholdedImage = thresholdImage(src, BINARIZATION_THRESHOLD);
		//char buffer[50];
		//sprintf(buffer, "Thresholded at %d", BINARIZATION_THRESHOLD);
		//imshow(buffer, thresholdedImage);

		Mat labels = computeLabelsMatrixBFS(thresholdedImage);
		Mat labeledImage = computeLabeledImage(labels);
		//imshow("Labeled", labeledImage);

		std::vector<Vec3b> colorsOfObjects = computeObjectsColorsBlackBackground(labeledImage);
		//printf("Number of stars: %d\n", colorsOfObjects.size());

		CenterOfMassInformation centerOfMassInformation = computeCentersOfMass(labeledImage, colorsOfObjects, AREA_THRESHOLD);
		//imshow("Centers of mass", centerOfMassInformation.image);

		std::vector<Point> inputPoints = centerOfMassInformation.points;
		inputCombinations.clear();
		generateCombinationsInput(0, 3, inputPoints);
		std::vector<Triangle> inputTriangles;

		for (int i = 0; i < inputCombinations.size(); i++)
		{
			Triangle triangle;
			triangle.points = inputCombinations[i];
			triangle.computeDistances();
			inputTriangles.push_back(triangle);
		}

		printf("\n\n\nInput image:%d\nPoints:%d\nTriangles:%d\n",
			imageIndex,
			inputPoints.size(),
			inputTriangles.size());

		std::vector<std::vector<Point>> srcMatchingPoints; // All sets of matched points found in source image
		std::vector<int> srcMatchingPointsCounter; // Count the number of times the set is matched

		for (int currentConstellationNumber = TARGET_CONSTELLATION; currentConstellationNumber < TARGET_CONSTELLATION+1; currentConstellationNumber++)
		{

			char fileName[250];
			if (currentConstellationNumber < 10)
				sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation0%d_data.txt", currentConstellationNumber);
			else
				sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation%d_data.txt", currentConstellationNumber);

			FILE* fp;
			fp = fopen(fileName, "r");
			if (fp != NULL)
			{
				std::vector<Point> constellationPoints;
				std::vector<Triangle> constellationTriangles;
				//std::vector<std::vector<int>> constellationLines;

				// Read constellation file
				int nbOfPoints, nbOfTriangles;
				fscanf(fp, "%d %d", &nbOfPoints, &nbOfTriangles);

				printf("\nConstellation %d\nPoints: %d\nTriangles: %d\n",
					currentConstellationNumber,
					nbOfPoints,
					nbOfTriangles);

				for (int i = 0; i < nbOfPoints; i++)
				{
					int index, x, y;
					fscanf(fp, "%d %d %d", &index, &x, &y); // index not used while reading, I use push_back so it's in order already
					constellationPoints.push_back(Point(x, y));
				}

				for (int i = 0; i < nbOfTriangles; i++)
				{
					Triangle currentTriangle;
					int x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6;
					double d1, d2, d3;
					fscanf(fp, "%d %d %d %d %lf %d %d %d %d %lf %d %d %d %d %lf",
						&x1, &y1, &x2, &y2, &d1,
						&x3, &y3, &x4, &y4, &d2,
						&x5, &y5, &x6, &y6, &d3);

					// TODO: adding these points MIGHT be redundant 
					currentTriangle.points.push_back(Point(x1, y1));
					currentTriangle.points.push_back(Point(x3, y3));
					currentTriangle.points.push_back(Point(x5, y5));

					Distance distance1(Point(x1, y1), Point(x2, y2), d1);
					Distance distance2(Point(x3, y3), Point(x4, y4), d2);
					Distance distance3(Point(x5, y5), Point(x6, y6), d3);

					currentTriangle.distances.push_back(distance1);
					currentTriangle.distances.push_back(distance2);
					currentTriangle.distances.push_back(distance3);

					constellationTriangles.push_back(currentTriangle);
				}

				fclose(fp);

				//auto startTime = std::chrono::high_resolution_clock::now();

				// Brute force comparison between source and constellation triangles
				printf("Started triangle computations...\n");
				std::vector<MatchingTrianglePair> allPairs;
				//int LIMIT_NB_TRIANGLES = 10;
				for (int i = 0; i < inputTriangles.size(); i++)
				{
					for (int j = 0; j < constellationTriangles.size(); j++)
					{
						MatchingTrianglePair currentPair(inputTriangles[i], constellationTriangles[j]);
						//if (allPairs.size() >= LIMIT_NB_TRIANGLES) break;
						if (currentPair.difference < TRIANGLES_DIFFERENCE_THRESHOLD)
						{
							allPairs.push_back(currentPair);
						}
					}
				}
				printf("Number of matching triangle pairs under %lf = %d\n", TRIANGLES_DIFFERENCE_THRESHOLD, allPairs.size());

				//sort(allPairs.begin(), allPairs.end()); // Trivial; all pairs are tested anyways.

				for (int i = 0; i < allPairs.size(); i++)
				{
					MatchingTrianglePair currentPair = allPairs[i];

					Point2f srcTri[3];
					srcTri[0] = Point2f(currentPair.constellationTriangle.points[0].x, currentPair.constellationTriangle.points[0].y);
					srcTri[1] = Point2f(currentPair.constellationTriangle.points[1].x, currentPair.constellationTriangle.points[1].y);
					srcTri[2] = Point2f(currentPair.constellationTriangle.points[2].x, currentPair.constellationTriangle.points[2].y);
					Point2f dstTri[3];
					dstTri[0] = Point2f(currentPair.inputTriangle.points[0].x, currentPair.inputTriangle.points[0].y);
					dstTri[1] = Point2f(currentPair.inputTriangle.points[1].x, currentPair.inputTriangle.points[1].y);
					dstTri[2] = Point2f(currentPair.inputTriangle.points[2].x, currentPair.inputTriangle.points[2].y);

					Mat warp_mat = getAffineTransform(srcTri, dstTri);

					// Warp constellation
					std::vector<Point2f> warpedConstellationPoints;
					for (int j = 0; j < constellationPoints.size(); j++)
					{
						Point2f nonWarpedPoint(constellationPoints[j].x, constellationPoints[j].y);
						Point2f warpedPoint;
						warpedPoint.x = warp_mat.at<double>(0, 0) * nonWarpedPoint.x + warp_mat.at<double>(0, 1) * nonWarpedPoint.y + warp_mat.at<double>(0, 2);
						warpedPoint.y = warp_mat.at<double>(1, 0) * nonWarpedPoint.x + warp_mat.at<double>(1, 1) * nonWarpedPoint.y + warp_mat.at<double>(1, 2);
						warpedConstellationPoints.push_back(warpedPoint);
					}

					std::vector<Point> srcMatchingPointsCurrent; // All points matched in the source image for current iteration
					int srcMatchingPointsCurrentCounter; // Number of times all points matched in the source image for current iteration

					std::vector<int> matchedIndexes;
					int matchingStarsNb = 0;
					double meanLuminosity = 0.0f;
					// Compute number of matching stars between warped constellation and source image
					for (int j = 0; j < warpedConstellationPoints.size(); j++)
					{
						Point2f currentPoint = warpedConstellationPoints[j];
						int yOfPoint = (int)(currentPoint.y + 0.5f); // Float -> int : add 0.5 then (int)
						int xOfPoint = (int)(currentPoint.x + 0.5f);
						if (yOfPoint > centerOfMassInformation.image.rows || yOfPoint < 0) break; // Check out of bounds
						if (xOfPoint > centerOfMassInformation.image.cols || xOfPoint < 0) break;

						bool brokeInnerLoop = false;
						for (int index0 = 0; index0 <= POSITION_VARIATION; index0++)
						{
							for (int index1 = (-1 * index0); index1 <= index0; index1++)
							{
								for (int index2 = (-1 * index0); index2 <= index0; index2++)
								{
									if (yOfPoint + index1 > (centerOfMassInformation.image.rows - 1) // Check out of bounds
										|| yOfPoint + index1 < 0
										|| xOfPoint + index2 >(centerOfMassInformation.image.cols - 1)
										|| xOfPoint + index2 < 0
										)
									{
										brokeInnerLoop = true;
										break;
									}
									if (centerOfMassInformation.image.at<uchar>(yOfPoint + index1, xOfPoint + index2) == WHITE)
									{
										srcMatchingPointsCurrent.push_back(Point(xOfPoint + index2, yOfPoint + index1));
										meanLuminosity += centerOfMassInformation.areasMat.at<int>(yOfPoint + index1, xOfPoint + index2);
										matchingStarsNb++;
										matchedIndexes.push_back(j);
										brokeInnerLoop = true;
										break;
									}
								}

								if (brokeInnerLoop) break;
							}

							if (brokeInnerLoop) break;
						}
					}


					//if (matchingStarsNb > (int)(constellationPoints.size() * 0.7f + 0.5))
					if (matchingStarsNb == constellationPoints.size())
					{
						meanLuminosity /= matchingStarsNb;

						sort(srcMatchingPointsCurrent.begin(), srcMatchingPointsCurrent.end(), MyPointSorter);
						bool setAlreadyFound = false;
						for (int j = 0; j < srcMatchingPoints.size(); j++)
						{
							if (srcMatchingPointsCurrent == srcMatchingPoints[j])
							{
								srcMatchingPointsCounter[j]++;
								srcMatchingPointsCurrentCounter = srcMatchingPointsCounter[j];
								setAlreadyFound = true;
								break;
							}
						}



						if (!setAlreadyFound && meanLuminosity > LUMINOSITY_THRESHOLD)
						{
							srcMatchingPoints.push_back(srcMatchingPointsCurrent);
							srcMatchingPointsCounter.push_back(1);
							printf("Image %d Pair %i Matching stars = %d, Difference = %lf, Luminosity=%lf\n", imageIndex, i, matchingStarsNb, currentPair.difference, meanLuminosity);
							
							iterationMatches++;

							Mat inputHighlightedImage = imread(fname, IMREAD_COLOR);
							char constellationImageFileName[250];
							if (currentConstellationNumber < 10)
								sprintf(constellationImageFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations//png//constellation0%d.png", currentConstellationNumber);
							else
								sprintf(constellationImageFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations//png//constellation%d.png", currentConstellationNumber);
							Mat constellationHighlightedImage = imread(constellationImageFileName, IMREAD_COLOR);

							// Drawing matching triangles
							line(inputHighlightedImage, currentPair.inputTriangle.points[0], currentPair.inputTriangle.points[1], Vec3b(255, 0, 0), 1);
							line(inputHighlightedImage, currentPair.inputTriangle.points[0], currentPair.inputTriangle.points[2], Vec3b(0, 255, 0), 1);
							line(inputHighlightedImage, currentPair.inputTriangle.points[1], currentPair.inputTriangle.points[2], Vec3b(0, 0, 255), 1);

							line(constellationHighlightedImage, currentPair.constellationTriangle.points[0], currentPair.constellationTriangle.points[1], Vec3b(255, 0, 0), 1);
							line(constellationHighlightedImage, currentPair.constellationTriangle.points[0], currentPair.constellationTriangle.points[2], Vec3b(0, 255, 0), 1);
							line(constellationHighlightedImage, currentPair.constellationTriangle.points[1], currentPair.constellationTriangle.points[2], Vec3b(0, 0, 255), 1);

							// Drawing points of warped constellation on input image
							for (int j = 0; j < matchedIndexes.size(); j++)
							{
								circle(inputHighlightedImage, warpedConstellationPoints[matchedIndexes[j]], 2, Vec3b(0, 0, 255), 3);
							}

							// Drawing lines of warped constellation on input image
							//for (int j = 0; j < constellationLines.size(); j++)
							//{
							//	std::vector<int> currentLine = constellationLines[j];
							//	if (std::find(matchedIndexes.begin(), matchedIndexes.begin(), currentLine[0]) != matchedIndexes.end() // If this line has both stars found
							//		&& std::find(matchedIndexes.begin(), matchedIndexes.begin(), currentLine[1]) != matchedIndexes.end())
							//	{
							//		line(inputHighlightedImage,
							//			warpedConstellationPoints[currentLine[0]], warpedConstellationPoints[currentLine[1]], Vec3b(255, 255, 0), 1);
							//	}
							//}

							char buffer4[50];
							sprintf(buffer4, "image %d inputHighlighted %d constellation %d", imageIndex, i, currentConstellationNumber);
							imshow(buffer4, inputHighlightedImage);
							
							/*char buffer5[50];
							sprintf(buffer5, "image %d constellationHighlighted %d constellation %d", imageIndex, i, currentConstellationNumber);
							imshow(buffer5, constellationHighlightedImage);*/

							//auto endTime = std::chrono::high_resolution_clock::now();
							//auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
							//printf(" Execution time:%d \n", duration);

							detectionsOnImage[imageIndex]++;
						}
					}
				}
			}
		}

		
		if (iterationMatches > 1)
		{
			printf("\n\nPossible false detections!!!\tFound %d detections on image %d.\n", iterationMatches, imageIndex);
			falseDetection = true;
			//break;
		}
		else
		{
			totalMatches += iterationMatches;
		}
	}

	if (!falseDetection)
	{
		printf("\n\nNo multiple detections.\n");
		printf("Matches: %d out of %d\n", totalMatches, NB_IMAGES);
		printf("Success rate: %lf\n", (double)totalMatches / NB_IMAGES);
	}
	else
	{
		printf("\n\nMulitple detections on 1 image found.\n");
		for (int i = 0; i < 10; i++)
		{
			if (detectionsOnImage[i] > 1)
			{
				printf("Image %d: %d detections\n", i, detectionsOnImage[i]);
			}
		}
	}

	Mat aux = Mat::zeros(100, 100, CV_8UC1);
	imshow("aux", aux);
	waitKey(0);
}

struct Detection {
	
	MatchingTrianglePair trianglePair;
	std::set<Point, PointCompare> srcPoints;
	double luminosity;
	double difference;
	int sortingOrder;

	Detection() {};
	
	Detection(std::set<Point, PointCompare> srcPts, MatchingTrianglePair triPair, double lum, int sortOrder)
	{
		srcPoints = srcPts;
		trianglePair = triPair;
		luminosity = lum;
		difference = trianglePair.difference;
		sortingOrder = sortOrder;
	}

	bool operator <(const Detection& a) const
	{
		if (sortingOrder == 0)
		{
			return luminosity > a.luminosity;
		}
		else
		{
			return difference < a.difference;
		}
	}
};

struct DetectionVisual {
	Mat inputImage;
	Mat constellationImage;
	int number;
	
	DetectionVisual(Mat inputImg, Mat constellationImg, int nb)
	{
		inputImage = inputImg;
		constellationImage = constellationImg;
		number = nb;
	}
};

void thesis_testOnSingularInputImage_withAllConstellations_withoutLines()
{
	char fname[MAX_PATH];
	Mat src;
	const int BINARIZATION_THRESHOLD = 25;
	//const double LUMINOSITY_THRESHOLD = 10.0f;
	const int NUMBER_OF_CONSTELLATIONS = 89;

	//const double TRIANGLES_DIFFERENCE_THRESHOLD = 0.05; // Values to test: 0.01, 0.05
	//const int AREA_THRESHOLD = 3; // Values to test : 3, 2
	//const int POSITION_VARIATION = 5; // Values to test : 2, 3, 5
	//const int TARGET_CONSTELLATION = 0;

	double params_triangDiffThresh[NUMBER_OF_CONSTELLATIONS];
	int params_areaThresh[NUMBER_OF_CONSTELLATIONS];
	int params_posVar[NUMBER_OF_CONSTELLATIONS];
	double params_lum[NUMBER_OF_CONSTELLATIONS];
	int params_methodsMultDetect[NUMBER_OF_CONSTELLATIONS];
	const int METHOD_GREATER_LUMINOSITY = 0;
	const int METHOD_LOWEST_DIFFERENCE = 1;
	const int METHOD_ANY = 2;

	char detectionParamsFileName[250];
	sprintf(detectionParamsFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//detection_parameters_constellations.txt");
	FILE* detectionParamsFp = fopen(detectionParamsFileName, "r");
	if (detectionParamsFp == NULL)
	{
		printf("Could not open detection params file.\n");
	}
	else
	{
		for (int i = 0; i < NUMBER_OF_CONSTELLATIONS; i++)
		{
			int constellationNb = -1;
			fscanf(detectionParamsFp, "%d,%lf,%d,%d,%lf,%d",
				&constellationNb,
				&params_triangDiffThresh[i],
				&params_areaThresh[i],
				&params_posVar[i],
				&params_lum[i],
				&params_methodsMultDetect[i]);
		}

		while (openFileDlg(fname))
		{
			clock_t startTime, endTime;

			src = imread(fname, IMREAD_GRAYSCALE);
			imshow("Source grayscale", src);

			startTime = clock();

			// TODO: Consider adaptive thresholding 
			Mat thresholdedImage = thresholdImage(src, BINARIZATION_THRESHOLD);
			char buffer[50];
			sprintf(buffer, "Thresholded at %d", BINARIZATION_THRESHOLD);
			imshow(buffer, thresholdedImage);

			Mat labels = computeLabelsMatrixBFS(thresholdedImage);
			Mat labeledImage = computeLabeledImage(labels);
			imshow("Labeled", labeledImage);

			std::vector<Vec3b> colorsOfObjects = computeObjectsColorsBlackBackground(labeledImage);
			//printf("Number of stars: %d\n", colorsOfObjects.size());
			int matchesOnConstellations[NUMBER_OF_CONSTELLATIONS] = { 0 };

			std::vector<DetectionVisual> detectionVisuals;

			omp_lock_t lock;
			omp_init_lock(&lock);

			omp_set_num_threads(NUMBER_OF_CONSTELLATIONS);
			#pragma omp parallel for
			for (int currentConstellationNumber = 0; currentConstellationNumber < NUMBER_OF_CONSTELLATIONS; currentConstellationNumber++)
			{
				if (params_methodsMultDetect[currentConstellationNumber] == -1) continue;

				int currentIterationMatches = 0;

				CenterOfMassInformation centerOfMassInformation = computeCentersOfMass(labeledImage, colorsOfObjects, params_areaThresh[currentConstellationNumber]);
				//imshow("Centers of mass", centerOfMassInformation.image);

				std::vector<Point> inputPoints = centerOfMassInformation.points;
				omp_set_lock(&lock);
				inputCombinations.clear();
				generateCombinationsInput(0, 3, inputPoints);

				std::vector<Triangle> inputTriangles;

				for (int i = 0; i < inputCombinations.size(); i++)
				{
					Triangle triangle;
					triangle.points = inputCombinations[i];
					triangle.computeDistances();
					inputTriangles.push_back(triangle);
				}
				omp_unset_lock(&lock);

				printf("\nInput image\nPoints:%d\nTriangles:%d\n",
					inputPoints.size(),
					inputTriangles.size());

				std::set<std::set<Point, PointCompare>, SetOfPointsCompare> matchingSrcPoints; // All sets of matched points found in source image
				std::vector<Detection> allDetections;

				char fileName[250];
				if (currentConstellationNumber < 10)
					sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation0%d_data.txt", currentConstellationNumber);
				else
					sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation%d_data.txt", currentConstellationNumber);

				FILE* fp;
				fp = fopen(fileName, "r");
				if (fp != NULL)
				{
					std::vector<Point> constellationPoints;
					std::vector<Triangle> constellationTriangles;

					// Read constellation file
					int nbOfPoints, nbOfTriangles;
					fscanf(fp, "%d %d", &nbOfPoints, &nbOfTriangles);

					printf("\nConstellation %d\nPoints: %d\nTriangles: %d\n",
						currentConstellationNumber,
						nbOfPoints,
						nbOfTriangles);

					for (int i = 0; i < nbOfPoints; i++)
					{
						int index, x, y;
						fscanf(fp, "%d %d %d", &index, &x, &y); // index not used while reading, I use push_back so it's in order already
						constellationPoints.push_back(Point(x, y));
					}

					for (int i = 0; i < nbOfTriangles; i++)
					{
						Triangle currentTriangle;
						int x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6;
						double d1, d2, d3;
						fscanf(fp, "%d %d %d %d %lf %d %d %d %d %lf %d %d %d %d %lf",
							&x1, &y1, &x2, &y2, &d1,
							&x3, &y3, &x4, &y4, &d2,
							&x5, &y5, &x6, &y6, &d3);

						// TODO: adding these points MIGHT be redundant 
						currentTriangle.points.push_back(Point(x1, y1));
						currentTriangle.points.push_back(Point(x3, y3));
						currentTriangle.points.push_back(Point(x5, y5));

						Distance distance1(Point(x1, y1), Point(x2, y2), d1);
						Distance distance2(Point(x3, y3), Point(x4, y4), d2);
						Distance distance3(Point(x5, y5), Point(x6, y6), d3);

						currentTriangle.distances.push_back(distance1);
						currentTriangle.distances.push_back(distance2);
						currentTriangle.distances.push_back(distance3);

						constellationTriangles.push_back(currentTriangle);
					}
					fclose(fp);

					// Brute force comparison between source and constellation triangles
					printf("Started triangle computations...\n");
					std::vector<MatchingTrianglePair> allPairs;
					
					// TODO: Parallelize the comparison
					//omp_set_num_threads(inputTriangles.size());
					//#pragma omp parallel for collapse(2)
					for (int i = 0; i < inputTriangles.size(); i++)
					{
						for (int j = 0; j < constellationTriangles.size(); j++)
						{
							MatchingTrianglePair currentPair(inputTriangles[i], constellationTriangles[j]);
							if (currentPair.difference < params_triangDiffThresh[currentConstellationNumber])
							{
								allPairs.push_back(currentPair);
							}
						}
					}
					printf("Number of matching triangle pairs under %lf = %d\n", params_triangDiffThresh[currentConstellationNumber], allPairs.size());

					//sort(allPairs.begin(), allPairs.end()); // Trivial; all pairs are tested anyways.

					for (int i = 0; i < allPairs.size(); i++)
					{
						MatchingTrianglePair currentPair = allPairs[i];

						Point2f srcTri[3];
						srcTri[0] = Point2f(currentPair.constellationTriangle.points[0].x, currentPair.constellationTriangle.points[0].y);
						srcTri[1] = Point2f(currentPair.constellationTriangle.points[1].x, currentPair.constellationTriangle.points[1].y);
						srcTri[2] = Point2f(currentPair.constellationTriangle.points[2].x, currentPair.constellationTriangle.points[2].y);
						Point2f dstTri[3];
						dstTri[0] = Point2f(currentPair.inputTriangle.points[0].x, currentPair.inputTriangle.points[0].y);
						dstTri[1] = Point2f(currentPair.inputTriangle.points[1].x, currentPair.inputTriangle.points[1].y);
						dstTri[2] = Point2f(currentPair.inputTriangle.points[2].x, currentPair.inputTriangle.points[2].y);

						Mat warp_mat = getAffineTransform(srcTri, dstTri);

						// Warp constellation
						std::vector<Point2f> warpedConstellationPoints;
						for (int j = 0; j < constellationPoints.size(); j++)
						{
							Point2f nonWarpedPoint(constellationPoints[j].x, constellationPoints[j].y);
							Point2f warpedPoint;
							warpedPoint.x = warp_mat.at<double>(0, 0) * nonWarpedPoint.x + warp_mat.at<double>(0, 1) * nonWarpedPoint.y + warp_mat.at<double>(0, 2);
							warpedPoint.y = warp_mat.at<double>(1, 0) * nonWarpedPoint.x + warp_mat.at<double>(1, 1) * nonWarpedPoint.y + warp_mat.at<double>(1, 2);
							warpedConstellationPoints.push_back(warpedPoint);
						}

						std::set<Point, PointCompare> srcMatchingPointsCurrent; // All points matched in the source image for current iteration
						int srcMatchingPointsCurrentCounter; // Number of times all points matched in the source image for current iteration

						std::vector<int> matchedIndexes;
						//int matchingStarsNb = 0;
						double meanLuminosity = 0.0f;
						// Compute number of matching stars between warped constellation and source image
						for (int j = 0; j < warpedConstellationPoints.size(); j++)
						{
							Point2f currentPoint = warpedConstellationPoints[j];
							int yOfPoint = (int)(currentPoint.y + 0.5f); // Float -> int : add 0.5 then (int)
							int xOfPoint = (int)(currentPoint.x + 0.5f);
							if (yOfPoint > centerOfMassInformation.image.rows || yOfPoint < 0) break; // Check out of bounds
							if (xOfPoint > centerOfMassInformation.image.cols || xOfPoint < 0) break;

							bool brokeInnerLoop = false;
							for (int index0 = 0; index0 <= params_posVar[currentConstellationNumber]; index0++)
							{
								for (int index1 = (-1 * index0); index1 <= index0; index1++)
								{
									for (int index2 = (-1 * index0); index2 <= index0; index2++)
									{
										if (yOfPoint + index1 > (centerOfMassInformation.image.rows - 1) // Check out of bounds
											|| yOfPoint + index1 < 0
											|| xOfPoint + index2 >(centerOfMassInformation.image.cols - 1)
											|| xOfPoint + index2 < 0
											)
										{
											brokeInnerLoop = true;
											break;
										}
										if (centerOfMassInformation.image.at<uchar>(yOfPoint + index1, xOfPoint + index2) == WHITE)
										{
											srcMatchingPointsCurrent.insert(Point(xOfPoint + index2, yOfPoint + index1));
											meanLuminosity += centerOfMassInformation.areasMat.at<int>(yOfPoint + index1, xOfPoint + index2);
											//matchingStarsNb++;
											matchedIndexes.push_back(j);
											brokeInnerLoop = true;
											break;
										}
									}

									if (brokeInnerLoop) break;
								}
								
								if (brokeInnerLoop) break;
							}

						}

						if (srcMatchingPointsCurrent.size() == constellationPoints.size())
						{
							meanLuminosity /= srcMatchingPointsCurrent.size();

							//sort(srcMatchingPointsCurrent.begin(), srcMatchingPointsCurrent.end(), MyPointSorter);
							//std::set<Point, PointCompare>::iterator iterator;
							bool setAlreadyFound = false;

							if (matchingSrcPoints.find(srcMatchingPointsCurrent) != matchingSrcPoints.end())
							{
								//srcMatchingPointsCurrentCounter = matchingSrcPointsCounter[j];
								//int index = matchingSrcPoints.find(srcMatchingPointsCurrent);
								//matchingSrcPointsCounter[]++;
								setAlreadyFound = true;
								break;
							}
							


							if (!setAlreadyFound && meanLuminosity > params_lum[currentConstellationNumber])
							{
								currentIterationMatches++;
								
								matchingSrcPoints.insert(srcMatchingPointsCurrent);
								Detection detection(srcMatchingPointsCurrent, currentPair, meanLuminosity, params_methodsMultDetect[currentConstellationNumber]);
								allDetections.push_back(detection);
							}
						}
					}
				}

					
				if (allDetections.size() > 0)
				{
					sort(allDetections.begin(), allDetections.end());
					
					for (int j = 0; j < allDetections.size(); j++)
					{
						printf("Detection %d Order=%d Lum=%lf Diff=%lf\n", j, allDetections[j].sortingOrder, allDetections[j].luminosity, allDetections[j].difference);
					}

					Detection selectedDetection = allDetections[0];
						
					printf("Number of detections: %d\n", allDetections.size());
					if (selectedDetection.sortingOrder == METHOD_GREATER_LUMINOSITY)
						printf("Showing the detection with Greatest Luminosity.\n");
					else
						printf("Showing the detection with Lowest Difference between triangles\n");
						
					printf("Nb. of points:%d\nDifference:%lf\nLuminosity:%lf\n", 
						selectedDetection.srcPoints.size(),
						selectedDetection.difference,
						selectedDetection.luminosity
					);

					Mat inputHighlightedImage = imread(fname, IMREAD_COLOR);
					char constellationImageFileName[250];
					if (currentConstellationNumber < 10)
						sprintf(constellationImageFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations//png//constellation0%d.png", currentConstellationNumber);
					else
						sprintf(constellationImageFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations//png//constellation%d.png", currentConstellationNumber);
					Mat constellationHighlightedImage = imread(constellationImageFileName, IMREAD_COLOR);

					// Drawing matching triangles
					line(inputHighlightedImage, selectedDetection.trianglePair.inputTriangle.points[0], selectedDetection.trianglePair.inputTriangle.points[1], Vec3b(255, 0, 0), 1);
					line(inputHighlightedImage, selectedDetection.trianglePair.inputTriangle.points[0], selectedDetection.trianglePair.inputTriangle.points[2], Vec3b(0, 255, 0), 1);
					line(inputHighlightedImage, selectedDetection.trianglePair.inputTriangle.points[1], selectedDetection.trianglePair.inputTriangle.points[2], Vec3b(0, 0, 255), 1);

					line(constellationHighlightedImage, selectedDetection.trianglePair.constellationTriangle.points[0], selectedDetection.trianglePair.constellationTriangle.points[1], Vec3b(255, 0, 0), 1);
					line(constellationHighlightedImage, selectedDetection.trianglePair.constellationTriangle.points[0], selectedDetection.trianglePair.constellationTriangle.points[2], Vec3b(0, 255, 0), 1);
					line(constellationHighlightedImage, selectedDetection.trianglePair.constellationTriangle.points[1], selectedDetection.trianglePair.constellationTriangle.points[2], Vec3b(0, 0, 255), 1);

					// Drawing points of warped constellation on input image
					for (Point point : selectedDetection.srcPoints)
					{
						circle(inputHighlightedImage, point, 2, Vec3b(0, 0, 255), 3);
					}

					DetectionVisual detectionVisual(inputHighlightedImage, constellationHighlightedImage, currentConstellationNumber);

					omp_set_lock(&lock);
					detectionVisuals.push_back(detectionVisual);
					omp_unset_lock(&lock);
				}


				if (currentIterationMatches > 0)
				{
					omp_set_lock(&lock);
					matchesOnConstellations[currentConstellationNumber] = currentIterationMatches;
					omp_unset_lock(&lock);
				}
			}

			for (int i = 0; i < detectionVisuals.size(); i++)
			{
				char bufferInputImage[100];
				sprintf(bufferInputImage, "inputHighlighted %d", detectionVisuals[i].number);
				imshow(bufferInputImage, detectionVisuals[i].inputImage);

				char bufferConstellationImage[100];
				sprintf(bufferConstellationImage, "constellationHighlighted %d", detectionVisuals[i].number);
				imshow(bufferConstellationImage, detectionVisuals[i].constellationImage);
			}

			endTime = clock();
			double timeTaken = double(endTime - startTime) / double(CLOCKS_PER_SEC);


			printf("\n\n\n ----- Stats ----- \n\n\n");
			printf("Time: %lf seconds.\n", timeTaken);
			for (int i = 0; i < NUMBER_OF_CONSTELLATIONS; i++)
			{
				if (matchesOnConstellations[i] > 0)
				{
					if (matchesOnConstellations[i] == 1)
					{
						printf("Constellation %d: %d match.\n", i, matchesOnConstellations[i]);
					}
					else
					{
						printf("Constellation %d: %d match.\n", i, matchesOnConstellations[i]);
					}
				}
			}
		}
	}
}

void thesis_testOnSingularInputImageFromParameters_withAllConstellations_withoutLines()
{
	char fname[MAX_PATH];
	Mat src;
	const int BINARIZATION_THRESHOLD = 25;
	const int NUMBER_OF_CONSTELLATIONS = 89;

	int SELECTION_CONSTELLATION = 51;
	int SELECTION_IMAGE = 7;

	double params_triangDiffThresh[NUMBER_OF_CONSTELLATIONS];
	int params_areaThresh[NUMBER_OF_CONSTELLATIONS];
	int params_posVar[NUMBER_OF_CONSTELLATIONS];
	double params_lum[NUMBER_OF_CONSTELLATIONS];
	int params_methodsMultDetect[NUMBER_OF_CONSTELLATIONS];
	const int METHOD_GREATER_LUMINOSITY = 0;
	const int METHOD_LOWEST_DIFFERENCE = 1;
	const int METHOD_ANY = 2;

	char detectionParamsFileName[250];
	sprintf(detectionParamsFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//detection_parameters_constellations.txt");
	FILE* detectionParamsFp = fopen(detectionParamsFileName, "r");
	if (detectionParamsFp == NULL)
	{
		printf("Could not open detection params file.\n");
	}
	else
	{
		for (int i = 0; i < NUMBER_OF_CONSTELLATIONS; i++)
		{
			int constellationNb = -1;
			fscanf(detectionParamsFp, "%d,%lf,%d,%d,%lf,%d",
				&constellationNb,
				&params_triangDiffThresh[i],
				&params_areaThresh[i],
				&params_posVar[i],
				&params_lum[i],
				&params_methodsMultDetect[i]);
		}

		clock_t startTime, endTime;

		sprintf(fname, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//evaluation_images//%d//%d.png", SELECTION_CONSTELLATION, SELECTION_IMAGE);
		src = imread(fname, IMREAD_GRAYSCALE);
		imshow("Source grayscale", src);

		startTime = clock();

		// TODO: Consider adaptive thresholding 
		Mat thresholdedImage = thresholdImage(src, BINARIZATION_THRESHOLD);
		char buffer[50];
		sprintf(buffer, "Thresholded at %d", BINARIZATION_THRESHOLD);
		imshow(buffer, thresholdedImage);

		Mat labels = computeLabelsMatrixBFS(thresholdedImage);
		Mat labeledImage = computeLabeledImage(labels);
		imshow("Labeled", labeledImage);

		std::vector<Vec3b> colorsOfObjects = computeObjectsColorsBlackBackground(labeledImage);
		//printf("Number of stars: %d\n", colorsOfObjects.size());
		int matchesOnConstellations[NUMBER_OF_CONSTELLATIONS] = { 0 };

		std::vector<DetectionVisual> detectionVisuals;

		omp_lock_t lock;
		omp_init_lock(&lock);

		omp_set_num_threads(NUMBER_OF_CONSTELLATIONS);
		#pragma omp parallel for
		for (int currentConstellationNumber = 0; currentConstellationNumber < NUMBER_OF_CONSTELLATIONS; currentConstellationNumber++)
		{
			if (params_methodsMultDetect[currentConstellationNumber] == -1) continue;

			int currentIterationMatches = 0;

			CenterOfMassInformation centerOfMassInformation = computeCentersOfMass(labeledImage, colorsOfObjects, params_areaThresh[currentConstellationNumber]);
			//imshow("Centers of mass", centerOfMassInformation.image);

			std::vector<Point> inputPoints = centerOfMassInformation.points;
			omp_set_lock(&lock);
			inputCombinations.clear();
			generateCombinationsInput(0, 3, inputPoints);

			std::vector<Triangle> inputTriangles;

			for (int i = 0; i < inputCombinations.size(); i++)
			{
				Triangle triangle;
				triangle.points = inputCombinations[i];
				triangle.computeDistances();
				inputTriangles.push_back(triangle);
			}
			omp_unset_lock(&lock);

			printf("\nInput image\nPoints:%d\nTriangles:%d\n",
				inputPoints.size(),
				inputTriangles.size());

			std::set<std::set<Point, PointCompare>, SetOfPointsCompare> matchingSrcPoints; // All sets of matched points found in source image
			std::vector<Detection> allDetections;

			char fileName[250];
			if (currentConstellationNumber < 10)
				sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation0%d_data.txt", currentConstellationNumber);
			else
				sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation%d_data.txt", currentConstellationNumber);

			FILE* fp;
			fp = fopen(fileName, "r");
			if (fp != NULL)
			{
				std::vector<Point> constellationPoints;
				std::vector<Triangle> constellationTriangles;

				// Read constellation file
				int nbOfPoints, nbOfTriangles;
				fscanf(fp, "%d %d", &nbOfPoints, &nbOfTriangles);

				printf("\nConstellation %d\nPoints: %d\nTriangles: %d\n",
					currentConstellationNumber,
					nbOfPoints,
					nbOfTriangles);

				for (int i = 0; i < nbOfPoints; i++)
				{
					int index, x, y;
					fscanf(fp, "%d %d %d", &index, &x, &y); // index not used while reading, I use push_back so it's in order already
					constellationPoints.push_back(Point(x, y));
				}

				for (int i = 0; i < nbOfTriangles; i++)
				{
					Triangle currentTriangle;
					int x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6;
					double d1, d2, d3;
					fscanf(fp, "%d %d %d %d %lf %d %d %d %d %lf %d %d %d %d %lf",
						&x1, &y1, &x2, &y2, &d1,
						&x3, &y3, &x4, &y4, &d2,
						&x5, &y5, &x6, &y6, &d3);

					// TODO: adding these points MIGHT be redundant 
					currentTriangle.points.push_back(Point(x1, y1));
					currentTriangle.points.push_back(Point(x3, y3));
					currentTriangle.points.push_back(Point(x5, y5));

					Distance distance1(Point(x1, y1), Point(x2, y2), d1);
					Distance distance2(Point(x3, y3), Point(x4, y4), d2);
					Distance distance3(Point(x5, y5), Point(x6, y6), d3);

					currentTriangle.distances.push_back(distance1);
					currentTriangle.distances.push_back(distance2);
					currentTriangle.distances.push_back(distance3);

					constellationTriangles.push_back(currentTriangle);
				}
				fclose(fp);

				// Brute force comparison between source and constellation triangles
				printf("Started triangle computations...\n");
				std::vector<MatchingTrianglePair> allPairs;

				// TODO: Parallelize the comparison
				//omp_set_num_threads(inputTriangles.size());
				//#pragma omp parallel for collapse(2)
				for (int i = 0; i < inputTriangles.size(); i++)
				{
					for (int j = 0; j < constellationTriangles.size(); j++)
					{
						MatchingTrianglePair currentPair(inputTriangles[i], constellationTriangles[j]);
						if (currentPair.difference < params_triangDiffThresh[currentConstellationNumber])
						{
							allPairs.push_back(currentPair);
						}
					}
				}
				printf("Number of matching triangle pairs under %lf = %d\n", params_triangDiffThresh[currentConstellationNumber], allPairs.size());

				//sort(allPairs.begin(), allPairs.end()); // Trivial; all pairs are tested anyways.

				for (int i = 0; i < allPairs.size(); i++)
				{
					MatchingTrianglePair currentPair = allPairs[i];

					Point2f srcTri[3];
					srcTri[0] = Point2f(currentPair.constellationTriangle.points[0].x, currentPair.constellationTriangle.points[0].y);
					srcTri[1] = Point2f(currentPair.constellationTriangle.points[1].x, currentPair.constellationTriangle.points[1].y);
					srcTri[2] = Point2f(currentPair.constellationTriangle.points[2].x, currentPair.constellationTriangle.points[2].y);
					Point2f dstTri[3];
					dstTri[0] = Point2f(currentPair.inputTriangle.points[0].x, currentPair.inputTriangle.points[0].y);
					dstTri[1] = Point2f(currentPair.inputTriangle.points[1].x, currentPair.inputTriangle.points[1].y);
					dstTri[2] = Point2f(currentPair.inputTriangle.points[2].x, currentPair.inputTriangle.points[2].y);

					Mat warp_mat = getAffineTransform(srcTri, dstTri);

					// Warp constellation
					std::vector<Point2f> warpedConstellationPoints;
					for (int j = 0; j < constellationPoints.size(); j++)
					{
						Point2f nonWarpedPoint(constellationPoints[j].x, constellationPoints[j].y);
						Point2f warpedPoint;
						warpedPoint.x = warp_mat.at<double>(0, 0) * nonWarpedPoint.x + warp_mat.at<double>(0, 1) * nonWarpedPoint.y + warp_mat.at<double>(0, 2);
						warpedPoint.y = warp_mat.at<double>(1, 0) * nonWarpedPoint.x + warp_mat.at<double>(1, 1) * nonWarpedPoint.y + warp_mat.at<double>(1, 2);
						warpedConstellationPoints.push_back(warpedPoint);
					}

					std::set<Point, PointCompare> srcMatchingPointsCurrent; // All points matched in the source image for current iteration
					int srcMatchingPointsCurrentCounter; // Number of times all points matched in the source image for current iteration

					std::vector<int> matchedIndexes;
					//int matchingStarsNb = 0;
					double meanLuminosity = 0.0f;
					// Compute number of matching stars between warped constellation and source image
					for (int j = 0; j < warpedConstellationPoints.size(); j++)
					{
						Point2f currentPoint = warpedConstellationPoints[j];
						int yOfPoint = (int)(currentPoint.y + 0.5f); // Float -> int : add 0.5 then (int)
						int xOfPoint = (int)(currentPoint.x + 0.5f);
						if (yOfPoint > centerOfMassInformation.image.rows || yOfPoint < 0) break; // Check out of bounds
						if (xOfPoint > centerOfMassInformation.image.cols || xOfPoint < 0) break;

						bool brokeInnerLoop = false;
						for (int index0 = 0; index0 <= params_posVar[currentConstellationNumber]; index0++)
						{
							for (int index1 = (-1 * index0); index1 <= index0; index1++)
							{
								for (int index2 = (-1 * index0); index2 <= index0; index2++)
								{
									if (yOfPoint + index1 > (centerOfMassInformation.image.rows - 1) // Check out of bounds
										|| yOfPoint + index1 < 0
										|| xOfPoint + index2 >(centerOfMassInformation.image.cols - 1)
										|| xOfPoint + index2 < 0
										)
									{
										brokeInnerLoop = true;
										break;
									}
									if (centerOfMassInformation.image.at<uchar>(yOfPoint + index1, xOfPoint + index2) == WHITE)
									{
										srcMatchingPointsCurrent.insert(Point(xOfPoint + index2, yOfPoint + index1));
										meanLuminosity += centerOfMassInformation.areasMat.at<int>(yOfPoint + index1, xOfPoint + index2);
										//matchingStarsNb++;
										matchedIndexes.push_back(j);
										brokeInnerLoop = true;
										break;
									}
								}

								if (brokeInnerLoop) break;
							}

							if (brokeInnerLoop) break;
						}

					}

					if (srcMatchingPointsCurrent.size() == constellationPoints.size())
					{
						meanLuminosity /= srcMatchingPointsCurrent.size();

						//sort(srcMatchingPointsCurrent.begin(), srcMatchingPointsCurrent.end(), MyPointSorter);
						//std::set<Point, PointCompare>::iterator iterator;
						bool setAlreadyFound = false;

						if (matchingSrcPoints.find(srcMatchingPointsCurrent) != matchingSrcPoints.end())
						{
							//srcMatchingPointsCurrentCounter = matchingSrcPointsCounter[j];
							//int index = matchingSrcPoints.find(srcMatchingPointsCurrent);
							//matchingSrcPointsCounter[]++;
							setAlreadyFound = true;
							break;
						}



						if (!setAlreadyFound && meanLuminosity > params_lum[currentConstellationNumber])
						{
							currentIterationMatches++;

							matchingSrcPoints.insert(srcMatchingPointsCurrent);
							Detection detection(srcMatchingPointsCurrent, currentPair, meanLuminosity, params_methodsMultDetect[currentConstellationNumber]);
							allDetections.push_back(detection);
						}
					}
				}
			}


			if (allDetections.size() > 0)
			{
				sort(allDetections.begin(), allDetections.end());

				for (int j = 0; j < allDetections.size(); j++)
				{
					printf("Detection %d Order=%d Lum=%lf Diff=%lf\n", j, allDetections[j].sortingOrder, allDetections[j].luminosity, allDetections[j].difference);
				}

				Detection selectedDetection = allDetections[0];

				printf("Number of detections: %d\n", allDetections.size());
				if (selectedDetection.sortingOrder == METHOD_GREATER_LUMINOSITY)
					printf("Showing the detection with Greatest Luminosity.\n");
				else
					printf("Showing the detection with Lowest Difference between triangles\n");

				printf("Nb. of points:%d\nDifference:%lf\nLuminosity:%lf\n",
					selectedDetection.srcPoints.size(),
					selectedDetection.difference,
					selectedDetection.luminosity
				);

				Mat inputHighlightedImage = imread(fname, IMREAD_COLOR);
				char constellationImageFileName[250];
				if (currentConstellationNumber < 10)
					sprintf(constellationImageFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations//png//constellation0%d.png", currentConstellationNumber);
				else
					sprintf(constellationImageFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations//png//constellation%d.png", currentConstellationNumber);
				Mat constellationHighlightedImage = imread(constellationImageFileName, IMREAD_COLOR);

				// Drawing matching triangles
				line(inputHighlightedImage, selectedDetection.trianglePair.inputTriangle.points[0], selectedDetection.trianglePair.inputTriangle.points[1], Vec3b(255, 0, 0), 1);
				line(inputHighlightedImage, selectedDetection.trianglePair.inputTriangle.points[0], selectedDetection.trianglePair.inputTriangle.points[2], Vec3b(0, 255, 0), 1);
				line(inputHighlightedImage, selectedDetection.trianglePair.inputTriangle.points[1], selectedDetection.trianglePair.inputTriangle.points[2], Vec3b(0, 0, 255), 1);

				line(constellationHighlightedImage, selectedDetection.trianglePair.constellationTriangle.points[0], selectedDetection.trianglePair.constellationTriangle.points[1], Vec3b(255, 0, 0), 1);
				line(constellationHighlightedImage, selectedDetection.trianglePair.constellationTriangle.points[0], selectedDetection.trianglePair.constellationTriangle.points[2], Vec3b(0, 255, 0), 1);
				line(constellationHighlightedImage, selectedDetection.trianglePair.constellationTriangle.points[1], selectedDetection.trianglePair.constellationTriangle.points[2], Vec3b(0, 0, 255), 1);

				// Drawing points of warped constellation on input image
				for (Point point : selectedDetection.srcPoints)
				{
					circle(inputHighlightedImage, point, 2, Vec3b(0, 0, 255), 3);
				}

				DetectionVisual detectionVisual(inputHighlightedImage, constellationHighlightedImage, currentConstellationNumber);

				omp_set_lock(&lock);
				detectionVisuals.push_back(detectionVisual);
				omp_unset_lock(&lock);
			}


			if (currentIterationMatches > 0)
			{
				omp_set_lock(&lock);
				matchesOnConstellations[currentConstellationNumber] = currentIterationMatches;
				omp_unset_lock(&lock);
			}
		}

		for (int i = 0; i < detectionVisuals.size(); i++)
		{
			char bufferInputImage[100];
			sprintf(bufferInputImage, "inputHighlighted %d", detectionVisuals[i].number);
			imshow(bufferInputImage, detectionVisuals[i].inputImage);

			char bufferConstellationImage[100];
			sprintf(bufferConstellationImage, "constellationHighlighted %d", detectionVisuals[i].number);
			imshow(bufferConstellationImage, detectionVisuals[i].constellationImage);
		}

		endTime = clock();
		double timeTaken = double(endTime - startTime) / double(CLOCKS_PER_SEC);


		printf("\n\n\n ----- Stats ----- \n\n\n");
		printf("Time: %lf seconds.\n", timeTaken);
		int correctDetections = 0;
		int incorrectDetections = 0;
		for (int i = 0; i < NUMBER_OF_CONSTELLATIONS; i++)
		{
			if (matchesOnConstellations[i] > 0)
			{
				if (i == SELECTION_CONSTELLATION)
				{
					correctDetections++;
				}
				else
				{
					incorrectDetections++;
				}

				if (matchesOnConstellations[i] == 1)
				{
					printf("Constellation %d: %d match.\n", i, matchesOnConstellations[i]);
				}
				else
				{
					printf("Constellation %d: %d match.\n", i, matchesOnConstellations[i]);
				}
			}
		}

		printf("\nCorrect detections: %d\nIncorrect detections: %d\n", correctDetections, incorrectDetections);
		waitKey(0);
	}
}



// Abandoned because running it resulted in a bluescreen.
void thesis_testOnRandomSelectionsFromFile_withAllConstellations_withoutLines()
{
	char fname[MAX_PATH];
	Mat src;
	const int BINARIZATION_THRESHOLD = 25;
	const int NUMBER_OF_CONSTELLATIONS = 89;

	double params_triangDiffThresh[NUMBER_OF_CONSTELLATIONS];
	int params_areaThresh[NUMBER_OF_CONSTELLATIONS];
	int params_posVar[NUMBER_OF_CONSTELLATIONS];
	double params_lum[NUMBER_OF_CONSTELLATIONS];
	int params_methodsMultDetect[NUMBER_OF_CONSTELLATIONS];
	const int METHOD_GREATER_LUMINOSITY = 0;
	const int METHOD_LOWEST_DIFFERENCE = 1;
	const int METHOD_ANY = 2;

	char detectionParamsFileName[250];
	sprintf(detectionParamsFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//detection_parameters_constellations.txt");
	FILE* detectionParamsFp = fopen(detectionParamsFileName, "r");
	if (detectionParamsFp == NULL)
	{
		printf("Could not open detection params file.\n");
	}
	else
	{
		for (int i = 0; i < NUMBER_OF_CONSTELLATIONS; i++)
		{
			int constellationNb = -1;
			fscanf(detectionParamsFp, "%d,%lf,%d,%d,%lf,%d",
				&constellationNb,
				&params_triangDiffThresh[i],
				&params_areaThresh[i],
				&params_posVar[i],
				&params_lum[i],
				&params_methodsMultDetect[i]);
		}

		FILE* randomSelectionsFp = fopen("D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//evaluation_data//evaluation_random_selections.txt", "r");
		if (randomSelectionsFp == NULL)
		{
			printf("Could not open random images selection file.\n");
		}
		else
		{
			int randomConstellationIndex, randomImageIndex;
			int randomImageIteration = -1;
			
			FILE* evaluationOutputFP = fopen("D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//evaluation_data//evaluation_random_selections_output.txt", "w");
			if (evaluationOutputFP == NULL) printf("Could not open evaluation output file.\n");

			while (fscanf(randomSelectionsFp, "%d,%d", &randomConstellationIndex, &randomImageIndex) == 2)
			{
				randomImageIteration++;

				clock_t startTime, endTime;

				sprintf(fname, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//evaluation_images//%d//%d.png", randomConstellationIndex, randomImageIndex);

				src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Source grayscale", src);

				startTime = clock();

				// TODO: Consider adaptive thresholding 
				Mat thresholdedImage = thresholdImage(src, BINARIZATION_THRESHOLD);
				char buffer[50];
				sprintf(buffer, "Thresholded at %d", BINARIZATION_THRESHOLD);
				imshow(buffer, thresholdedImage);

				Mat labels = computeLabelsMatrixBFS(thresholdedImage);
				Mat labeledImage = computeLabeledImage(labels);
				imshow("Labeled", labeledImage);

				std::vector<Vec3b> colorsOfObjects = computeObjectsColorsBlackBackground(labeledImage);
				//printf("Number of stars: %d\n", colorsOfObjects.size());
				int matchesOnConstellations[NUMBER_OF_CONSTELLATIONS] = { 0 };

				std::vector<DetectionVisual> detectionVisuals;

				omp_lock_t lock;
				omp_init_lock(&lock);

				omp_set_num_threads(NUMBER_OF_CONSTELLATIONS);
				#pragma omp parallel for
				for (int currentConstellationNumber = 0; currentConstellationNumber < NUMBER_OF_CONSTELLATIONS; currentConstellationNumber++)
				{
					if (params_methodsMultDetect[currentConstellationNumber] == -1) continue;

					int currentIterationMatches = 0;

					CenterOfMassInformation centerOfMassInformation = computeCentersOfMass(labeledImage, colorsOfObjects, params_areaThresh[currentConstellationNumber]);
					//imshow("Centers of mass", centerOfMassInformation.image);

					std::vector<Point> inputPoints = centerOfMassInformation.points;
					omp_set_lock(&lock);
					inputCombinations.clear();
					generateCombinationsInput(0, 3, inputPoints);

					std::vector<Triangle> inputTriangles;

					for (int i = 0; i < inputCombinations.size(); i++)
					{
						Triangle triangle;
						triangle.points = inputCombinations[i];
						triangle.computeDistances();
						inputTriangles.push_back(triangle);
					}
					omp_unset_lock(&lock);

					printf("\nInput image\nPoints:%d\nTriangles:%d\n",
						inputPoints.size(),
						inputTriangles.size());

					std::set<std::set<Point, PointCompare>, SetOfPointsCompare> matchingSrcPoints; // All sets of matched points found in source image
					std::vector<Detection> allDetections;

					char fileName[250];
					if (currentConstellationNumber < 10)
						sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation0%d_data.txt", currentConstellationNumber);
					else
						sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation%d_data.txt", currentConstellationNumber);

					FILE* fp;
					fp = fopen(fileName, "r");
					if (fp != NULL)
					{
						std::vector<Point> constellationPoints;
						std::vector<Triangle> constellationTriangles;

						// Read constellation file
						int nbOfPoints, nbOfTriangles;
						fscanf(fp, "%d %d", &nbOfPoints, &nbOfTriangles);

						printf("\nConstellation %d\nPoints: %d\nTriangles: %d\n",
							currentConstellationNumber,
							nbOfPoints,
							nbOfTriangles);

						for (int i = 0; i < nbOfPoints; i++)
						{
							int index, x, y;
							fscanf(fp, "%d %d %d", &index, &x, &y); // index not used while reading, I use push_back so it's in order already
							constellationPoints.push_back(Point(x, y));
						}

						for (int i = 0; i < nbOfTriangles; i++)
						{
							Triangle currentTriangle;
							int x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6;
							double d1, d2, d3;
							fscanf(fp, "%d %d %d %d %lf %d %d %d %d %lf %d %d %d %d %lf",
								&x1, &y1, &x2, &y2, &d1,
								&x3, &y3, &x4, &y4, &d2,
								&x5, &y5, &x6, &y6, &d3);

							// TODO: adding these points MIGHT be redundant 
							currentTriangle.points.push_back(Point(x1, y1));
							currentTriangle.points.push_back(Point(x3, y3));
							currentTriangle.points.push_back(Point(x5, y5));

							Distance distance1(Point(x1, y1), Point(x2, y2), d1);
							Distance distance2(Point(x3, y3), Point(x4, y4), d2);
							Distance distance3(Point(x5, y5), Point(x6, y6), d3);

							currentTriangle.distances.push_back(distance1);
							currentTriangle.distances.push_back(distance2);
							currentTriangle.distances.push_back(distance3);

							constellationTriangles.push_back(currentTriangle);
						}
						fclose(fp);

						// Brute force comparison between source and constellation triangles
						printf("Started triangle computations...\n");
						std::vector<MatchingTrianglePair> allPairs;

						// TODO: Parallelize the comparison
						//omp_set_num_threads(inputTriangles.size());
						//#pragma omp parallel for collapse(2)
						for (int i = 0; i < inputTriangles.size(); i++)
						{
							for (int j = 0; j < constellationTriangles.size(); j++)
							{
								MatchingTrianglePair currentPair(inputTriangles[i], constellationTriangles[j]);
								if (currentPair.difference < params_triangDiffThresh[currentConstellationNumber])
								{
									allPairs.push_back(currentPair);
								}
							}
						}
						printf("Number of matching triangle pairs under %lf = %d\n", params_triangDiffThresh[currentConstellationNumber], allPairs.size());

						//sort(allPairs.begin(), allPairs.end()); // Trivial; all pairs are tested anyways.

						for (int i = 0; i < allPairs.size(); i++)
						{
							MatchingTrianglePair currentPair = allPairs[i];

							Point2f srcTri[3];
							srcTri[0] = Point2f(currentPair.constellationTriangle.points[0].x, currentPair.constellationTriangle.points[0].y);
							srcTri[1] = Point2f(currentPair.constellationTriangle.points[1].x, currentPair.constellationTriangle.points[1].y);
							srcTri[2] = Point2f(currentPair.constellationTriangle.points[2].x, currentPair.constellationTriangle.points[2].y);
							Point2f dstTri[3];
							dstTri[0] = Point2f(currentPair.inputTriangle.points[0].x, currentPair.inputTriangle.points[0].y);
							dstTri[1] = Point2f(currentPair.inputTriangle.points[1].x, currentPair.inputTriangle.points[1].y);
							dstTri[2] = Point2f(currentPair.inputTriangle.points[2].x, currentPair.inputTriangle.points[2].y);

							Mat warp_mat = getAffineTransform(srcTri, dstTri);

							// Warp constellation
							std::vector<Point2f> warpedConstellationPoints;
							for (int j = 0; j < constellationPoints.size(); j++)
							{
								Point2f nonWarpedPoint(constellationPoints[j].x, constellationPoints[j].y);
								Point2f warpedPoint;
								warpedPoint.x = warp_mat.at<double>(0, 0) * nonWarpedPoint.x + warp_mat.at<double>(0, 1) * nonWarpedPoint.y + warp_mat.at<double>(0, 2);
								warpedPoint.y = warp_mat.at<double>(1, 0) * nonWarpedPoint.x + warp_mat.at<double>(1, 1) * nonWarpedPoint.y + warp_mat.at<double>(1, 2);
								warpedConstellationPoints.push_back(warpedPoint);
							}

							std::set<Point, PointCompare> srcMatchingPointsCurrent; // All points matched in the source image for current iteration
							int srcMatchingPointsCurrentCounter; // Number of times all points matched in the source image for current iteration

							std::vector<int> matchedIndexes;
							//int matchingStarsNb = 0;
							double meanLuminosity = 0.0f;
							// Compute number of matching stars between warped constellation and source image
							for (int j = 0; j < warpedConstellationPoints.size(); j++)
							{
								Point2f currentPoint = warpedConstellationPoints[j];
								int yOfPoint = (int)(currentPoint.y + 0.5f); // Float -> int : add 0.5 then (int)
								int xOfPoint = (int)(currentPoint.x + 0.5f);
								if (yOfPoint > centerOfMassInformation.image.rows || yOfPoint < 0) break; // Check out of bounds
								if (xOfPoint > centerOfMassInformation.image.cols || xOfPoint < 0) break;

								bool brokeInnerLoop = false;
								for (int index0 = 0; index0 <= params_posVar[currentConstellationNumber]; index0++)
								{
									for (int index1 = (-1 * index0); index1 <= index0; index1++)
									{
										for (int index2 = (-1 * index0); index2 <= index0; index2++)
										{
											if (yOfPoint + index1 > (centerOfMassInformation.image.rows - 1) // Check out of bounds
												|| yOfPoint + index1 < 0
												|| xOfPoint + index2 >(centerOfMassInformation.image.cols - 1)
												|| xOfPoint + index2 < 0
												)
											{
												brokeInnerLoop = true;
												break;
											}
											if (centerOfMassInformation.image.at<uchar>(yOfPoint + index1, xOfPoint + index2) == WHITE)
											{
												srcMatchingPointsCurrent.insert(Point(xOfPoint + index2, yOfPoint + index1));
												meanLuminosity += centerOfMassInformation.areasMat.at<int>(yOfPoint + index1, xOfPoint + index2);
												//matchingStarsNb++;
												matchedIndexes.push_back(j);
												brokeInnerLoop = true;
												break;
											}
										}

										if (brokeInnerLoop) break;
									}

									if (brokeInnerLoop) break;
								}

							}

							if (srcMatchingPointsCurrent.size() == constellationPoints.size())
							{
								meanLuminosity /= srcMatchingPointsCurrent.size();

								//sort(srcMatchingPointsCurrent.begin(), srcMatchingPointsCurrent.end(), MyPointSorter);
								//std::set<Point, PointCompare>::iterator iterator;
								bool setAlreadyFound = false;

								if (matchingSrcPoints.find(srcMatchingPointsCurrent) != matchingSrcPoints.end())
								{
									//srcMatchingPointsCurrentCounter = matchingSrcPointsCounter[j];
									//int index = matchingSrcPoints.find(srcMatchingPointsCurrent);
									//matchingSrcPointsCounter[]++;
									setAlreadyFound = true;
									break;
								}


								if (!setAlreadyFound && meanLuminosity > params_lum[currentConstellationNumber])
								{
									currentIterationMatches++;

									matchingSrcPoints.insert(srcMatchingPointsCurrent);
									Detection detection(srcMatchingPointsCurrent, currentPair, meanLuminosity, params_methodsMultDetect[currentConstellationNumber]);
									allDetections.push_back(detection);
								}
							}
						}
					}


					if (allDetections.size() > 0 && currentConstellationNumber == randomConstellationIndex)
					{
						sort(allDetections.begin(), allDetections.end());

						for (int j = 0; j < allDetections.size(); j++)
						{
							printf("Detection %d Order=%d Lum=%lf Diff=%lf\n", j, allDetections[j].sortingOrder, allDetections[j].luminosity, allDetections[j].difference);
						}

						Detection selectedDetection = allDetections[0];

						printf("Number of detections: %d\n", allDetections.size());
						if (selectedDetection.sortingOrder == METHOD_GREATER_LUMINOSITY)
							printf("Showing the detection with Greatest Luminosity.\n");
						else
							printf("Showing the detection with Lowest Difference between triangles\n");

						printf("Nb. of points:%d\nDifference:%lf\nLuminosity:%lf\n",
							selectedDetection.srcPoints.size(),
							selectedDetection.difference,
							selectedDetection.luminosity
						);

						Mat inputHighlightedImage = imread(fname, IMREAD_COLOR);
						char constellationImageFileName[250];
						if (currentConstellationNumber < 10)
							sprintf(constellationImageFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations//png//constellation0%d.png", currentConstellationNumber);
						else
							sprintf(constellationImageFileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations//png//constellation%d.png", currentConstellationNumber);
						Mat constellationHighlightedImage = imread(constellationImageFileName, IMREAD_COLOR);

						// Drawing matching triangles
						line(inputHighlightedImage, selectedDetection.trianglePair.inputTriangle.points[0], selectedDetection.trianglePair.inputTriangle.points[1], Vec3b(255, 0, 0), 1);
						line(inputHighlightedImage, selectedDetection.trianglePair.inputTriangle.points[0], selectedDetection.trianglePair.inputTriangle.points[2], Vec3b(0, 255, 0), 1);
						line(inputHighlightedImage, selectedDetection.trianglePair.inputTriangle.points[1], selectedDetection.trianglePair.inputTriangle.points[2], Vec3b(0, 0, 255), 1);

						line(constellationHighlightedImage, selectedDetection.trianglePair.constellationTriangle.points[0], selectedDetection.trianglePair.constellationTriangle.points[1], Vec3b(255, 0, 0), 1);
						line(constellationHighlightedImage, selectedDetection.trianglePair.constellationTriangle.points[0], selectedDetection.trianglePair.constellationTriangle.points[2], Vec3b(0, 255, 0), 1);
						line(constellationHighlightedImage, selectedDetection.trianglePair.constellationTriangle.points[1], selectedDetection.trianglePair.constellationTriangle.points[2], Vec3b(0, 0, 255), 1);

						// Drawing points of warped constellation on input image
						for (Point point : selectedDetection.srcPoints)
						{
							circle(inputHighlightedImage, point, 2, Vec3b(0, 0, 255), 3);
						}

						DetectionVisual detectionVisual(inputHighlightedImage, constellationHighlightedImage, currentConstellationNumber);

						omp_set_lock(&lock);
						detectionVisuals.push_back(detectionVisual);
						omp_unset_lock(&lock);
					}


					if (currentIterationMatches > 0)
					{
						omp_set_lock(&lock);
						matchesOnConstellations[currentConstellationNumber] = currentIterationMatches;
						omp_unset_lock(&lock);
					}
				}

				for (int i = 0; i < detectionVisuals.size(); i++)
				{
					char bufferInputImage[100];
					sprintf(bufferInputImage, "img %d inputHighlighted %d", randomImageIteration, detectionVisuals[i].number);
					imshow(bufferInputImage, detectionVisuals[i].inputImage);

					char bufferConstellationImage[100];
					sprintf(bufferConstellationImage, "img %d constellationHighlighted %d", randomImageIteration, detectionVisuals[i].number);
					imshow(bufferConstellationImage, detectionVisuals[i].constellationImage);
				}

				endTime = clock();
				double timeTaken = double(endTime - startTime) / double(CLOCKS_PER_SEC);


				printf("\n\n\n ----- Stats ----- \n\n\n");
				printf("Time: %lf seconds.\n", timeTaken);
				int falseConstellationsDetected = 0;
				int correctConstellationsDetected = 0;
				for (int i = 0; i < NUMBER_OF_CONSTELLATIONS; i++)
				{
					if (matchesOnConstellations[i] > 0)
					{
						if (i == randomConstellationIndex)
						{
							correctConstellationsDetected++;
						}
						else
						{
							falseConstellationsDetected++;
						}

						if (matchesOnConstellations[i] == 1)
						{
							printf("Constellation %d: %d match.\n", i, matchesOnConstellations[i]);
						}
						else
						{
							printf("Constellation %d: %d matches.\n", i, matchesOnConstellations[i]);
						}
					}
				}

				fprintf(evaluationOutputFP, "%d %d %d %d %lf\n", randomConstellationIndex, randomImageIndex, correctConstellationsDetected, falseConstellationsDetected, timeTaken);

				//waitKey(0);
			}


		}

		
	}
}


Mat dilateNTimesWithParams(Mat src, int n, int objectValue, int backgroundValue)
{
	int height = src.rows;
	int width = src.cols;
	Mat newSource = src.clone();

	Mat dst(height, width, CV_8UC1, RGB(backgroundValue, backgroundValue, backgroundValue));
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	const int OBJECT_PIXEL = objectValue;

	for (int nTimes = 0; nTimes < n; nTimes++)
	{
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (newSource.at<uchar>(i, j) == OBJECT_PIXEL)
				{
					dst.at<uchar>(i, j) = OBJECT_PIXEL;
					for (int k = 0; k < 8; k++)
					{
						dst.at<uchar>(i + di[k], j + dj[k]) = OBJECT_PIXEL;
					}
				}
			}
		}

		newSource = dst.clone();
	}

	return dst;
}

Mat erodeNTimesWithParams(Mat src, int n, int objectValue, int backgroundValue)
{
	int height = src.rows;
	int width = src.cols;
	Mat newSource = src.clone();

	Mat dst(height, width, CV_8UC1, RGB(backgroundValue, backgroundValue, backgroundValue));
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	const int OBJECT_PIXEL = objectValue;
	const int BACKGROUND_PIXEL = backgroundValue;
	bool allObjectPixels = false;
	bool oneBGPixelFound = false;

	for (int nTimes = 0; nTimes < n; nTimes++)
	{
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (newSource.at<uchar>(i, j) == OBJECT_PIXEL)
				{
					for (int k = 0; k < 8; k++)
					{
						if (newSource.at<uchar>(i + di[k], j + dj[k]) == BACKGROUND_PIXEL)
						{
							oneBGPixelFound = true;
							break;
						}
					}

					if (!oneBGPixelFound)
					{
						dst.at<uchar>(i, j) = OBJECT_PIXEL;
					}
					else
					{
						dst.at<uchar>(i, j) = BACKGROUND_PIXEL;
					}
				}
				else
				{
					dst.at<uchar>(i, j) = BACKGROUND_PIXEL;
				}

				oneBGPixelFound = false;
			}
		}


		newSource = dst.clone();
	}

	return dst;
}

Mat filterForStars(Mat src)
{
	const int BLUE_BRIGHT_VALUE = 19;
	const int GREEN_BRIGHT_VALUE = 71;
	const int RED_BRIGHT_VALUE = 179;

	const int BLUE_VARIATION = 15;
	const int GREEN_VARIATION = 20;
	const int RED_VARIATION = 70;

	Mat filtered(src.rows, src.cols, CV_8UC1, Scalar(BLACK, BLACK, BLACK));
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			Vec3b currentPixel = src.at<Vec3b>(i, j);

			if (currentPixel[0] > BLUE_BRIGHT_VALUE - BLUE_VARIATION
				&& currentPixel[0] < BLUE_BRIGHT_VALUE + BLUE_VARIATION
				&& currentPixel[1] > GREEN_BRIGHT_VALUE - GREEN_VARIATION
				&& currentPixel[1] < GREEN_BRIGHT_VALUE + GREEN_VARIATION
				&& currentPixel[2] > RED_BRIGHT_VALUE - RED_VARIATION
				&& currentPixel[2] < RED_BRIGHT_VALUE + RED_VARIATION)
			{
				filtered.at<uchar>(i, j) = WHITE;
			}
		}
	}
	return filtered;
}

void writeConstellationInfoInFile(std::vector<Point> points, std::vector<Triangle> triangles, int firstNb, int secondNb)
{
	FILE* fp;
	char fileName[250];
	sprintf(fileName, "D://Facultate//AN III//Semester 2//Image Processing//OpenCVApplication-VS2017_OCV340_basic//thesis_data//constellations_processing_data//constellation%d%d_data.txt", firstNb, secondNb);
	fp = fopen(fileName, "w+");
	if (fp == NULL)
	{
		printf("Error opening file.\n");
	}
	else
	{
		int numberOfPoints = points.size();
		int numberOfTriangles = triangles.size();
		
		fprintf(fp, "%d %d", numberOfPoints, numberOfTriangles);
		
		for (int i = 0; i < points.size(); i++)
		{
			fprintf(fp, "\n%d %d %d ", i, points[i].x, points[i].y);
		}

		for (int i = 0; i < triangles.size(); i++)
		{
			fprintf(fp, "\n");
			for (int j = 0; j < 3; j++)
			{
				fprintf(fp, "%d %d %d %d %lf ",
					triangles[i].distances[j].p1.x, triangles[i].distances[j].p1.y,
					triangles[i].distances[j].p2.x, triangles[i].distances[j].p2.y,
					triangles[i].distances[j].value);
			}
		}

		fclose(fp);
	}
}



void testing()
{
	int target_thread_num = 100;
	omp_set_num_threads(target_thread_num);

	#pragma omp parallel for
	for (int i = 1; i < 100; ++i)
	{
		printf("%d\n", omp_get_thread_num());
	}


	//std::set<Point, PointCompare> testSet; // All sets of matched points found in source image
	//testSet.insert(Point(0, 0));
	//testSet.insert(Point(0, 1));
	//
	//std::set<Point, PointCompare> testSet2; // All sets of matched points found in source image
	//testSet2.insert(Point(0, 1));
	//testSet2.insert(Point(0, 0));
	//


	//std::set<std::set<Point, PointCompare>, SetOfPointsCompare> testSet3; // All sets of matched points found in source image
	//testSet3.insert(testSet);
	//testSet3.insert(testSet2);

	//std::set<std::set<Point, PointCompare>, SetOfPointsCompare>::iterator setIter;
	//setIter = testSet3.find(testSet2);

	//
	//

	//printf("abcd");
	int a;
	scanf("%d\n", &a);

	waitKey(0);
}

void testConstellationPreprocessingOnSelectedImage()
{
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_COLOR);
		imshow("Source", src);

		Mat starsOnly = filterForStars(src);
		imshow("Stars only", starsOnly);

		//Mat erosionDilation = erodeNTimesWithParams(starsOnly, 2, WHITE, BLACK);
		//erosionDilation = dilateNTimesWithParams(erosionDilation, 1, WHITE, BLACK);
		//erosionDilation = erodeNTimesWithParams(starsOnly, 3, WHITE, BLACK);
		//erosionDilation = dilateNTimesWithParams(erosionDilation, 3, WHITE, BLACK);
		//imshow("Erosion & Dilation", erosionDilation);
		Mat erosionDilation = starsOnly;

		Mat labels = computeLabelsMatrixBFS(erosionDilation);

		Mat labeledImage = computeLabeledImage(labels);
		imshow("Labeled", labeledImage);
		std::vector<Vec3b> colorsOfObjects = computeObjectsColorsBlackBackground(labeledImage);
		printf("Number of stars: %d\n", colorsOfObjects.size());

		CenterOfMassInformation centerOfMassInformation = computeCentersOfMass(labeledImage, colorsOfObjects, 0);
		Mat centerOfMassImage = centerOfMassInformation.image;
		imshow("Centers of mass", centerOfMassInformation.image);

		/*
		printf("Centers of mass:");
		for (int i = 0; i < centerOfMassInformation.points.size(); i++)
		{
			printf("\nx=%d y=%d", centerOfMassInformation.points[i].x, centerOfMassInformation.points[i].y);
		}
		*/

		constellationCombinations.clear();
		generateCombinationsConstellation(0, 3, centerOfMassInformation.points);

		
		printf("Nb of combinations: %d\n", constellationCombinations.size());
		
		std::vector<Triangle> triangles;
		for (int i = 0; i < constellationCombinations.size(); i++)
		{
			Triangle triangle;
			triangle.points = constellationCombinations[i];
			triangle.computeDistances();
			triangles.push_back(triangle);
		}
		
		int firstNb = fname[130] - '0'; // Get "01" from D://...//constellation01.png
		int secondNb = fname[131] - '0';

		writeConstellationInfoInFile(centerOfMassInformation.points, triangles, firstNb, secondNb);
			
	}
}

void preprocessConstellations()
{
	char fname[MAX_PATH];
	Mat src;

	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_COLOR);
		imshow("Source", src);

		Mat starsOnly = filterForStars(src);
		imshow("Stars only", starsOnly);

		//Mat erosionDilation = erodeNTimesWithParams(starsOnly, 1, WHITE, BLACK);
		//erosionDilation = dilateNTimesWithParams(erosionDilation, 1, WHITE, BLACK);
		//imshow("Erosion & Dilation", erosionDilation);
		Mat erosionDilation = starsOnly;

		Mat labels = computeLabelsMatrixBFS(erosionDilation);

		Mat labeledImage = computeLabeledImage(labels);
		imshow("Labeled", labeledImage);
		std::vector<Vec3b> colorsOfObjects = computeObjectsColorsBlackBackground(labeledImage);
		printf("Number of stars: %d\n", colorsOfObjects.size());

		CenterOfMassInformation centerOfMassInformation = computeCentersOfMass(labeledImage, colorsOfObjects, 0);
		imshow("Centers of mass", centerOfMassInformation.image);

		/*
		printf("Centers of mass:");
		for (int i = 0; i < centerOfMassInformation.points.size(); i++)
		{
			printf("\nx=%d y=%d", centerOfMassInformation.points[i].x, centerOfMassInformation.points[i].y);
		}
		*/

		generateCombinationsConstellation(0, 3, centerOfMassInformation.points);

		printf("Nb of combinations: %d\n", constellationCombinations.size());

		std::vector<Triangle> triangles;
		for (int i = 0; i < constellationCombinations.size(); i++)
		{
			Triangle triangle;
			triangle.points = constellationCombinations[i];
			triangle.computeDistances();
			triangles.push_back(triangle);
		}


		printf("Nb of triangles = %d", triangles.size());
		for (int i = 0; i < triangles.size(); i++)
		{
			printf("Triangle #%d\n", i);
			printf("Distances: %lf %lf %lf\n", triangles[i].distances[0], triangles[i].distances[1], triangles[i].distances[2]);
		}

	}
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf("===== Image Processing=====\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Change gray levels\n");
		printf(" 11 - Four colored squares\n");
		printf(" 12 - Inverse matrix\n");
		printf(" 13 - RGB24 split channels\n");
		printf(" 14 - Color to grayscale\n");
		printf(" 15 - Grayscale to Black & White\n");
		printf(" 16 - RGB to HSV\n");
		printf(" 17 - Grayscale Image Histogram\n");
		printf(" 18 - Multilevel thresholding\n");
		printf(" 19 - Floyd-Steinberg dithering\n");
		printf(" 20 - Geometric properties of binary objects\n");
		printf(" 21 - Label binary images w/ BFS\n");
		printf(" 22 - Label binary images w/ Two-pass \n");
		printf(" 23 - Border tracing algorithm w/ Chain Codes Extraction\n");
		printf(" 24 - Reconstruct border of image from file\n");
		printf(" 25 - Dilate\n");
		printf(" 26 - Erode\n");
		printf(" 27 - Open\n");
		printf(" 28 - Close\n");
		printf(" 29 - Boundary extraction\n");
		printf(" 30 - Region filling\n");
		printf(" 31 - Statistical Properties\n");
		printf(" 32 - General Filter\n");
		printf(" 33 - Frequency Domain\n");
		printf(" 34 - Median Filter\n");
		printf(" 35 - 2D Gaussian Filter\n");
		printf(" 36 - 2 1D Gaussian Filters\n");
		printf(" 37 - Canny Edge Detection\n");
		printf(" 100 - Project\n");
		printf(" 111 - Testing\n");
		printf(" 122 - Thesis Evaluation\n");
		printf(" 123 - Thesis\n");
		printf(" 124 - Test 1 Image Constellation Preprocessing\n");
		printf(" 125 - Preprocess Constellation\n");
		printf(" 133 - Constellation Detection on singular image without lines\n");
		printf("\n\n\n===== Pattern Recognition Systems=====\n");
		printf("-Lab 1\n");
		printf("	38 - Least Mean Squares - Model 1\n");
		printf("	39 - Least Mean Squares - Model 2\n");
		printf("-Lab 2\n");
		printf("	40 - Ransac Line\n");
		printf("-Lab 3\n");
		printf("	41 - Hough Transform Line Fitting\n");
		printf("-Lab 4\n");
		printf("	42 - Distance Transform & Pattern Matching\n");
		printf("-Lab 5\n");
		printf("	43 - Statistical Data Analysis\n");
		printf("-Lab 6\n");
		printf("	44 - Principal Component Analysis\n");
		printf("-Lab 7\n");
		printf("	45 - K-means Clustering\n");
		printf("-Lab 8\n");
		printf("	46 - K-nearest Neighbours Classifier\n");
		printf("-Lab 9\n");
		printf("	47 - Naive Bayesian Classifier - Digit Recognition\n");
		printf("-Lab 10\n");
		printf("	48 - Linear Classifiers and the Perceptron Algorithm\n");
		printf("-Lab 11\n");
		printf("	49 - Adaptive Boosting Classification\n");
	
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				changeGrayLevel();
				break;
			case 11:
				fourSquaresColorImage();
				break;
			case 12:
				inverseMatrix();
				break;
			case 13:
				rgb24SplitChannels();
				break;
			case 14:
				colorToGrayscale();
				break;
			case 15:
				grayscaleToBW();
				break;
			case 16:
				rgbToHsv();
				break;
			case 17:
				histogram();
				break;
			case 18:
				multilevelThresholding();
				break;
			case 19:
				floydSteinbergDithering();
				break;
			case 20:
				geometricProperties();
				break;
			case 21:
				labelBinaryImages();
				break;
			case 22:
				labelBinaryImagesWithTwoPass();
				break;
			case 23:
				borderTracingAlgorithm();
				break;
			case 24:
				reconstructFromFile();
				break;
			case 25:
				dilate();
				break;
			case 26:
				erode();
				break;
			case 27:
				open1();
				break;
			case 28:
				close1();
				break;
			case 29:
				boundaryExtraction();
				break;
			case 30:
				regionFilling();
				break;
			case 31:
				statisticalProperties();
				break;
			case 32:
				generalFilter();
				break;
			case 33:
				frequencyDomain();
				break;
			case 34:
				medianFilter();
				break;
			case 35:
				twoDGaussianFilter();
				break;
			case 36:
				oneDGaussianFilters();
				break;
			case 37:
				cannyEdgeDetection();
				break;
			case 38:
				leastMeanSquares_model1();
				break;
			case 39:
				leastMeanSquares_model2();
				break;
			case 40:
				ransac();
				break;
			case 41:
				houghTransform();
				break;
			case 42:
				distanceTransformPatternMatching();
				break;
			case 43:
				statisticalDataAnalysis();
				break;
			case 44:
				principalComponentAnalysis();
				break;
			case 45:
				kMeansClustering();
				break;
			case 46:
				kNearestNeighboursClassifier();
				break;
			case 47:
				naiveBayesianClassifier();
				break;
			case 48:
				linearClassifier();
				break;
			case 49:
				adaptiveBoosting();
				break;
			case 100:
				project();
				break;
			case 122:
				thesis_evaluation_noLines_onTargetConstellation_onFolder();
				break;
			case 123:
				thesis();
				break;
			case 124:
				testConstellationPreprocessingOnSelectedImage();
				break;
			case 125:
				preprocessConstellations();
				break;
			case 133:
				thesis_testOnSingularInputImage_withoutLines();
				break;
			case 111:
				testing();
				break;
			case 222:
				thesis_testOnSingularInputImage_withAllConstellations_withoutLines();
				break;
			case 233:
				thesis_testOnSingularInputImageFromParameters_withAllConstellations_withoutLines();
				break;
		}
	}
	while (op!=0);
	return 0;
}