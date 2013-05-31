/**
 * Common Libs
 */
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>

/**
 * OpenCV libs
 */

#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <vector>
#include <cxcore.h>
/**
 * KLT Tracker Related Libs
 */
#include "pnmio.h"
#include "klt.h"

/**
 * name space
 */
using namespace cv;
using namespace std;
#define FILE_SPERATOR "/"
#ifdef __WIN32__
#undef FILE_SPERATOR
#define FILE_SPERATOR "\\"
#endif
#define PI 3.1415926


struct Video {
	string name;
	string type;
	string parrent;
	string fullpath;
};

struct record {
	int frameStart, frameEnd;
	double dist, ydist;
	CvPoint start, end;
	int col;				//Right = 1 , Left = 2;

};

double getangle(CvPoint p1, CvPoint p3)
{
	double angle=0;
	int q =0;
	if(p3.x == p1.x)
		angle = 90;
	else if(p3.x>p1.x)
	{
		if(p3.y>p1.y)
		{
			angle = (double)(p3.y - p1.y) / (p3.x - p1.x);
			angle = atan(angle);
			angle = angle * 180/PI;
			q = 4;
		}
		else
		{
			angle = (double)(p1.y - p3.y) / (p3.x - p1.x);
			angle = atan(angle);
			angle = angle * 180/PI;
			q = 1;							
		}
	}
	else if(p3.x<p1.x)
	{
		if(p3.y>p1.y)
		{
			angle = (double)(p3.y - p1.y) / (p1.x - p3.x);
			angle = atan(angle);
			angle = angle * 180/PI;
			q = 3;
		}
		else
		{
			angle = (double)(p1.y - p3.y) / (p1.x - p3.x);
			angle = atan(angle);
			angle = angle * 180/PI;
			q = 2;
		}
	}
	
	switch(q)
	{
		case 1: angle = angle;break;
		case 2: angle = 180 - angle;break;
		case 3: angle = 180 + angle;break;
		case 4: angle = 360 - angle;break;
		default : break;
	}

	//cout<<"\nAngle::"<<angle<<"("<<p1.x<<","<<p1.y<<")";
	return angle;
}


bool checkfield(CvPoint p, vector<CvPoint> upper, vector<CvPoint> lower)
{
	CvPoint p1,p2,p3,p4;
	double m=0,m2=0;
	vector<CvPoint>::iterator it2 = lower.begin();
	for(vector<CvPoint>::iterator it = upper.begin() ; it != upper.end(); ++it, ++it2)
	{
		p2 = *it;
		p1 = *(it-1);
		
		p4 = *it2;
		p3 = *(it2-1);
		if(p2.x>p.x)
			break;
	}
	
	if(p.y<p1.y && p.y<p2.y)
		return true;
	else if(p.y<p1.y || p.y<p2.y)
	{
		double angle = getangle(p1,p2);
		double angle2 = getangle(p1,p);
		if(angle2>angle)
			return true;
		else
			return false;
	}
	else if(p.y>p3.y && p.y>p4.y)
		return true;
	else if(p.y>p3.y || p.y>p4.y)
	{
		double angle = getangle(p3,p4);
		double angle2 = getangle(p3,p);
		if(angle2<angle)
			return true;
		else
			return false;
	}
	return false;
}

int found(struct record temp, vector<struct record> data)
{
	for(int i=0;i<data.size();i++)
		if(data[i].start.x == temp.start.x && data[i].start.y == temp.start.y && data[i].frameStart == temp.frameStart)
			return i;
	return -1;
}

int kltagain(vector<struct record> data, string vid, int mos)
{
	//cout<<"\nMOS::"<<mos;
	cout << "Start Tracking!!\n";

	/**
	 * Instantiating KLT tracker
	 */
	unsigned char *img1, *img2;
	int resizeScale = 1;
	KLT_TrackingContext tc;
	KLT_FeatureList fl;
	KLT_FeatureTable ft;
	int nFeatures = 100;
	int ncols, nrows;

	/*
	 * Reading AVI file
	 */
	VideoCapture capture(vid);
	if (!capture.isOpened()) {
		cout << "Error in opening the video";
		return 1;
	}
	VideoWriter videoWriter;

	Mat originalFrame;
	int nFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);
	nFrames = 2000;
	tc = KLTCreateTrackingContext();
	fl = KLTCreateFeatureList(nFeatures);
	ft = KLTCreateFeatureTable(nFrames, nFeatures);
	tc->sequentialMode = TRUE;
	tc->writeInternalImages = FALSE;
	tc->affineConsistencyCheck = 2; /* set this to 2 to turn on affine consistency check */
	printf("%s", "# Of Frames: ");
	printf("%d\n", nFrames);
	int tracks[nFeatures][nFrames];
			
	for (int frameIndex = 0; frameIndex < nFrames; frameIndex++) 
	{
	
		cout<<endl<<frameIndex;
		capture >> originalFrame;
		if (originalFrame.empty())
			break;
			
		int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
		int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

		Mat rot_mat(2, 3, CV_32FC1);
		Point center = Point(0, 0);
		double scale = 1.0;
		/**
		 * Rotate based on the line angles
		 */
		rot_mat = getRotationMatrix2D(center, 0, scale);
		Mat frame;
		/// Rotate the warped image
		warpAffine(
				originalFrame,
				frame,
				rot_mat,
				Size(originalFrame.cols / resizeScale,
						originalFrame.rows / resizeScale));
		Mat gray;
		cvtColor(frame, gray, CV_BGR2GRAY);

		
		if(frameIndex<mos)
			continue;
		
		if(frameIndex>mos+40)
		{
			return 1;
		}

		nrows = frame.rows;
		ncols = frame.cols;

		if (frameIndex == mos) 
		{
			
			img1 = (unsigned char *) malloc(ncols * nrows * sizeof(unsigned char));
			int index = 0;
			for (int i = 0; i < gray.rows; i++) 
			{
				for (int j = 0; j < gray.cols; j++) 
				{
					img1[index] = gray.at<uchar>(i, j);
					index++;
				}
			}

			KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
			KLTStoreFeatureList(fl, ft, 0);
			for (int i = 0; i < nFeatures; i++) 
			{
				for (int j = 0; j < nFrames; j++) 
				{
					tracks[i][j] = -1;
				}
			}

		} 
		else 
		{
			img2 = img1;
			int index = 0;
			for (int i = 0; i < gray.rows; i++) 
			{
				for (int j = 0; j < gray.cols; j++) 
				{
					img1[index] = gray.at<uchar>(i, j);
					index++;
				}
			}
			KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
			KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
			KLTStoreFeatureList(fl, ft, frameIndex);

			/**
			 * Creating the Video
			 */
			
			struct record temp;
			CvPoint p1, p2, p3;
			for (int i = 0; i < ft->nFeatures; i++) 
			{
				int flag = 0; p2.x = 0; p2.y = 0; p1.x = 0; p1.y = 0;
				for (int j = frameIndex; j >= 0 && ft->feature[i][j]->val >= 0; j--) 
				{
					if (ft->feature[i][j]->val > 0 && flag == 1) 
					{
						p1.x = ft->feature[i][j]->x;
						p1.y = ft->feature[i][j]->y;							
						flag = 2;
						temp.frameStart = j;
						break;
					}
					else if (ft->feature[i][frameIndex]->val == 0) 
					{
						if(flag == 0)
						{
							p2.x = ft->feature[i][j]->x;
							p2.y = ft->feature[i][j]->y;	
							p3.x = ft->feature[i][j-1]->x;
							p3.y = ft->feature[i][j-1]->y;	
							flag =1;
							temp.frameEnd = j;
						}
					}
				}
				if(p2.x !=0 && p2.y!=0 && p1.x !=0 && p1.y!=0 && flag == 2)
				{
						temp.start = p1;
						temp.end = p2;
						if(temp.frameStart != 0)
						{
							int pos = found(temp, data);
							if(pos != -1)
							{	
								if(data[pos].col == 1)
								{
									line(frame, p1, p2, CV_RGB(255, 0, 0),1,8,0);
									circle(frame,p1, 2,CV_RGB(255, 0, 0), -1);
								}
								else if(data[pos].col == 2)
								{
									line(frame, p1, p2, CV_RGB(0, 255, 0),1,8,0);
									circle(frame,p1, 2,CV_RGB(0, 255, 0), -1);
								}
								else
								{
									cout<<endl<<data[pos].col;
									line(frame, p1, p2, CV_RGB(0, 0, 255),1,8,0);
									circle(frame,p1, 2,CV_RGB(0, 0, 255), -1);
								}
							}
							else 								
							{
								line(frame, p1, p2, CV_RGB(255, 255, 0),1,8,0);
								circle(frame,p1, 2,CV_RGB(255, 255, 0), -1);
							} 
							
						}		
				}
				
			}//for ends
		}
		
		imshow("Frame",frame);
		char name[20];
		sprintf(name, "Frames/%d.png",frameIndex);
		imwrite(name, frame);
		cvWaitKey(10);		
			
	}
	KLTFreeFeatureTable(ft);
	KLTFreeFeatureList(fl);
	KLTFreeTrackingContext(tc);
	
	return 0;				
}

int main(int argc, char** argv) 
{
	double startTime = clock();
	vector<CvScalar> con_col;
	vector<struct record> data;
	vector<double> Leftward;
	vector<double> Rightward;
	int mos;
	if (argc < 2) {
		cout << "Input File is missing!!\n";
		cout << "1-Input file containing the video names 2-Scale\n";
		return -1;
	}
	int resizeScale = 1;
	string result_video = "result_video";
	

	cout << "Start KLT Tracker with following parameters : \n";
	cout << "Scale: 1/" << resizeScale << '\n';
	cout << "Result Video Folder: " << result_video << '\n';
	cout << "Reading Input File : " << argv[1] << '\n';

	
	char *mn = new char[40];
	string vid,s,s1, dataset;	// names for different files
	vid.assign(argv[1]);
	
	ifstream v("mapping.txt");		// contains MOS information
	int flag10=0;
	while(getline(v,s))
	{
		if(flag10==1)
			break;
		char *pch;
		char *str = new char[s.size()+1];;
		strncpy(str, s.c_str(), s.size());
		pch = strtok (str,";");
		int c=0;
		while(pch != NULL )
		{
			string tok(pch);
			
			if(c==1)
			{
				tok += ".avi";
				int l = tok.length();
				if(tok.compare(0,l,vid)==0)
				{						
					dataset = tok.substr(0,l-4);
					flag10=1;
				}
			
			}
			if(c==2)
			{
				mos = atoi(tok.c_str());
			}
			c++;
			pch = strtok (NULL, ";");
		}
	}
	
	//cout<<"\nDataset :: "<<dataset;
	//cout<<"\nMOS::"<<mos;
	
	vid = "Videos/"+vid;			// Videos must be in this folder
	dataset = "Dataset/" + dataset;		// The dataset will be outputted here


	cout << "Start Tracking!!\n";

	/**
	 * Instantiating KLT tracker
	 */
	unsigned char *img1, *img2;
	KLT_TrackingContext tc;
	KLT_FeatureList fl;
	KLT_FeatureTable ft;
	int nFeatures = 100;
	int ncols, nrows;

	/*
	 * Reading AVI file
	 */
	VideoCapture capture(vid);
	if (!capture.isOpened()) {
		cout << "Error in opening the video";
		return 1;
	}
	VideoWriter videoWriter;

	Mat originalFrame;
	int nFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);
	nFrames = 2000;
	tc = KLTCreateTrackingContext();
	fl = KLTCreateFeatureList(nFeatures);
	ft = KLTCreateFeatureTable(nFrames, nFeatures);
	tc->sequentialMode = TRUE;
	tc->writeInternalImages = FALSE;
	tc->affineConsistencyCheck = 2; /* set this to 2 to turn on affine consistency check */
	printf("%s", "# Of Frames: ");
	printf("%d\n", nFrames);
	int tracks[nFeatures][nFrames];
			
	for (int frameIndex = 0; frameIndex < nFrames; frameIndex++) 
	{
	
		cout<<endl<<frameIndex;
		capture >> originalFrame;
		if (originalFrame.empty())
			break;
			
		int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
		int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

		Mat rot_mat(2, 3, CV_32FC1);
		Point center = Point(0, 0);
		double scale = 1.0 / resizeScale;
		/**
		 * Rotate based on the line angles
		 */
		rot_mat = getRotationMatrix2D(center, 0, scale);
		Mat frame;
		/// Rotate the warped image
		warpAffine(
				originalFrame,
				frame,
				rot_mat,
				Size(originalFrame.cols / resizeScale,
						originalFrame.rows / resizeScale));
		Mat gray;
		cvtColor(frame, gray, CV_BGR2GRAY);

		
		if(frameIndex<mos)
			continue;
		
		if(frameIndex>mos+40)
		{
			double max1 =0, max2=0;
			for(int i=0;i<data.size();i++)
			{
				data[i].dist = sqrt((data[i].end.x - data[i].start.x) * (data[i].end.x - data[i].start.x)) + ((data[i].end.y - data[i].start.y) * (data[i].end.y - data[i].start.y));
				if(data[i].dist>max1)
					max1 = data[i].dist;
				
				data[i].ydist = data[i].start.y;
				if(data[i].ydist > max2)
					max2 = data[i].ydist;
					
				if(data[i].start.x > data[i].end.x)
					data[i].col = 2;
				else
					data[i].col = 1;
			}
			
			ofstream myfile;
			myfile.open (dataset.c_str());
			
			for(int i=0;i<data.size();i++)
			{
					myfile<<"\n"<<i<<"\t"<<data[i].start.x<<"\t"<<data[i].start.y<<"\t"<<data[i].end.x<<"\t"<<data[i].end.y<<"\t"<<data[i].frameStart<<"\t"<<data[i].frameEnd<<"\t"<<(data[i].dist/max1)<<"\t"<<"0"<<"\t"<<(data[i].ydist/max2)<<"\t"<<data[i].col;
			}	
			myfile.close();
			double endTime = clock();
			kltagain(data, vid, mos);
			
			cout<<"\n\nElapsed Time::"<<endTime-startTime<<endl;
			return 1;
		}

		nrows = frame.rows;
		ncols = frame.cols;

		if (frameIndex == mos) 
		{
			
			img1 = (unsigned char *) malloc(ncols * nrows * sizeof(unsigned char));
			int index = 0;
			for (int i = 0; i < gray.rows; i++) 
			{
				for (int j = 0; j < gray.cols; j++) 
				{
					img1[index] = gray.at<uchar>(i, j);
					index++;
				}
			}

			KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
			KLTStoreFeatureList(fl, ft, 0);
			for (int i = 0; i < nFeatures; i++) 
			{
				for (int j = 0; j < nFrames; j++) 
				{
					tracks[i][j] = -1;
				}
			}

		} 
		else 
		{
			img2 = img1;
			int index = 0;
			for (int i = 0; i < gray.rows; i++) 
			{
				for (int j = 0; j < gray.cols; j++) 
				{
					img1[index] = gray.at<uchar>(i, j);
					index++;
				}
			}
			KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
			KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
			KLTStoreFeatureList(fl, ft, frameIndex);

			/**
			 * Creating the Video
			 */
			
			struct record temp;
			CvPoint p1, p2, p3;
			for (int i = 0; i < ft->nFeatures; i++) 
			{
				int flag = 0; p2.x = 0; p2.y = 0; p1.x = 0; p1.y = 0;
				for (int j = frameIndex; j >= 0 && ft->feature[i][j]->val >= 0; j--) 
				{
					if (ft->feature[i][j]->val > 0 && flag == 1) 
					{
						p1.x = ft->feature[i][j]->x;
						p1.y = ft->feature[i][j]->y;							
						flag = 2;
						temp.frameStart = j;
						break;
					}
					else if (ft->feature[i][frameIndex]->val == 0) 
					{
						if(flag == 0)
						{
							p2.x = ft->feature[i][j]->x;
							p2.y = ft->feature[i][j]->y;	
							p3.x = ft->feature[i][j-1]->x;
							p3.y = ft->feature[i][j-1]->y;	
							flag =1;
							temp.frameEnd = j;
						}
					}
				}
				if(p2.x !=0 && p2.y!=0 && p1.x !=0 && p1.y!=0 && flag == 2)
				{
						temp.start = p1;
						temp.end = p2;
						if(temp.frameStart != 0)
						{
							int pos = found(temp, data);
							if(pos == -1)
							{	
								data.push_back(temp);
							}
							else 								
							{
								line(frame, p1, p2, CV_RGB(255, 0, 0),1,8,0);
								circle(frame,p1, 2,CV_RGB(255, 0, 0), -1);
								
								data[pos].end = p2;
								data[pos].frameEnd = temp.frameEnd;
							} 
							
						}		
				}
				
			}//for ends
		}
		
		imshow("Frame",frame);
		cvWaitKey(10);		
			
	}
	KLTFreeFeatureTable(ft);
	KLTFreeFeatureList(fl);
	KLTFreeTrackingContext(tc);
	
	return 0;

}
