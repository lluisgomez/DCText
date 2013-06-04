#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <math.h>
#include <stdlib.h>
#include<iostream>
#include "FrequencyTransforms.h"

using namespace cv;
using namespace std;

// DCT text detection
void dctext(Mat& input, Mat& output,
	     int threshold, int low_thresh, int cliffthresh) 
{

  // selected coefficients of 8*8 DCT;
  int numbest = 18;
  int bestcoeffs[18] = {3, 4, 5, 11, 12, 13, 19, 20, 21, 43, 44, 45, 51, 52, 53, 59, 60, 61};
  int bestcoeffs_v[18] = {3, 4, 5, 11, 12, 13, 19, 20, 21, 43, 44, 45, 51, 52, 53, 59, 60, 61};
  int bestcoeffs_h[18] = {24, 25, 26, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 45, 46, 47};
  const int block_size = 8;


  if ((input.cols%block_size != 0)||(input.rows%block_size != 0))
  {
	input = input(Rect(0,0,input.cols-input.cols%block_size,input.rows-input.rows%block_size));
  }
	

  
  int block_rows = input.rows / block_size, block_cols = input.cols / block_size;
  Mat block_energies_v(block_rows, block_cols, DataType<float>::type);
  Mat block_energies_h(block_rows, block_cols, DataType<float>::type);
  Mat block_energies(block_rows, block_cols, DataType<float>::type, Scalar(-1));
  //block_energies = -1;
  Mat block_means(block_rows, block_cols, DataType<float>::type);
  Mat block_energies_new(block_rows, block_cols, DataType<float>::type);
  DiscreteCosineTransform<float> dct(Size(block_size,block_size), true);
 
  // compute dct energies
  for(int i=0; i<input.rows; i += block_size)
  {
	  for(int j=0; j<input.cols; j += block_size)
    {
		  float dctsum_v=0, dctsum_h=0;
		  Mat block_dct;
      Mat block = input(Rect(j, i, block_size, block_size));
      dct.do_transform(block,block_dct);
		  double *block_dct_ptr = (double*)block_dct.data;
		  for(int l=0; l<numbest; l++)
			  dctsum_v += fabs(block_dct_ptr[bestcoeffs_v[l]]);
		  block_energies_v.at<float>(i/block_size,j/block_size) = fabs(dctsum_v);
		  for(int l=0; l<numbest; l++)
			  dctsum_h += fabs(block_dct_ptr[bestcoeffs_h[l]]);
		  block_energies_h.at<float>(i/block_size,j/block_size) = fabs(dctsum_h);
		  block_energies.at<float>(i/block_size,j/block_size) = (block_energies_v.at<float>(i/block_size,j/block_size) + block_energies_h.at<float>(i/block_size,j/block_size))/2;
		  block_means.at<float>(i/block_size,j/block_size) = block_dct_ptr[0];
    }
  }
  for (int i=1; i<block_energies.rows-1; i++)
  {
	  for (int j=1; j<block_energies.cols-1; j++)
	  {
		  if (block_energies.at<float>(i,j)>threshold*0.9)
		  block_energies.at<float>(i,j) = ((0.5*block_energies_h.at<float>(i,j-1) + block_energies_h.at<float>(i,j) + 0.5*block_energies_h.at<float>(i,j+1))/2 + (0.5*block_energies_v.at<float>(i-1,j) + block_energies_v.at<float>(i,j) + 0.5*block_energies_v.at<float>(i+1,j))/2)/2;
	  }
  }

  // now mark blocks as text if their energies are greather than a threshold
  Mat text_blocks = block_energies > threshold;

   // ignore first and last columns and rows (they're too noisy)
  for(int i=0; i<text_blocks.rows; i++)
    text_blocks.at<unsigned char>(i,0) = text_blocks.at<unsigned char>(i,text_blocks.cols-1) = 0;
  for(int j=0; j<text_blocks.cols; j++)
    text_blocks.at<unsigned char>(0,j) = text_blocks.at<unsigned char>(text_blocks.rows-1,j) = 0;
  
  // ignore "cliffs" in dct 0 coeff
  if(cliffthresh)
  {
	  for(int i=1; i<text_blocks.rows-1; i++)
    {
		  for(int j=1;j<text_blocks.cols-1;j++)
		  {
			  if( fabs(block_means.at<float>(i,j-1) - block_means.at<float>(i,j+1)) > cliffthresh && block_energies.at<float>(i,j-1) + block_energies.at<float>(i,j+1) < threshold*2)
	      text_blocks.at<unsigned char>(i,j)=0;
			  if( fabs(block_means.at<float>(i-1,j) - block_means.at<float>(i+1,j)) > cliffthresh && block_energies.at<float>(i-1,j) + block_energies.at<float>(i+1,j) < threshold*2)
	      text_blocks.at<unsigned char>(i,j)=0;
		  }
    }
  }

  //Draw detected text blocks
  for(int i=0; i<output.rows; i++)
  {
    for(int j=0; j<output.cols; j++)
      if(text_blocks.at<unsigned char>(i/block_size,j/block_size))
      { 
        // all candidate 8*8 text blocks drawn in black color;
          output.at<unsigned char>(i,j) = 0;
      }
  }
}


int main (int argc, char** argv)
{

    if (argc < 2)
    {
      cout << "DCText: Text Detection by DCT energy localization\n        Takes an image as input and saves results in output.png (text blocks are drawn in black)\n        Usage: dctext input.png" << endl;
      exit(0);
    }

    int low_thresh_g=12, threshold_g=20;
    int cliffthresh_g=100;
    
    Mat originalImage = imread(argv[1],0);
		Mat input(originalImage.size(), DataType<float>::type);
    originalImage.copyTo(input);
		Mat output(originalImage);


		dctext(input, output, threshold_g, low_thresh_g, cliffthresh_g);
		imwrite("output.png", output);
}
