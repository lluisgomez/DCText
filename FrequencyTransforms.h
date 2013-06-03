#include "opencv2/imgproc/imgproc.hpp"
#include <fftw3.h>

using namespace cv;
using namespace std;

template<class T>
class FrequencyTransform
{
 public:

  FrequencyTransform() {}

  virtual void do_transform(Mat& input, Mat& output) = 0;

 protected:

};

template<class T>
class DiscreteCosineTransform : public FrequencyTransform<T>
{
 public:

  DiscreteCosineTransform(Size size, bool _normalize=false)  : normalize(_normalize)
    {    
      input  = Mat(size.width, size.height, DataType<double>::type);
      output = Mat(size.width, size.height, DataType<double>::type);

      p = fftw_plan_r2r_2d(size.width, size.height, (double*)input.data, (double*)output.data, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
    }

  virtual void do_transform(Mat& _input, Mat& _output)
    {
      _input.convertTo(input,input.type());

      fftw_execute(p);

      if(normalize)
	      output /= 2.0 * output.rows * 2.0 * output.cols;

      output.copyTo(_output);

    }

  ~DiscreteCosineTransform()
    {
      fftw_destroy_plan(p);
    }

 protected:
  Mat  input, output;

  fftw_plan p;
  bool normalize;
};
