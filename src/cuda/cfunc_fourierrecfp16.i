/*interface*/
%module cfunc_fourierrecfp16

%{
#define SWIG_FILE_WITH_INIT
#include "cfunc_fourierrec.cuh"
%}

class cfunc_fourierrec
{
public:
  %immutable;
  size_t n;
  size_t nproj;
  size_t nz;
  
  %mutable;
  cfunc_fourierrec(size_t nproj, size_t nz, size_t n);
  ~cfunc_fourierrec();
  void backprojection(size_t f, size_t g, size_t theta_, size_t nproj0, size_t stream);  
};
