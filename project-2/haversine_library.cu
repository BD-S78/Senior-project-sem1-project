#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

__global__ void haversine_distance_kernel(int size, const double *x1,const double *y1,
    const double *x2,const double *y2, double *dist)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  // https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
  const double PI = 3.14159265358979323846;
  const double R = 6371.0;
  const double d2r = PI / 180.0;

  double lat1 = y1[i] * d2r;
  double lat2 = y2[i] * d2r;
  double dlat = (y2[i] - y1[i]) * d2r;
  double dlon = (x2[i] - x1[i]) * d2r;

  double a = sin(dlat / 2) * sin(dlat / 2)
           + cos(lat1) * cos(lat2) * sin(dlon / 2) * sin(dlon / 2);

  dist[i] = 2 * R * asin(sqrt(a));
}


void run_kernel(int size, const double *x1,const double *y1, const double *x2,const double *y2, double *dist)
   
{
  dim3 dimBlock(1024);
  printf("in run_kernel dimBlock.x=%d\n",dimBlock.x);

  dim3 dimGrid(ceil((double)size / dimBlock.x));
  
  haversine_distance_kernel<<<dimGrid, dimBlock>>>
    (size,x1,y1,x2,y2,dist);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_kernel launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}
