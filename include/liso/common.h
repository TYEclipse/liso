#pragma once

#include <cmath>

#include <pcl/point_types.h>

using PointType = pcl::PointXYZINormal;

const double SCAN_PERIOD = 0.1;
const bool DISTORTION = false;
const double DISTANCE_SQ_THRESHOLD = 25;
const double NEARBY_SCAN = 2.5;

template <typename T>
class Accumulator
{
private:
    T m = 0;
    T s = 0;
    T N = 0;
    T minx = 0;
    T maxx = 0;
public:

    Accumulator()
    {}

    Accumulator(T init)
    {
        this->addDataValue(init);
    }

    void addDataValue(T x)
    {
        if(N<1)
        {
          minx = x;
          maxx = x;
        }
        N++;
        s=s+(N-1)/N*(x-m)*(x-m);
        m=m+(x-m)/N;
        if(x<minx) minx = x;
        if(x>maxx) maxx = x;
    }
    T mean()
    {
        return  m;
    }
    T var()
    {
        return s/N;
    }
    T stddev()
    {
        return sqrt(var());
    }
    T min()
    {
        return minx;
    }
    T max()
    {
        return maxx;
    }
    T distribution(T x)
    {
      x = (x-m)/stddev()*M_SQRT1_2;
      return (erf(x)+1)/2;
    }
};
