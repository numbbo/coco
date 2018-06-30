/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   hoy.h
 * Author: volz
 *
 * Created on 21. Juni 2018, 14:48
 */

#ifndef HOY_H
#define HOY_H

#include <vector>

class HVCalculator{
public:
    HVCalculator();
    double computeHV(int dim, int n, std::vector<double> points, std::vector<double> refPoint);
private:
    void stream(double regLow[], double regUp[], const std::vector<double*>& cubs, int lev, double cov);
    static bool cmp(double* a, double* b);
    inline bool covers(const double* cub, const double regLow[]);
    inline bool partCovers(const double* cub, const double regUp[]);
    inline int containsBoundary(const double* cub, const double regLow[], const int split);
    inline double getMeasure(const double regLow[], const double regUp[]);
    inline int isPile(const double* cub, const double regLow[], const double regUp[]);
    inline double computeTrellis(const double regLow[], const double regUp[], const double trellis[]);
    inline double getMedian(std::vector<double>& bounds);
    
    static int dataNumber;
    static int dimension;
    static double dSqrtDataNumber;
    static double volume;
    
};



#endif /* HOY_H */

