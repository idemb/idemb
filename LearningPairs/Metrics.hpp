#ifndef _METRICS_H
#define _METRICS_H
#include "Distance.hpp"
#include "armadillo"
using namespace arma;



double L1DistanceScalar(double p1, double p2){
  return fabs(p1-p2);
}


class L2Distance : public Distance
{
	public:
	virtual	~L2Distance(){
	}

	double distance (vec p1, vec p2){
	  vec diff = (p1-p2);
	  double res =  sqrt(dot(diff,diff));
	  return res;
	}
};





Distance * createDistance(string s){
	if (s=="L2Distance"){
		return new L2Distance();
	}
	else{
		cerr<<"Unsupported Distance"<<endl;
		exit(1);
	}
}

#endif



void error(string message){
	cerr<<message<<endl;
	exit(1);
}



