#ifndef _DISTANCE_H
#define _DISTANCE_H


#include "armadillo"


	class Distance{
	public:
		virtual ~Distance(){

		}
		double virtual distance(vec p1, vec p2) = 0;
	};




class ZeroOneDistance : public Distance
{
	public:
	virtual	~ZeroOneDistance (){
	}

	double distance (vec p1, vec p2){
	  vec diff = (p1-p2);
	  double var_sqr =  dot(diff,diff);
	  return (var_sqr == 0 ? 0 : 1);
	}
};





#endif
