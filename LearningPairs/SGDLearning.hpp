#ifndef _SGD_LEARNING_H
#define _SGD_LEARNING_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <cassert>
#include <exception>
#include <time.h>
#include "armadillo"
#include "ID/IDpair.hpp"
#include "Services.hpp"

typedef int Label;

#define _same -1
#define _notsame 1


using namespace std;
using namespace arma;

#define EPOCH_TIMES 3
static int prunePairsFactor = 1;
static double thresholdValue = 1;



void printFlattenedSquare(const vector<double> & W){
	cout << "W.size(): " << W.size() << " \nW:" << endl;
	for (size_t i = 0; i < W.size(); i++) {
		cout << W[i] << " , ";
		if ((i + 1) % (int) sqrt(W.size()) == 0) {
			cout << endl;
		}
	}
}
#include "Distance.hpp"

Distance * createZeroOneDistance(){
	return new ZeroOneDistance();
}


class SGDLearning{
public:
virtual ~SGDLearning(){

}


protected:



inline vector<int> indToPair(int arg, int size){
	int i = arg / size;//line
	int j = arg % size;//row
	return {i,j};
}

inline int pairToInd(int i, int j, int n){
	return n * i + j;
}



public:


std::vector<double> constructWregDistance(Grid points_pair, Distance * distObj) {
	points_pair.double_grid();
    std::vector<double> Wreg(points_pair.get_num_of_vertices(), 0);
    std::vector<double> pair_vertex(points_pair.get_num_of_dims(), 0);
    std::vector<double> first_vertex(pair_vertex.size()/2, 0);
    std::vector<double> second_vertex(pair_vertex.size()/2, 0);

    for (size_t i = 0; i < points_pair.get_num_of_vertices(); i++) {

        //get a vertex from the grid by index
        points_pair.get_vertex(i, pair_vertex);
        std::vector<double>::iterator it = pair_vertex.begin();

        //the first half of pair_vertex holds the values of the first point
        first_vertex.assign(it , it + pair_vertex.size()/2);
        it += pair_vertex.size()/2;

        //the second half of pair_vertex holds the values of the second point
        second_vertex.assign(it , it + pair_vertex.size()/2);

        //assign Wreg the value (|second_vertex|-|first_vertex|)
        Wreg[i] = distObj->distance(first_vertex, second_vertex);
    }

return Wreg;
}


std::vector<double> construct_Wreg(Grid points_pair) {
	points_pair.double_grid();
	size_t num_of_ver = points_pair.get_num_of_vertices();

	std::vector<double> Wreg(num_of_ver, 0);
	for (size_t ind = 0; ind < num_of_ver; ind++) {
		Wreg[ind] = thresholdValue;

		vector<int> ij = indToPair(ind, (int) sqrt(Wreg.size()));//FIXME

		if (ij[0]==ij[1]){
			Wreg[ind] = 0;
		}
	}
	return Wreg;
}


protected:
virtual void makeSymmetric(std::vector<double> & W) {
	float eps = 0;
	int n = (int) sqrt(W.size());
	if (abs(sqrt(W.size()) - n) > eps) {
		cerr << "makeSymmetric: flat vector cannot represent square matrix" << endl;
		exit(1);
	}
	for (size_t ind = 0; ind < W.size(); ind++) {
		vector<int> v = indToPair(ind,n);
		int i=v[0];
		int j=v[1];
		if (j > i) {// in the source pair-index
			W[pairToInd(j,i,n)] = W[ind];//target pair-index is swapped
		}

	}
}



virtual std::vector<double> learn_similar(
		const std::vector<std::vector<double> >& examples,
		const IDpair& idpair,
		const std::vector<std::vector<size_t> >& indices_of_pairs,
		const std::vector<Label>& tags, const std::vector<double>& Wreg,
		const double C,const double lambda, double& thold)= 0;


public:
std::vector<double> run(const std::vector<std::vector<double> >& examples,
		const std::vector<std::vector<size_t> >& indices_of_pairs,
		const std::vector<Label>& tags,
		const std::vector<std::vector<double> >& discrete_points,
		std::vector<double> Wreg,
		const double C,const double lambda, double& thold) {
	//std::vector<double> Wreg;
	std::vector<double> W;

	Grid grid_pair(discrete_points);

	IDpair id_pair(grid_pair);

	//Wreg = construct_Wreg(grid_pair);

	W = learn_similar(examples, id_pair, indices_of_pairs, tags, Wreg, C, lambda,
			thold);

	return W;
}



// -1(similar) , 1(non-similar)
Label classification(const std::vector<double>& W,
		const std::vector<IndexValuePair>& non_zero, double thold) {
	double dotProd = 0;
	Label ans = 0;

	for (size_t i = 0; i < non_zero.size(); i++)
		dotProd += non_zero[i].m_value * W[non_zero[i].m_index];

	if (dotProd < thold) //sign
		ans = _same;
	else
		ans = _notsame;

	return ans;
}

};

#endif
