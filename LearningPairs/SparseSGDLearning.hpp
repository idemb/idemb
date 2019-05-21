#ifndef _SPARSE_SGD_LEARNING_H
#define _SPARSE_SGD_LEARNING_H




#include "SGDLearning.hpp"
#include <cmath>
#include <limits>
class SparseSGDLearning: public SGDLearning {
public:
	virtual ~SparseSGDLearning() {
	}

protected:
	inline double getWElement(const std::vector<double>& W_hat,
			const std::vector<double>& Wreg, int ind, double beta,
			double gamma_) {
		return beta * W_hat[ind] + gamma_ * Wreg[ind];
	}

	//replaces W_hat[i] with (newValue - gamma_ * Wreg[ind])/beta;
	inline void setWElement(std::vector<double>& W_hat,
			const std::vector<double>& Wreg, int ind, double newValue,
			double beta, double gamma_) {
		W_hat[ind] = (newValue - gamma_ * Wreg[ind])/beta;
	}


	inline void updateWElement(std::vector<double>& W_hat, int ind, double summand, double beta) {
		W_hat[ind] += summand/beta;
	}



#define _CORRECT_SEMIMETRIC
	const double eps = std::numeric_limits<double>::epsilon();
	inline void updateWElement(std::vector<double>& W_hat, const std::vector<double>& Wreg, int ind, double summand,
			double beta, double gamma_) {
#ifndef _CORRECT_SEMIMETRIC
		updateWElement(W_hat, ind, summand, beta);
#else

		double curValue = getWElement(W_hat, Wreg, ind, beta, gamma_) + summand;
		setWElement(W_hat, Wreg, ind, (curValue < eps) ? eps : curValue , beta, gamma_);

		vector<int> ij = indToPair(ind, (int) sqrt(W_hat.size()));//FIXME
		//cout<<ij[0]<<","<<ij[1]<<endl;
		if (ij[0]==ij[1]){
			//setWElement(W_hat, Wreg, ind, 0, beta, gamma_);
			//printFlattenedSquare(toW(W_hat, Wreg, beta, gamma_));

		}



#endif
	}


	// currently not in use, if W[i]<0 ? 0 : W[i],
	inline void correctWElement(std::vector<double>& W_hat,
			const std::vector<double>& Wreg, int ind, double beta,
			double gamma_) {
		if (getWElement(W_hat, Wreg, ind, beta, gamma_)< 0){
			 setWElement(W_hat, Wreg, ind, 0, beta, gamma_);
		}
	}

	//when W_hat will be a sparse metric object this will be replace as well...
	vector<double> toW(const vector<double> & W_hat, const vector<double> & Wreg,double beta,double gamma_){
		vector<double> W(Wreg.size(),0);
		for (size_t i=0;i< Wreg.size();i++) {
			W[i] = getWElement(W_hat, Wreg, i, beta, gamma_); //     cout << i << " , ";
		}
		return W;
	}




	inline void SGD_similar_sparse(std::vector<double>& W_hat,
			const std::vector<double>& Wreg, const std::vector<IndexValuePair>& volume,
			Label tag, double & thold, const double C, const double lambda, size_t etha,
			double & beta, double & gamma_) {
		double CtimesLambda = C * lambda;

		size_t sparse_size = volume.size();
		double dotProd = 0;


		double old_beta = beta;
		double old_gamma = gamma_; //technically redundant in the current code

		//printFlattenedSquare(toW(W_hat,Wreg, beta, gamma_));
		beta = (1.0 - CtimesLambda / (double) etha) * beta;
		gamma_ = (1.0 - CtimesLambda / (double) etha) * gamma_ + CtimesLambda / (double) etha;


		//this means taking simplex point from old_w and subtracting it from new_w after regluarization based update
		//ID(X_pi_1, X_pi_2) * W
		for (size_t i = 0; i < sparse_size; i++)
			dotProd += volume[i].m_value
					* getWElement(W_hat, Wreg, volume[i].m_index, old_beta, old_gamma);

		// 1 - { (ID(X_pi_1, X_pi_2) * W) - threshold } * y_i
		if ((1 - ((dotProd - thold) * tag)) > 0) {//note in x1=x2  W[ind]=0 and dotprod<thold sign
			for (auto& simplex_point : volume) {

				double summand = (1.0 / (double) etha)
								* (C * tag * simplex_point.m_value); //pre-calculating C and -C is an option, plus if C=1...
				updateWElement(W_hat, Wreg, simplex_point.m_index, summand, beta, gamma_);

			}
			//thold -= ((1.0 / (double) etha) * (C * tag));
			//cout<<thold<<endl;
		}
		//practically this means {W_old=W;W-=lambda*c*(W-Wreg);if(should_update(W_old){W+=compute_summand(W_old)}};
		//so it's equivalent to: {W_old=W;if(should_update(W_old){W+=compute_summand(W_old)};W-=lambda*c*(W_old-W_reg)};





	}






	std::vector<double> learn_similar(
			const std::vector<std::vector<double> >& examples,
			const IDpair& idpair,
			const std::vector<std::vector<size_t> >& indices_of_pairs,
			const std::vector<Label>& tags, const std::vector<double>& Wreg,
			const double C,const double lambda, double& thold){
		assert(tags.size() == indices_of_pairs.size());

		//size_t W_size = idpair.get_total_num_of_vertices();


		std::vector<double> W_hat(Wreg.size(), 0);
		size_t num_of_pairs = indices_of_pairs.size();

		double beta = 1;
		double gamma_ = 0;

		bool isRandomInd = true;
		for (int j = 0; j < EPOCH_TIMES; ++j) {

			std::vector<int> random_indexes(num_of_pairs);
			std::iota(std::begin(random_indexes), std::end(random_indexes), 0); // Fill with 0, 1, ..., n.
			std::random_shuffle(random_indexes.begin(), random_indexes.end());

			for (size_t i = 0; i < num_of_pairs / prunePairsFactor; i++) {

				//get random index
				size_t random_index = isRandomInd ? random_indexes.back() : i;
				random_indexes.pop_back();
				if (indices_of_pairs[random_index][0] >= examples.size()|| indices_of_pairs[random_index][1]>=examples.size() ){
					cerr<<"SparseSGD, learn_similar: index of example is out of bound"<<endl;
					exit(1);
				}
				const std::vector<IndexValuePair>& volume = idpair(
						examples[indices_of_pairs[random_index][0]],
						examples[indices_of_pairs[random_index][1]]);

				SGD_similar_sparse(W_hat, Wreg, volume, tags[random_index],
						thold, C, lambda,  i + 1, beta, gamma_);
			}
		}
		vector<double> W = toW(W_hat, Wreg, beta, gamma_);
		makeSymmetric(W);
		return W;
	}

};

SGDLearning * createSGDLearning() {
	return new SparseSGDLearning();
}

#endif
