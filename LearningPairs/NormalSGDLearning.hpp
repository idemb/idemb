#ifndef _NORMAL_SGD_LEARNING_H
#define _NORMAL_SGD_LEARNING_H




#include "SGDLearning.hpp"

class NormalSGDLearning: public SGDLearning {
	virtual ~NormalSGDLearning() {

	}

protected:
	void SGD_similar(std::vector<double>& W, const std::vector<double>& Wreg,
			const std::vector<IndexValuePair>& volume, Label tag, double & thold,
			const double C, const double lambda, size_t etha) {
		bool useNonNeagtiveStep = true;

		std::vector<double> W_old(W);
		size_t size = W.size();
		size_t sparse_size = volume.size();
		double dotProd = 0;

		//ID(X_pi_1, X_pi_2) * W
		for (size_t i = 0; i < sparse_size; i++)
			dotProd += volume[i].m_value * W[volume[i].m_index];

		// 1 - { (ID(X_pi_1, X_pi_2) * W) - threshold } * y_i

		//gradient is -  tag * simplex_point.m_value and we need to subtract it
		//due to max( . ,0) this is the gradient only when it's >0 o.w. sg is 0
		if ((1 - ((dotProd - thold) * tag)) > 0) {//sign
			for (auto& simplex_point : volume) {
				W[simplex_point.m_index] += (1.0 / (double) etha)
						* (C * tag * simplex_point.m_value);
				if (useNonNeagtiveStep) {
					W[simplex_point.m_index] =
							W[simplex_point.m_index] < 0 ?
									0 : W[simplex_point.m_index];
				}
			}
			//thold -= ((1.0 / (double) etha) * (C * tag));
		}
		//cout<<W<<endl;
		//printFlattenedSquare(W);
		for (size_t i = 0; i < size; i++) {
			W[i] -= (1.0 * lambda * C / (double) etha) * ((W_old[i] - Wreg[i]));//not accurate to fit what I do in sparse:, should be Wold for a true sgd
			if (useNonNeagtiveStep) {
				//if we get closer to the regularizer we can't go below zero
				//W[i] = W[i] < 0 ? 0 : W[i];
			}
		}

	}

	virtual std::vector<double> learn_similar(
			const std::vector<std::vector<double> >& examples,
			const IDpair& idpair,
			const std::vector<std::vector<size_t> >& indices_of_pairs,
			const std::vector<Label>& tags, const std::vector<double>& Wreg,
			const double C,const double lambda, double& thold) {

		assert(tags.size() == indices_of_pairs.size());

		size_t W_size = idpair.get_total_num_of_vertices();
		//cout<<Wreg.size()<<","<< W_size;

		std::vector<double> W(Wreg.size(), 0);

		assert(Wreg.size() == W_size);

		size_t num_of_pairs = indices_of_pairs.size();

		bool isRandomInd = true;
		for (int j = 0; j < EPOCH_TIMES; ++j) {

			std::vector<int> random_indexes(num_of_pairs);
			std::iota(std::begin(random_indexes), std::end(random_indexes), 0); // Fill with 0, 1, ..., n.
			std::random_shuffle(random_indexes.begin(), random_indexes.end());

			for (size_t i = 0; i < num_of_pairs / prunePairsFactor; i++) {

				//get random index
				size_t random_index = isRandomInd ? random_indexes.back() : i;

				random_indexes.pop_back();

				const std::vector<IndexValuePair>& volume = idpair(
						examples[indices_of_pairs[random_index][0]],
						examples[indices_of_pairs[random_index][1]]);

				SGD_similar(W, Wreg, volume, tags[random_index], thold, C, lambda,
						i + 1);
			}

		}
		makeSymmetric(W);
		return W;
	}

};

SGDLearning * createSGDLearning() {
	return new NormalSGDLearning();
}

#endif

