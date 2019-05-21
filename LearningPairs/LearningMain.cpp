//TODO: check synthetic euclidean with distReg=true

#define MD_PRINT_STR_WNAME(os, name, a) \
    do { (os) << (name) << " = " << (a) << std::endl; } while(false)


#define MD_PRINT(a)        MD_PRINT_STR_WNAME(std::cout, #a, (a))

#include "SGDLearning.hpp"


#include "SparseSGDLearning.hpp"


//#include "NormalSGDLearning.hpp"

#include <ctime>

#include "Metrics.hpp"



//colors
const double lambda = 1;
const double C = 0.5;
const bool useSinglesAdapter=false;
bool distReg=false;
///////////////////////////////////colors////////////////////////////////////////////////////////////


#include "CSVReader.hpp"

#include <iostream>     // cout, endl
#include <fstream>      // fstream
#include <vector>
#include <string>
#include <algorithm>    // copy
#include <iterator>     // ostream_operator
#include <map>


#include <sstream>





int hex2int(string s) {
    unsigned int x;
    std::stringstream ss;
    ss << std::hex << s;
    ss >> x;

    return static_cast<int>(x) ;
}

int hexToR(string h) {return hex2int(h.substr(0,2));}
int hexToG(string h) {return hex2int(h.substr(2,2));}
int hexToB(string h) {return hex2int(h.substr(4,2));}


vector<double> hex2RGB(string h){
	int r = hexToR(h);
	int g = hexToG(h);
	int b = hexToB(h);
	return {(double)r, (double)g, (double)b};
}

string removeQuotes(string s){
	int length = s.length();
	if (s[0] == '\"' && s[length-1]=='\"'){
		return s.substr(1,length-1);
	}
	else {
		cerr<<s<<" is not quoted correctly";
		exit(1);
	}
}


//indices
//typedef double Element;
template <typename Element>
void pairsToSinglesAndIndicesRep(const vector<vector<Element> > & pairs, vector<Element> & singles, vector<vector<size_t> >  & pairsIndices){
	std::map<Element,size_t>  indices;
	size_t ind =0;
	for (size_t i=0;i<pairs.size(); i++){
		for (int j=0;j<2;j++){
			if (! indices.count(pairs[i][j])){

				Element single = pairs[i][j];
				singles.push_back(single);
				indices[single] = ind;
				ind++;

			}
			else{

			}
		}

		vector<size_t> pairIndices = {indices[pairs[i][0]], indices[pairs[i][1]]};

		pairsIndices.push_back(pairIndices);
	}
}


//////////////////////////colors end

#ifdef PEGASOS_ADAPTER

///////pegasos adapter begin
#include "Services.hpp"
#include <SgdSolver.hpp>
#include <IDEmbedders.hpp>


void toContinuousVector(const vector< vector< vector<double> > > & pairsVector, vector<double> & continuousOutput){

	for (size_t i=0;i<pairsVector.size();i++){
		for (size_t j=0; j<pairsVector[i].size();j++){//just 0 and 1, the two indices of a pair
			for (size_t k=0; k<pairsVector[i][j].size();k++){
				continuousOutput.push_back(pairsVector[i][j][k]);
			}
		}
	}

}



void samplesAndIndicesToPairs(const vector< vector <double> > & samples, const std::vector<std::vector<size_t> >& trainIndicesOfPairs , vector< vector< vector<double> > > & outputPairs){
	for (size_t i=0; i<trainIndicesOfPairs.size(); i++){
		vector< vector<double> > currentPair = {samples[trainIndicesOfPairs[i][0]], samples[trainIndicesOfPairs[i][1]]};
		outputPairs.push_back( currentPair );
	}

}



//using embedder for singles now, need to work on regularization
void trainAndTestSinglesAdapter(const std::vector< vector <double> > & trainSamples,
		 const std::vector<std::vector<size_t> >& trainIndicesOfPairs,
			const std::vector<Label>& trainTags,
			const std::vector<std::vector<double> >& gridpair, double& tholdArg,
			const std::vector<vector<double> > & testSamples,
			const std::vector<std::vector<size_t> >& testIndicesOfPairs,
			const std::vector<Label>& testTags,
			const double lambda,
			bool doPrintW){

	vector< vector< vector<double> > >  trainPairs;
	samplesAndIndicesToPairs(trainSamples, trainIndicesOfPairs, trainPairs);
	vector<double> trainData ;
	assert(trainPairs[0][0].size() == trainPairs[0][1].size());//asserts that it's really a pair of vectors of the same size
	size_t N = trainPairs[0][0].size()*2;
	toContinuousVector(trainPairs, trainData);
	std::vector<std::vector<double> > lGridpair = gridpair;
	lGridpair.insert(lGridpair .end(),  gridpair.begin(),  gridpair.end());
	ID_N(lGridpair,ID1_T);




	SgdSolver<ID_N> sgdSolver(ID_N(lGridpair,ID1_T));
	sgdSolver.train(
			  	N,//number of features
				trainData, // data example t starts at X[t*N]
				trainTags,
			 0.5
			  		);
	exit(1);

}
#endif

/////pegasos adapter end


void trainAndTest(const std::vector< vector <double> > & trainSamples,
		 const std::vector<std::vector<size_t> >& trainIndicesOfPairs,
			const std::vector<Label>& trainTags,
			const std::vector<std::vector<double> >& gridpair, double& tholdArg,
			const std::vector<vector<double> > & testSamples,
			const std::vector<std::vector<size_t> >& testIndicesOfPairs,
			const std::vector<Label>& testTags,
			const double C,
			const double lambda,
			bool doPrintW){

	if (useSinglesAdapter){
#ifdef PEGASOS_ADAPTER
	  trainAndTestSinglesAdapter(trainSamples,trainIndicesOfPairs,trainTags,gridpair,tholdArg,testSamples,testIndicesOfPairs,testTags,lambda,doPrintW);
#endif
		return;
	}
	time_t tstart, tend;
	tstart = time(0);


	SGDLearning * learning = createSGDLearning();
	Distance * distObj = createZeroOneDistance();
	//Distance * distObj = new L2Distance();

	vector<double> Wreg;
	if (distReg){
		Wreg = learning->constructWregDistance(gridpair, distObj);
	}else{
		Wreg = learning->construct_Wreg(gridpair);
	}
	std::vector<double> W = learning->run(trainSamples, trainIndicesOfPairs, trainTags, gridpair, Wreg, 0.5, 1,
			tholdArg);
	delete distObj;
	tend = time(0);
	cout << "It took " << difftime(tend, tstart) << " second(s)." << endl;
	if (doPrintW){
		//Flattened
		printFlattenedSquare(W);
	}



	Grid grid(gridpair);
	IDpair id_pair(grid);// is not being used as an input to SGD-init but created inside

	//gridpar vector include discrete_points twice, one for each hyper-axis, i.e., X and Y
	size_t numOfErrors = 0;
	size_t fn = 0; size_t fp = 0; size_t tn =0; size_t tp = 0;
	for (size_t i = 0; i < testIndicesOfPairs.size(); i++) {
		if (testIndicesOfPairs[i][0]>=testSamples.size() || testIndicesOfPairs[i][1]>=testSamples.size() ){
			cerr<<"Classification: index of example is out of bound"<<endl;
			exit(1);
		}
		vector<double> examp0 = testSamples[testIndicesOfPairs[i][0]];
		vector<double> examp1 = testSamples[testIndicesOfPairs[i][1]];

		std::vector<IndexValuePair> vol = id_pair(examp0, examp1);

		Label s = learning->classification(W, vol, tholdArg);

		
		if (s != testTags[i])
			numOfErrors++;

		if (s == -1 && testTags[i] ==1)
			fn++;
		if (s == 1 && testTags[i] ==-1)
			fp++;
		if (s == -1 && testTags[i] ==-1)
			tn++;
		if (s == 1 && testTags[i] ==1)
			tp++;
	}
	delete learning;

	//cout << "Number of errors: " << numOfErrors << endl;
	cout << "Error rate: " << (double) numOfErrors / testTags.size() <<  endl;
	cout << "Confusion Matrix: " << endl;
	cout << "[ " <<  tp << "\t," <<  fn << "\t]" <<endl;
	cout << "[ " <<  fp << "\t," <<  tn << "\t]" <<endl;
	//cout << "thold: " << thold << ".\n" << endl;

}



/////////////////////validate

typedef std::vector<size_t>  VElement;

//TODO iterator and compare both ways as a unit-test
void kFoldSplit(const std::vector<VElement> & data, std::vector<VElement> & dataTrain, std::vector<VElement> & dataTest,
		const std::vector<Label> label, std::vector<Label> & labelTrain, std::vector<Label> & labelTest,int k, int ind){
	assert(label.size()==data.size());
	size_t totalSize = data.size();
	size_t testSize = totalSize/k;

	size_t trainSize = totalSize-testSize;
	size_t testLocation = ind*testSize;
	dataTrain.resize(trainSize);
	labelTrain.resize(trainSize);
	dataTest.resize(testSize);
	labelTest.resize(testSize);
	size_t train_i=0;
	for (;train_i<testLocation;train_i++){
		dataTrain[train_i] = data[train_i];
		labelTrain[train_i] = label[train_i];
	}

	for (size_t i=0; i<testSize; i++){
		dataTest[i] = data[i+testLocation];
		labelTest[i] = label[i+testLocation];
	}
	for (;train_i<trainSize;train_i++){
		dataTrain[train_i] = data[train_i+testSize];
		labelTrain[train_i] = label[train_i+testSize];
	}
}


void validateAFold(
		const std::vector<VElement>& data, // data example t starts at X[t*N]
		const std::vector<Label>& label,
		int k, int ind,
		const std::vector<vector<double> > & examples,const std::vector<std::vector<double> >& gridpair,
		const double C,
		const double lambda,
		double& tholdArg, bool doPrintW
		)
{


				vector<VElement> train;
				vector<VElement> test;
				vector<VElement> dataTrain;
				vector<VElement> dataTest;
				vector<Label> labelTrain;
				vector<Label> labelTest;

				kFoldSplit(data, dataTrain, dataTest, label, labelTrain, labelTest, k, ind);


				trainAndTest( examples,
						dataTrain,
							labelTrain,
							gridpair, tholdArg,
							examples,
							dataTest,
							labelTest,
							C,
							lambda,
							doPrintW
						);
}


void coshuffle(std::vector<VElement> & data, std::vector<Label> & labels){

	//indices
	assert(data.size()==labels.size());
	std::vector<int> random_indices(data.size());
	std::iota(std::begin( random_indices), std::end( random_indices), 0); // Fill with 0, 1, ..., n.
	std::random_shuffle( random_indices.begin(), random_indices.end());

	vector<VElement> dataRes(data.size());
	vector<Label> labelsRes(labels.size());
	for (size_t i=0; i<data.size(); i++){
		//cout<<random_indices[i]<<","<<data.size()<<endl;
		dataRes[i] = data[random_indices[i]];
		labelsRes[i] = labels[random_indices[i]];
	}
	data = dataRes;
	labels = labelsRes;
}


void validate(
		std::vector<VElement> data, // data example t starts at X[t*N]
		std::vector<Label> labels,
		int k,
		const std::vector<vector<double> > & examples,const std::vector<std::vector<double> >& gridpair,

		const double C,
		const double lambda,
		double& tholdArg,
		bool doPrintW
		)
{
	bool shuffle = true;
	if (shuffle){
		coshuffle(data,labels);
	}
	for (int ind=0;ind<k;ind++){
	  cout<<"Running fold index: "<< ind <<endl;
	  validateAFold(data,labels,k,ind,examples,gridpair, C, lambda, tholdArg, doPrintW);
	}
}



////////////validate end













typedef vector<double> Element;
typedef vector<Element> pair;

vector<double> divideAxis(double maxVal, int partitionSize){
	vector<double > res;
	for (int i=0;i<partitionSize;i++){
		res.push_back(maxVal*((double)i/partitionSize));
	}
	return res;
}
void smallExample(){

	std::vector<Element >examples;
	std::vector<std::vector<size_t> > pairsIndices;
	std::vector<std::vector<double> > discrete_points;
	vector<Label> tags;
	int dim =1;
	bool useUnifiedGrid = true;
	if (useUnifiedGrid){//unified for all dimensions of the feature vectors

			vector<double> gridForX1 = divideAxis(255.5,8);
				//	{ 0, 31, 63, 64+31, 128, 128+31, 128+63, 128+64+31, 255};

			for (int d=0; d< dim; d++){
				discrete_points.push_back(gridForX1);
			}
		}

	double tholdArg = thresholdValue;
	examples = {{0},{1},{2}};
	pairsIndices = {{0,0}};
	tags={1};
	cout<<"a"<<endl;
	trainAndTest(examples, pairsIndices,tags,discrete_points, tholdArg, examples, pairsIndices,tags, lambda, C, false);
}

//#define STR(x) #x << '=' << x
//#define MD_PRINT_STR_WNAME(os, name, a) \
//    do { (os) << (name) << " is value " << (a) << ", in function "  <<  __FUNCTION__ << std::endl; } while(false)




void colorsMain(bool startWithBest){
	int gridSize = 5;
	
	  //cout<<"grid size: "<<gridSize<<endl;
	int kFoldConst = 4;
	//cout<<"TODO: check if file exists"<<endl;
	std::ifstream file("AnswersA.csv");

	typedef vector<double> Element;
	typedef vector<Element> pair;
	vector<pair> pairs;
	vector<Label> tags;
	for(CSVIterator loop(file); loop != CSVIterator(); ++loop)
	{

		vector<double> left = hex2RGB(removeQuotes((*loop)[2]));
		vector<double> right = hex2RGB(removeQuotes((*loop)[3]));
		Label tag01 = std::stoi(removeQuotes((*loop)[4]));
		Label tag = ((tag01==1) ? _same : _notsame); //copied from below sign
		//cout<<tag<<endl;
		//cout<<left<<endl;
		//cout<<right<<endl;
		pairs.push_back({left, right});
		tags.push_back(tag);
	}

	std::vector<Element >examples;
	std::vector<std::vector<size_t> > pairsIndices;
	pairsToSinglesAndIndicesRep(pairs, examples, pairsIndices);

	int dim =3;
	bool useUnifiedGrid = true;

	cout<<"Looping all Hyperparameters"<<endl;
	
	vector<int> gridSizeVec = {3,4,5,7,10,14,20};
	vec lambdaVec = {0.01,0.25,1,4,100};
	vec CVec = {0.01,0.25,1,4,100};	CVec *= 0.5;

	vector<int> sEF= {0,0,0};//startExperimentsFrom
	if (startWithBest){
		sEF= {2,2,2};
	}


	for (size_t indC=sEF[0]; indC<CVec.size(); indC++){	  
	  for (size_t indLambda=sEF[2]; indLambda<lambdaVec.size(); indLambda++){
	    for (size_t indGrid=sEF[1]; indGrid<gridSizeVec.size(); indGrid++){
				double C = CVec[indC];
				int gridSize = gridSizeVec[indGrid];
				double lambda = lambdaVec[indLambda];

				cout<<"====================================="<<endl;
				cout<<"Current configuration hyperparameters:"<<endl;
				MD_PRINT(C);MD_PRINT(gridSize);MD_PRINT(lambda);

				vector<double> gridForX1 = divideAxis(255.5,gridSize);
				std::vector<std::vector<double> > discrete_points;
				for (int d=0; d< dim; d++){
					discrete_points.push_back(gridForX1);
				}

				double tholdArg = thresholdValue;

				//smallExample();
				if (false){//if you want to check it when train and test are the same data
					trainAndTest(examples, pairsIndices,tags,discrete_points, tholdArg, examples, pairsIndices,tags, C, lambda, false);
				}
				validate(pairsIndices,tags,kFoldConst,examples,discrete_points, C, lambda, tholdArg,false);
			}
		}
	}


}

void multiLabelViaEquivMain(){
	arma::mat A;
	std::string name = "test.bin";

	A.load(name,arma::raw_binary);
	A.print("A");
}



void syntheticMain(Distance * distObject, bool sameTrainAndTest) {
	int dim = 1;//default value
	bool useJustFromGrid = false; //taking samples from the grid itself
	bool useUnifiedGrid =true;
	bool doPrintW = (dim == 1);
	thresholdValue = 40;
	//thresholdValue = 0;
	vec rep = thresholdValue * ones(dim,1);
	thresholdValue = sqrt(dot(rep,rep));
	cout<<"Dimension:"<<dim<<endl;
	cout<<"thresholdValue: "<<thresholdValue<<endl;


	if (useJustFromGrid && !useUnifiedGrid){
		cerr<<"sgdMain (useJustFromGrid && !useUnifiedGrid) not supported.";
		exit(1);
	}
	//const size_t numOfSamples = 5000000; //5M ~217sec
	const size_t numOfSamples = 500000; //500k ~22sec
	//const size_t numOfSamples = 250000;
	//const size_t numOfSamples = 50000; //50k ~2sec
	//const size_t numOfSamples = 500;

	std::vector<std::vector<double> > discrete_points;
	std::vector<double> gridForX1;
	if (useUnifiedGrid){//unified for all dimensions of the feature vectors

		gridForX1 =
				{ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };

		for (int d=0; d< dim; d++){
			discrete_points.push_back(gridForX1);
		}
	}
	else{//specify each grid separately

		gridForX1 =
						{ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };

		std::vector<double> gridForX2 = { 0, 200 };
		discrete_points.push_back(gridForX1);
		discrete_points.push_back(gridForX2);
		dim = discrete_points.size();
	}
	//std::vector<std::vector<double> > discrete_points = {{0,0},{0,200},{100,0},{100,200}};
	std::vector<vector<double> > examples;
	std::vector<std::vector<size_t> > indices_of_pairs;
	//arma_rng::set_seed_random();
	if (useJustFromGrid) {
		for (size_t j = 0; j < gridForX1.size(); j++) {
			vector<double> cur_vec;
			for (int d = 0; d < dim; d++)
				cur_vec.push_back(discrete_points[d][j]);
			examples.push_back( cur_vec );
		}
		for (size_t i = 0; i < gridForX1.size(); i++) {
			for (size_t j = 0; j < gridForX1.size(); j++) {
				vector<size_t> p1 = { i, j };
				indices_of_pairs.push_back(p1);
			}
		}

	} else {
		// loop for creating pairs
		for (int i = 0; i < 2; i++) {
			vector< vec > A;
			for (int d = 0; d < dim; d++)
				A.push_back(randi<vec>(numOfSamples / 2, distr_param(0, 100)));

			for (size_t j = 0; j < A[0].size(); j++) {
				vector<double> cur_vec;
				for (int d = 0; d < dim; d++)
					cur_vec.push_back( A[d](j) );
				examples.push_back(cur_vec);

			}

		}
		for (size_t i = 0; i < numOfSamples / 2; i++) {
			vector<size_t> cur_pair = { i, i + numOfSamples / 2 }; //0000001000,0000001000
			indices_of_pairs.push_back(cur_pair);
		}
	}

	vector<Label> tags(indices_of_pairs.size(), 0);
	// generate tags.
	int counter = 0;

	for (size_t i = 0; i < indices_of_pairs.size(); i++) {

		double dist = distObject->distance(examples[indices_of_pairs[i][0]],
				examples[indices_of_pairs[i][1]]);

		tags[i] = dist < thresholdValue ? _same : _notsame; //sign

		if (tags[i] > 0)
			counter++;
	}


	cout << "Number of good examples: " << counter << " out of: "
			<< indices_of_pairs.size() << " examples" << endl;

	double tholdArg = thresholdValue;		//argument for init, might not be changed at all in the case of threshold regularization

	std::vector<std::vector<double> > gridpair(discrete_points);

	//feel the interpolation:
	const double lambda = 0.00001;
	const double C = 500;

	//const double lambda = 1;
	//const double C = 0.5;

	if (sameTrainAndTest){//if you want to check it when train and test to be the same data
		trainAndTest(examples,indices_of_pairs,tags,gridpair, tholdArg,examples,indices_of_pairs,tags ,C ,lambda, doPrintW);
	}
	else{
		validate(indices_of_pairs,tags,4,examples,discrete_points,C, lambda, tholdArg,  doPrintW);

	}
}



int runAll(){


	smallExample();

	cout<<"after small example"<<endl;

	Distance * distObject2 =  createDistance("L2Distance");
	syntheticMain(distObject2,true);





	distReg = true;
	colorsMain(true);
}


int main() {
	distReg = true;
	colorsMain(true);
	distReg = false;
	runAll();
	exit(1);
	Distance * distObject2 =  createDistance("L2Distance");
	syntheticMain(distObject2,false);

	//smallExample();
	//exit(1);
	colorsMain(false);
	

	return 1;
}

