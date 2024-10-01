#include "option.h"
#include "graph.h"
#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <fstream>
#include <cstring>
#include <random>
#include <omp.h>

using namespace std;



// compute eps-delta approx of INF_F using generalized SRA
float estimateFakeInfluenceParallel(Graph &g, int n, double epsilon_prime, double delta_prime, vector<float> &global_loss, vector<double> &group_inf) {
	int b = n - g.getNumFakeSeeds();
	double eps = epsilon_prime * (1 - (epsilon_prime * b)/((2 + 2.0/3.0 * epsilon_prime) * log(2.0 / delta_prime) * b));
	long long int gamma = (1 + epsilon_prime) * (2 + 2.0/3.0 * eps) * log(2.0 / delta_prime) * (1.0 / (eps*eps)) * b / 10;
	
	long long int counter = 0;
	long long int global_running_total = 0;
    int c = 100;
    int groupSize = g.getGroupSize();
    vector<int>temp_inf(groupSize,0);
    omp_set_num_threads(12);
	#pragma omp parallel
	{
		vector<bool> visit(n,false);
		vector<int> visit_index(n,0);
        vector<int>inf_vec(groupSize,0);
        vector<float>loss_vec(groupSize,0.0);
		long long int running_total;
		while (global_running_total < gamma) {
			running_total = 0;
			for (int i = 0; i < c; i++) {
                running_total += g.generateFakeInfluenceSample(visit, visit_index, loss_vec, inf_vec);
                for(int j = 0; j < groupSize; j++){
                    temp_inf[j] += inf_vec[j];
                    global_loss[j] += loss_vec[j];
                    inf_vec[j] = 0;
                    loss_vec[j] = 0.0f;
                }

			}
			#pragma omp critical
			{
				global_running_total += running_total;
				counter += c;
                cout << "counter = " << counter << endl;
			}
		}
	}
    for(int i = 0; i < g.getGroupSize(); i++){
        global_loss[i] /= counter;
        group_inf[i] = (float)temp_inf[i] / counter;
    }
//    return (float)global_running_total / c ;
	return (float)global_running_total / counter;
}

int main(int argc, char ** argv)
{
	srand(time(NULL));
	
	OptionParser op(argc, argv);
	if (!op.validCheck()){
		printf("Parameters error, please check the readme.txt file for correct format!\n");
		return -1;
	}

	char * inFile = op.getPara("-i");
	if (inFile == NULL){
		inFile = (char*)"network";
	}

	char * outFile = op.getPara("-o");
	if (outFile == NULL){
		outFile = (char*)"fake.inf";
	}

	char * fakeSeedsFile = op.getPara("-fakeseeds");
	if (fakeSeedsFile == NULL){
		fakeSeedsFile = (char*)"fake.seeds";
	}

    char * nodeWeightFile = op.getPara("-nw");
    if (nodeWeightFile == nullptr){
        nodeWeightFile = (char*)"lurker.txt";
    }

    char * groupFile = op.getPara("-group");
    if (groupFile == nullptr){
        groupFile = (char*)"attr.txt";
    }

	char * tmp = op.getPara("-epsilon");
	float epsilon = 0.3;
	if (tmp != NULL){
		epsilon = atof(tmp);
	}

	float ew = -1.0;
	tmp = op.getPara("-ew");
	if (tmp != NULL){
		ew = atof(tmp);
	}
	bool fixed = (ew < 0.0) ? false : true;

	Graph g(0, 0, 0.0, 0, false);
	g.readGraph(inFile, fixed, ew);
	g.readFakeSeeds(fakeSeedsFile);
    g.readNodeWeight(nodeWeightFile);
    g.readNodeGroup(groupFile);
	int n = g.getNumNodes();

	float delta = 1.0/n;
	tmp = op.getPara("-delta");
    if (tmp != NULL){
    	delta = atof(tmp);
    }

    cout << "\n*******************" << endl;
	cout << "\tSTART" << endl;
	cout << "*******************\n" << endl;

    double epsilon_prime = epsilon / 2.0;
    double delta_prime = delta / 9.0;

    double start = omp_get_wtime();
    int groupSize = g.getGroupSize();
    vector<float> loss(groupSize,0.0);
    vector<double> group_inf(groupSize,0.0);
    float inf_f = estimateFakeInfluenceParallel(g, n, epsilon_prime, delta_prime, loss, group_inf);
    cout << "Time to estimate INF_F: " << omp_get_wtime()-start << "s" << endl;
    cout << "INF_F = " << inf_f << endl;

    cout << "fake seed set: ";
    const vector<int> &fs = g.getFakeSeeds();
	for (unsigned int s = 0; s < g.getNumFakeSeeds(); s++) {
		cout << fs[s] << " ";
	}
	cout << endl;

	ofstream out(outFile);
	out << inf_f << endl;
    for(int i = 0; i < groupSize; i++){
        out << loss[i] << " ";
    }
    out << endl;
    for(int i = 0; i < groupSize; i++){
        out << group_inf[i] << " ";
    }
    out << endl;
	out.close();

	cout << "\n*******************" << endl;
	cout << "\tALL DONE" << endl;
	cout << "*******************" << endl;

	//
	// ALL DONE
	//
}