/*
* Functionality: convert from a graph file in weighted edge list to two binary files that encode the graph and its transpose
* Syntax:
	./el2bin <graph file input> <binary graph output> <transpose binary graph output> 

* The graph file input must follow the following format:
	<number of nodes> <number of edges>
	<first node of edge 1> <second node of edge 1> <weight of edge 1>
	...
	<first node of the last edge> <second node of the last edge> <weight of the last edge>

* The binary graph outputs will be used for fast reading by the other algorithm

* Adapted from original implementation by Hung T. Nguyen (hungnt@vcu.edu)
* Author: Michael Simpson (mesimp@cs.ubc.ca)
*/

//// ../../datasets/Facebook/nm.txt ../../datasets/Facebook/network.txt ../../datasets/Facebook/network.bin ../../datasets/Facebook/networkrev.bin

#include <cstdio>
#include <fstream>
#include <cmath>
#include <cstring>
#include <vector>

using namespace std;

int main(int argc, char ** argv)
{
//    ifstream nm(argv[1]);
	ifstream in(argv[1]);
	int n,u,v;
	long long m;
	float w;
    in >> n >> m;
//    nm.close();
	printf("%d %lld\n", n, m);
	vector<int> degree(n,0);
	vector<int> rev_degree(n,0);
	vector<vector<int> > eList(n);
	vector<vector<int> > rev_eList(n);
	vector<vector<float> > weight(n);
	vector<vector<float> > rev_weight(n);

	printf("Reading the graph!\n");

	for (long long i = 0; i < m; i++){
		in >> u >> v >> w;
		degree[u]++;
		eList[u].push_back(v);
		weight[u].push_back(w);
		rev_degree[v]++;
		rev_eList[v].push_back(u);
		rev_weight[v].push_back(w);
	}
	
	in.close();

	FILE * pFile1;
	FILE * pFile2;
	pFile1 = fopen(argv[2],"wb");
	pFile2 = fopen(argv[3],"wb");
	fwrite(&n, sizeof(int), 1, pFile1);
	fwrite(&m, sizeof(long long), 1, pFile1);
	
	// Write node degrees
	fwrite(&degree[0], sizeof(int), n, pFile1);
	fwrite(&rev_degree[0], sizeof(int), n, pFile2);
	
	// Write neighbors
	printf("writing neighbours\n");
	for (int i = 0; i < n; i++){
		printf("forward edge %d: ", i);
		for (int j = 0; j < eList[i].size(); j++) {
			printf("%d ", eList[i][j]);
		}
		printf("\n");
		fwrite(&eList[i][0], sizeof(int), eList[i].size(), pFile1);
		printf("reverse edge %d: ", i);
		for (int j = 0; j < rev_eList[i].size(); j++) {
			printf("%d ", rev_eList[i][j]);
		}
		printf("\n");
		fwrite(&rev_eList[i][0], sizeof(int), rev_eList[i].size(), pFile2);
	}

	// Write weights
	for (int i = 0; i < n; i++){
		fwrite(&weight[i][0], sizeof(float), weight[i].size(), pFile1);
		fwrite(&rev_weight[i][0], sizeof(float), rev_weight[i].size(), pFile2);
	}

	fclose(pFile1);
	fclose(pFile2);
	printf("Done!\n");
	return 1;
}
