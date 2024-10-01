#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstdlib>

using namespace std;
//// ../dataset/facebook/nm.txt  ../dataset/facebook/graph.txt  ../dataset/facebook/net.txt 1 TR
////   ../dataset/facebook/graph.txt  ../dataset/facebook/net.txt 1
int main(int argc, char ** argv)
{
//    ifstream nm(argv[1]);
	ifstream in(argv[1]);
	unsigned int flag = atoi(argv[3]);
    string model(argv[4]);
	unsigned long long n,m,i;
    in >> n >> m;
//    nm.close();
	printf("%lld, %lld\n", n, m);
	vector<unsigned long long> node(n,0);
	vector<unsigned long long> v1, v2;
    float weight[3] = {0.1, 0.01, 0.001};
	v1.reserve(m);
	v2.reserve(m);
	
	unsigned long long t1,t2;
	in >> t1 >> t2;
	v1.push_back(t1);
	v2.push_back(t2);
	node[t2]++;
	if (flag == 0) node[t1]++;
	printf("Reading the graph!\n");
	for (i = 1; i < m; i++) {
		in >> t1 >> t2;
		node[t2]++;
		if (flag == 0) node[t1]++;

		v1.push_back(t1);
		v2.push_back(t2);
		if (i %100000 == 0) printf("%lld\n", i);
	}
	in.close();
	ofstream out(argv[2]);
	printf("Writing down to file!\n");
	out << n << " ";

	if (flag == 0) {
		out << v1.size()*2 << endl;
	} else{
		out << v1.size() << endl;
	}
    for (i = 0; i < v1.size(); i++) {
        if(model == "WC"){
            out << v1[i] << " " << v2[i] << " " << (float) 1.0/(float)node[v2[i]] << endl;
            if (flag == 0) out << v2[i] << " " << v1[i] << " " << (float) 1.0/(float)node[v1[i]] << endl;
        }
        else if(model == "TR"){
            int random_num = rand() % 3;
            out << v1[i] << " " << v2[i] << " " << weight[random_num] << endl;
            if (flag == 0) out << v2[i] << " " << v1[i] << " " << weight[random_num] << endl;
        }
        else if(model == "UN"){
            out << v1[i] << " " << v2[i] << " " << 0.03 << endl;
            if (flag == 0) out << v2[i] << " " << v1[i] << " " << 0.03 << endl;
        }
    }

	out.close();
}
