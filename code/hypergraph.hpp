#include "graph.h"
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <set>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <fstream>
#include <cstring>
#include <random>
#include <climits>
#include <omp.h>

using namespace std;

typedef pair<int,int> ii;
typedef pair<float,float> ff;
typedef vector<int> vi;
typedef vector<bool> vb;
typedef vector<float> vf;
typedef vector<ii> vii;
typedef vector<vi> vvi;
typedef vector<unordered_set<int>> vus;
typedef vector<vii> vvii;

struct CompareBySecond {
    bool operator()(pair<int, int> a, pair<int, int> b)
    {
        return a.second < b.second;
    }
};

struct item{
    int id;
    float gain;
    int iter;
};

struct CompareItems {
    bool operator()(const item& a, const item& b) {
        return a.gain < b.gain;  // 按照gain属性降序排序
    }
};

template<typename T>
T SumVector(vector<T>& vec)
{
    T res = 0;
    for (size_t i=0; i<vec.size(); i++)
    {
        res += vec[i];
    }
    return res;
}


// compute approximation of mitigation of S_M using 30K monte carlo simulations
double estimateMitigation(Graph &g, int n, vector<int> seeds, vector<double> &g_loss)
{
	double global_running_total = 0.0;
	long long int num = 500;
    omp_set_num_threads(12);
	#pragma omp parallel
	{
		vb visit(n,false);
		vb seed(n,false);
		vi visit_index(n,0);
		vi aw_length(n,-1);
		vi aw_close(n,0);
		vb adoption(n,false);
		vb fr_visit(n,false);
		vi fr_index(n,0);
		vb fake_reachable(n,false);
		vvi visit_neighbours(n);
		vvi parents(n);
		vvi parent_arrivals(n);
		vvi parent_permutation(n);
		priority_queue<ii, vii, greater<ii> > pq;

		for (unsigned int i = 0; i < seeds.size(); i++) {
			seed[seeds[i]] = true;
		}

		double running_total = 0;
		#pragma omp for
		for (int i = 0; i < num; i++) {
	    	running_total += g.generateMitigationSample(seeds, seed, visit, visit_index, aw_length, aw_close, adoption, fr_visit, fr_index, fake_reachable, visit_neighbours, parents, parent_arrivals, parent_permutation, pq, g_loss);
		}

		#pragma omp critical
		{
			global_running_total += running_total;
		}
	}

//    return running_total / num;
    for(unsigned int i = 0; i < g_loss.size(); i++){
        g_loss[i] /= num;
    }
	return global_running_total / num;
}

/*
* generate hyperedges in parallel following TCIC model
*/
void addHyperedgeParallel(Graph &g, HyperGraph &hg, long long num, bool sa_upper, bool ordinary)
{
	int numNodes = g.getNumNodes();
	bool empty;

	vvi hyperedges;
    vf rr_weight;
    vi rr_group;
    omp_set_num_threads(10);
    int cnt = 0;
	#pragma omp parallel 
	{
		ii root_data;   //随机选择的开始节点v
		int phase_one_traversal_data;   //保存访问，即错误信息影响了多少的节点，以及放问了但是没激活的节点
		ii phase_two_traversal_data;  //保存tie-breaking的节点数目
		vb phase_one_visit(numNodes,false); //是否被激活
		vb phase_two_visit(numNodes,false);
		vb phase_three_visit(numNodes,false);
		vi phase_one_visit_index(numNodes,0);
		vi phase_two_visit_index(numNodes,0);
		vi dist(numNodes,INT_MAX);
		vi delayed_dist(numNodes,INT_MAX);
		vi aw_length(numNodes,-1);
		vb overlap(numNodes,false);
		vi tb_nodes(numNodes,0);
		vi tb_index(numNodes,0);
		vb adopt_fake(numNodes,false);
		vb adopt_true(numNodes,false);
		vvi visit_neighbours(numNodes);
		vvi visit_neighbours_meet_len(numNodes);
		vvi parent_permutation(numNodes);
		vvi phase_three_parents(numNodes);
		vvi phase_three_parent_arrivals(numNodes);
		vi hyperedge;
		vvi hyperedges_priv;
		priority_queue<ii, vii, greater<ii> > pq;
        vf weights;
        vi group_ids;

		#pragma omp for
		for (int i = 0; i < num; i++) {
            // root_data.first is the id, the second is the distance
            //phaseOne is right, check once
//            cout << "走进了第一步\n";
            empty = hg.phaseOne(g, root_data, phase_one_traversal_data, pq, phase_one_visit, phase_one_visit_index,  dist, aw_length);
//            cout << "走出了第一步\n";
//            if(!empty){
//                cout << "phase one good\n";
//            }
	    	if (!empty) {
//                cout << "走进了第二步\n";
                if(!ordinary)
				    hg.phaseTwo(g, root_data, phase_two_traversal_data, pq, phase_one_visit, parent_permutation, phase_two_visit, phase_two_visit_index,
								delayed_dist, aw_length, overlap, tb_nodes, visit_neighbours, visit_neighbours_meet_len,  hyperedge, sa_upper);
                else
                    hg.phaseTwoOrdinary(g, root_data, phase_two_traversal_data, pq, phase_one_visit, parent_permutation, phase_two_visit, phase_two_visit_index,
                                delayed_dist, aw_length, overlap, tb_nodes, visit_neighbours, visit_neighbours_meet_len,  hyperedge, sa_upper);
//                cout << "走出了第二步\n";
                // phase_two_traversal_data.first is the number of tie-breaking
				if (phase_two_traversal_data.first > 0){
//                    cout << "走进了第三步\n";
                    hg.phaseThree(g, root_data.first, phase_two_traversal_data.first, pq, tb_nodes, aw_length, visit_neighbours, visit_neighbours_meet_len, parent_permutation,
                                  phase_three_parents, phase_three_parent_arrivals, adopt_fake, adopt_true, phase_three_visit, tb_index, hyperedge, sa_upper);
//                    cout << "走出了第三步\n";
                }
                #pragma omp critical
                {
                    hyperedges_priv.push_back(hyperedge);
                    weights.push_back(g.getNodeWeight_with_index(root_data.first));
                    group_ids.push_back(g.getNodeGroup_with_index(root_data.first));
                    if(hyperedge[0]!=root_data.first)
                        cout <<"Wrong\n";
                }
			}
//            cout << "开始重置\n";
			hg.reset(g, phase_one_traversal_data, phase_two_traversal_data, phase_one_visit_index, phase_one_visit, phase_two_visit_index, visit_neighbours,
							visit_neighbours_meet_len, parent_permutation, aw_length, hyperedge);
//            cout << "重置成功\n";
		}

		#pragma omp critical
        {
            cout << "cnt = " << cnt << endl;
            cnt += 1;
            hyperedges.insert(hyperedges.end(), hyperedges_priv.begin(), hyperedges_priv.end());
            rr_weight.insert(rr_weight.end(), weights.begin(), weights.end());
            rr_group.insert(rr_group.end(), group_ids.begin(), group_ids.end());
        }

	}

	for (unsigned int i = 0; i < hyperedges.size(); i++) {
		hg.addEdge(hyperedges[i], rr_weight[i], rr_group[i]);
	}
}


/*
* linear pass over coverages to find node with maximum marginal coverage
* also maintains top k marginals for computation of improved upper bound
*/
int getMaxIndex(int n, vector<float> &node_weight, vector<float> &k_max_mc) {
	int max_ind = -1;
	float max_cov = 0;

	for (int i = 0; i < n; i++) {
		if (node_weight[i] > max_cov) {
			max_ind = i;
			max_cov = node_weight[i];
		}
		if (node_weight[i] > k_max_mc[0]) {
			k_max_mc[0] = node_weight[i];
			sort(k_max_mc.begin(), k_max_mc.end());
		}
	}

	return max_ind;
}

/*
* greedy algorithm for weighted max cover over collection of RR sets w/ improved UB computation
*/
ff buildSeedSet(Graph &g, HyperGraph &hg, unsigned int n, unsigned int k, vector<int> &seeds, vector<float> &group_save_better)
{	
	unsigned int i, j;
	int  max_index, group_id;
    float coverage_weight, cur_cov_ub;
	vector<int > edge_list, node_list;
	vector<float> k_max_mc(k,0);
    int c = g.getGroupSize();
    vector<vector<float>> node_group_w(n,vector<float>(c,0));
	vector<float> node_weight(n,0);
	for (i = 0; i < n; i++) {
		node_weight[i] = hg.getNodeWeight(i);
        for(j = 0; j < c; j++)
            node_group_w[i][j] = hg.getNodeGroupWeight(i,j);
	}

	float cur_coverage = 0;
	float improved_cov_ub = 1.0 * 1e10;
	long long numEdge = hg.getNumEdge();

	// check if an edge is removed
	vector<bool> edge_removed(numEdge, false);
    vector<bool> nodeMark(n , true);
	
	unsigned int cur_seed = 0;
	// building each seed at a time
	while(cur_seed < k) {
        max_index = getMaxIndex(n, node_weight, k_max_mc);
        if (max_index == -1) break; // all sets have been covered

        cur_cov_ub = cur_coverage;
        for (i = 0; i < k; i++) {
            cur_cov_ub += k_max_mc[i];
            k_max_mc[i] = 0; // reset for next iteration
        }
        if (cur_cov_ub < improved_cov_ub) improved_cov_ub = cur_cov_ub;

        seeds.push_back(max_index);
        cur_coverage += node_weight[max_index];
//        if(abs(SumVector(node_group_w[max_index]) - node_weight[max_index]) > 1e-5){
//            cout << "Wrong in line 271\n";
//            cout << SumVector(node_group_w[max_index]) - node_weight[max_index] << endl;
//        }

        for (i = 0; i < c; i++){
            group_save_better[i] += max(node_group_w[max_index][i],0.0f);
            node_group_w[max_index][i] = 0;
        }
        node_weight[max_index] = 0;
        edge_list = hg.getNode(max_index);
        nodeMark[max_index] = false;
        /*
         * hg.getEdge to gain the hyperedge, edge_list[i] is the rr set index, hyperedge[0] is the random
         * begin node u, g.getNodeWeight_with_index(hg.getEdge(edge_list[i])[0]) get the rr set's weight
         */
        for (i = 0; i < edge_list.size(); i++) {
            if (edge_removed[edge_list[i]]) continue;
            node_list = hg.getEdge(edge_list[i]);
            coverage_weight = g.getNodeWeight_with_index(node_list[0]);
            group_id = g.getNodeGroup_with_index(node_list[0]);
            for (j = 0; j < node_list.size(); j++) {
                if (nodeMark[node_list[j]]){
                    node_weight[node_list[j]] -= coverage_weight;
                    node_group_w[node_list[j]][group_id] -= coverage_weight;
                }
            }
            edge_removed[edge_list[i]] = true;
		}
		cur_seed++;
	}

	getMaxIndex(n, node_weight, k_max_mc);
	cur_cov_ub = cur_coverage;
	for (i = 0; i < k; i++) {
		cur_cov_ub += k_max_mc[i];
	}
	if (cur_cov_ub < improved_cov_ub) improved_cov_ub = cur_cov_ub;

	return {improved_cov_ub, cur_coverage};
}
/*
* greedy algorithm for max cover over collection of RR sets for influence minimization
*/
float buildSeedSetIM(Graph &g, HyperGraph &hg, unsigned int n, unsigned int k, vector<int> &seeds)
{
    priority_queue<pair<int, int>, vector<pair<int, int>>, CompareBySecond>heap;
    vector<int>coverage(n, 0);

    for (int i = 0; i < n; i++)
    {
        pair<int, int>tep(make_pair(i, (int)hg.getNode(i).size()));
        heap.push(tep);
        coverage[i] = (int)hg.getNode(i).size();
    }
    int maxInd;

    long long influence = 0;
    long long numEdge = hg.getNumEdge();

    // check if an edge is removed
    vector<bool> edgeMark(numEdge, false);
    // check if an node is remained in the heap
    vector<bool> nodeMark(n , true);

    seeds.clear();
    while ((int)seeds.size()<k)
    {
        pair<int, int>ele = heap.top();
        heap.pop();
        if (ele.second > coverage[ele.first])
        {
            ele.second = coverage[ele.first];
            heap.push(ele);
            continue;
        }

        maxInd = ele.first;
        vector<int>e = hg.getNode(maxInd);  //the edge influence
        influence += coverage[maxInd];
        seeds.push_back(maxInd);
        nodeMark[maxInd] = false;

        for (unsigned int j = 0; j < e.size(); ++j){
            if (edgeMark[e[j]])continue;

            vector<int>nList = hg.getEdge(e[j]);
            for (unsigned int l = 0; l < nList.size(); ++l){
                if (nodeMark[nList[l]])coverage[nList[l]]--;
            }
            edgeMark[e[j]] = true;
        }
    }
    return 1.0*influence ;
}
void Saturate(Graph &g, HyperGraph &hg, vi &attrs, vf &mc, float &fairWeight, vi &sol, unsigned int k, vf &group_inf, float inf_f)
{
    int c = g.getGroupSize();
    int n = g.getNumNodes();
    vector<float> caps(c, 0);
    vector<int>temp_sol;

    double eps = 0.05, g_min = 0.0, g_max = 1.0;
    long long numEdge = hg.getNumEdge();

    while (g_min < (1 - eps) * g_max){
        vector<vector<float>>node_weight(n,vector<float>(c,0));
        unordered_map<int,bool>appear;
        bool is_cover = true;
        vector<vector<bool>> covs(c,vector<bool>(numEdge,false)); // first dimension is group
        vector<float>covs_weight(c,0);
        temp_sol.clear();
        double g_cur = (g_min + g_max) / 2;
        for(int j = 0; j < c; j++){
            caps[j] = max(1.0,g_cur * mc[j]);
        }
        priority_queue<item, vector<item>, CompareItems> pq;
        //// first node
        int max_id = -1;
        float max_gain = 0;
        for(int i = 0; i < n; i++){
            float gain = 0, gain_j;
            for(int j = 0; j < c; j++){
                gain_j = hg.getNodeGroupWeight(i,j);
                if (gain_j <= caps[j])
                    gain += gain_j;
                else
                    gain += caps[j];
            }
            if (gain > 0)
                pq.push({i, gain, 0}); // iteration用于加速，CELF算法同样如此，lazy Algorithm
            if (gain > max_gain){
                max_id = i;
                max_gain = gain;
            }
        }
        temp_sol.emplace_back(max_id);
        appear[max_id] = true;
        for(int i = 0; i < c; i++){
            covs_weight[i] = min(hg.getNodeGroupWeight(max_id,i),caps[i]);
            vector<int> eList = hg.getGroupNode(max_id,i);
            for (int u : eList){
                covs[i][u] = true;
            }
        }
        //// first node end
        //// later
        for (int it = 1; it < k; it++){
            max_id = -1;
            while(!pq.empty()){
                auto q_item = pq.top();
                pq.pop();
                if(!appear[q_item.id] && q_item.iter < it){
                    vector<float>temp_gain(c,0.0);
                    auto eList_for = hg.getNode(q_item.id);
                    for(int u : eList_for){
                        if(!covs[attrs[u]][u]){
                            temp_gain[attrs[u]] += g.getNodeWeight_with_index(hg.getEdge(u)[0]);
                        }
                    }
                    float gain = 0;
                    for (int j = 0; j < c; j++){
                        float ww = min(temp_gain[j], max(0.0f, caps[j] - covs_weight[j]));
                        gain += ww;
                        node_weight[q_item.id][j] = ww;
                    }
                    if (gain > 0)
                        pq.push({q_item.id, gain, it});
                }
                else if(!appear[q_item.id]){
                    max_id = q_item.id;
                    break;
                }
            }
            if (max_id >= 0){
                temp_sol.emplace_back(max_id);
                appear[max_id] = true;
                for(int i = 0; i < c; i++){
                    covs_weight[i] += node_weight[max_id][i];
                }
                auto eList  = hg.getNode(max_id);
                for(int u : eList){
                    covs[attrs[u]][u] = true;
                }
            }
            else
                break;
        }

        for (int j = 0; j < c; j++){
            if(covs_weight[j] < caps[j] - 0.05){
                is_cover = false;
                break;
            }
        }

        if (is_cover){
            cout << "now satisfied, gmin = " << g_cur << endl;
            g_min = g_cur;
            unsigned int len = temp_sol.size();
            sol.resize(len);
            for(int i = 0; i < len; i++)
                sol[i] = temp_sol[i];
            float min_w = 1e5 * 1.0;
            vf ratio(c, 0.0f);
            for(int i = 0; i < c; i++){
                ratio[i] = covs_weight[i] / numEdge * inf_f / group_inf[i];
//                ratio[i] = covs_weight[i] / numEdge * inf_f * (1-1/exp(1)) / group_inf[i];
            }
            for(int i = 0; i < c; i++){
                if(ratio[i] < min_w){
                    min_w = ratio[i];
                }
            }
            fairWeight = min_w;
            cout << "now satisfied, fairWeight = " << fairWeight << endl;
        }
        else
            g_max = g_cur;
    }
}
/*
* build seed set fairly
*/
ff buildSeedFair(Graph &g, HyperGraph &hg, unsigned int n, unsigned int k, vi &seeds, float alpha, vi &better, vf &group_loss, float inf_f, vf &group_save)
{
    long long numEdge = hg.getNumEdge();
    int c = g.getGroupSize();
    cout << "now iteration rr set number = " << numEdge << " and group size = " << c << endl;
    vector<int>attrs(numEdge,0);
    vector<float>mc(c,0);

    for(int i = 0; i < numEdge; i++){
        vector<int> nList = hg.getEdge(i);
        attrs[i] = g.getNodeGroup_with_index(nList[0]);
        mc[attrs[i]] += g.getNodeWeight_with_index(nList[0]);
    }
    vector<int> fairSol;
    float fairWeight;
    cout << "*****************************\n";
    for(int i = 0; i < c; i++){
        cout << mc[i] << " ";
    }
    cout << "\n*****************************\n";
    Saturate(g, hg, attrs, mc, fairWeight, fairSol, k, group_loss, inf_f);
    cout << "Saturate end\n";
    float opt_g = alpha * fairWeight;
    cout << "opt_g = " << opt_g << endl;
    vector<int>opt_sol;
    vector<vector<float>>node_weight(n,vector<float>(c,0));
    unordered_map<int,bool>show;
    vector<vector<bool>> covs(c,vector<bool>(numEdge,false)); // first dimension is group
    vector<float>covs_weight(c,0);
    priority_queue<item, vector<item>, CompareItems> pq;
    //// first node
    int max_id = -1;
    float max_gain = 0;
    for(int i = 0; i < n; i++){
        float gain = 0;
        for(int j = 0; j < c; j++){
            //* (1.0f - 1.0f/exp(1.0f))
            gain += min(1.0f, hg.getNodeGroupWeight(i,j) / numEdge * inf_f / group_loss[j]  / opt_g);
        }
        if (gain > 0)
            pq.push({i, gain, 0}); // iteration用于加速，CELF算法同样如此，lazy Algorithm
        if (gain > max_gain){
            max_id = i;
            max_gain = gain;
        }
    }
    cout << "The first node id = " << max_id << endl;
    show[max_id] = true;
    opt_sol.push_back(max_id);
    vf vals(c,0.0f);
    for(int i = 0; i < c; i++){
        covs_weight[i] = hg.getNodeGroupWeight(max_id,i);
        auto eList = hg.getGroupNode(max_id,i);
        for (int u : eList){
            covs[i][u] = true;
        }
    }
    for(int i = 0; i < c; i++){
        //* (1.0f - 1.0f/exp(1.0f))
        vals[i] = min(1.0f, covs_weight[i] / numEdge * inf_f / group_loss[i]  / opt_g);
    }
    for (int it = 1; it < k; it++){
        max_id = -1;
        while(!pq.empty()){
            auto q_item = pq.top();
            pq.pop();
            if(!show[q_item.id] && q_item.iter < it){
                vector<float>temp_gain(c,0.0);
                auto eList_for = hg.getNode(q_item.id);
                for(int u : eList_for){
                    if(!covs[attrs[u]][u]){
                        temp_gain[attrs[u]] += g.getNodeWeight_with_index(hg.getEdge(u)[0]);
                    }
                }
                float gain = 0;
                for (int j = 0; j < c; j++){
                    //* (1.0f - 1.0f/exp(1.0f))
                    float ww = min(temp_gain[j] / numEdge * inf_f / group_loss[j]  / opt_g, 1.0f - vals[j]);
                    gain += ww;
                    node_weight[q_item.id][j] = temp_gain[j];
                }
                if (gain > 1e-4)
                    pq.push({q_item.id, gain, it});
            }
            else if(!show[q_item.id]){
                max_id = q_item.id;
                break;
            }
        }
        if (max_id >= 0){
            opt_sol.push_back(max_id);
            show[max_id] = true;
            for(int i = 0; i < c; i++){
                covs_weight[i] += node_weight[max_id][i];
                //* (1.0f - 1.0f/exp(1.0f))
                vals[i] = min(1.0f, covs_weight[i] / numEdge * inf_f / group_loss[i]  / opt_g);
            }
            auto eList  = hg.getNode(max_id);
            for(int u : eList){
                covs[attrs[u]][u] = true;
            }
        }
        else
            break;
    }
    unsigned int fair_len = fairSol.size();
    cout << "The budget use to satisfied Saturate is " << fair_len << endl;
    if ( (SumVector(vals) / c < (1.0f - 1e-4f)) && opt_sol.size() == k){
        opt_sol.clear();
        for(unsigned int i = 0; i < fair_len; i++){
            opt_sol.emplace_back(fairSol[i]);
        }
    }
    unsigned int len = opt_sol.size();

    cout << "When alpha = " << alpha << " ,The budget use to satisfied opt fairness is " << len << endl;
    cout << "The total budget k = " << k  << "\n";
    if(len > k){
        cout << "Something Wrong\n";
        return {-1.0, opt_g};
    }
    else if(len < k){
//        cout << "len < k and seeds.size = " << len << endl;
        seeds.resize(len);
        for(int i = 0; i < len; i++){
            seeds[i] = opt_sol[i];
        }
        int it = 0;
        while(seeds.size() < k && it < k){
            if(!show[better[it]]){
//                cout << "enter while\n";
                show[better[it]] = true;
                seeds.push_back(better[it]);
            }
            it++;
        }
//        cout << "seeds.size = " << seeds.size() << endl;
    }
    else if(len == k){
        seeds.resize(len);
        for(int i = 0; i < k; i++){
            seeds[i] = opt_sol[i];
        }
    }

    float coverage_weight;
    int group_id;
    vector<float> node_w(n,0);
    vector<vector<float>> node_group_w(n,vector<float>(c,0));
    for (int i = 0; i < n; i++){
        node_w[i] = hg.getNodeWeight(i);
        for(int j = 0; j < c; j++){
            node_group_w[i][j] = hg.getNodeGroupWeight(i,j);
        }
    }

    vector<bool> edge_removed(numEdge, false);
    vector<bool> nodeMark(n, true);
    unsigned int cur_seed = 0;
    float coverage = 0;
    cout << seeds.size() << endl;
    while(cur_seed < seeds.size()) {
        coverage += node_w[seeds[cur_seed]];
        for(int i = 0; i < c; i++){
            group_save[i] += node_group_w[seeds[cur_seed]][i];
            node_group_w[seeds[cur_seed]][i] = 0;
        }
        node_w[seeds[cur_seed]] = 0;
        auto edge_list = hg.getNode(seeds[cur_seed]);
        nodeMark[seeds[cur_seed]] = false;
        for (int i = 0; i < edge_list.size(); i++) {
            if (edge_removed[edge_list[i]]) continue;
            auto node_list = hg.getEdge(edge_list[i]);
            coverage_weight = g.getNodeWeight_with_index(node_list[0]);
            group_id = g.getNodeGroup_with_index(node_list[0]);
            for (int j = 0; j < node_list.size(); j++) {
                if (nodeMark[node_list[j]]){
                    node_w[node_list[j]] -= coverage_weight;
                    node_group_w[node_list[j]][group_id] -= coverage_weight;
                }

            }
            edge_removed[edge_list[i]] = true;
        }
        cur_seed++;
    }
    return {coverage, opt_g};
}
/*
* linear pass over coverages to find node with maximum marginal coverage
*/
int getMaxIndexBaseline(int n, vector<int> &node_weight) {
	int max_ind = -1;
	int max_cov = 0;

	for (int i = 0; i < n; i++) {
		if (node_weight[i] > max_cov) {
			max_ind = i;
			max_cov = node_weight[i];
		}
	}

	return max_ind;
}

/*
* greedy algorithm for weighted max cover over collection of RDR sets w/o improved UB computation
*/
int buildSeedSetBaseline(Graph &g, HyperGraph &hg, unsigned int n, unsigned int k, vector<int> &seeds)
{	
	unsigned int i, j;
	int coverage_weight, max_index;
	vector<int > edge_list, node_list;

	vector<int> node_weight(n,0);
	for (i = 0; i < n; i++) {
		node_weight[i] = hg.getNodeWeight(i);
	}

	int coverage = 0;
	long long numEdge = hg.getNumEdge();

	// check if an edge is removed
	vector<bool> edge_removed(numEdge, false);
	
	unsigned int cur_seed = 0;
	// building each seed at a time
	while(cur_seed < k) {
		max_index = getMaxIndexBaseline(n, node_weight);
		if (max_index == -1) break; // all sets have been covered 
		seeds.push_back(max_index);
		coverage += node_weight[max_index];
		edge_list = hg.getNode(max_index);
        for (i = 0; i < edge_list.size(); i++) {
            if (edge_removed[edge_list[i]]) continue;
            /*
             * hg.getEdge to gain the hyperedge, edge_list[i] is the rr set index, hyperedge[0] is the random
             * begin node u, g.getNodeWeight_with_index(hg.getEdge(edge_list[i])[0]) get the rr set's weight
             */
            coverage_weight = g.getNodeWeight_with_index(hg.getEdge(edge_list[i])[0]);
            node_list = hg.getEdge(edge_list[i]);
            for (j = 0; j < node_list.size(); j++) {
                node_weight[node_list[j]] -= coverage_weight;
            }
            edge_removed[edge_list[i]] = true;
        }
		cur_seed++;
	}

	return coverage;
}
