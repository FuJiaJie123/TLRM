//
// Created by jiajie on 2024/3/14.
//
#include "option.h"
#include "hypergraph.hpp"
#include "graph.h"
#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <omp.h>

using namespace std;

double computeThetaMax(int n, double epsilon, double epsilon_prime, double delta, double delta_prime, double mit_lower_bound, int k) {
    double Delta = delta - delta_prime;
    double log_n_choose_k = lgammal(n+1)-lgammal(k+1)-lgammal(n-k+1);
    double denom = epsilon * (1+epsilon_prime) - 2 * epsilon_prime * (1-1/exp(1));
    double numerator = 8 * n * (1-1/exp(1)) * (3+epsilon_prime) * (log(27.0 / (4 * Delta)) + log_n_choose_k);
    return numerator / (3 * mit_lower_bound * denom * denom);
}

/*
* read fake seeds
*/
float readFakeInfluence(const char* filename, vector<float> &group_loss, vector<float> &group_inf)
{
    float fi;
    ifstream in(filename);
    in >> fi;
    float inf;
    cout << "group loss [" << group_loss.size() << "] : ";
    for(int i = 0; i < group_loss.size(); i++){
        in >> inf;
        group_loss[i] = inf;
        cout << group_loss[i] << " " ;
    }
    cout << endl;
    cout << "group inf [" << group_inf.size() << "] : ";
    for(int i = 0; i < group_inf.size(); i++){
        in >> inf;
        group_inf[i] = inf;
        cout << group_inf[i] << " " ;
    }
    cout << endl;
    in.close();
    return fi;
}

double Estimate_IM(HyperGraph &hg, unsigned int n, long long num_edge, double inf_f, vector<int> &seeds) {
    unsigned int i, j;
    vector<int > edge_list, node_list;

    vector<int>coverage(n, 0);

    for (i = 0; i < n; i++)
    {
        coverage[i] = (int)hg.getNode(i).size();
    }

    vector<bool> edge_removed(num_edge, false);
    vector<bool> nodeMark(n , true);
    unsigned int cur_seed = 0;
    float influence = 0;

    while(cur_seed < seeds.size()) {
        influence += coverage[seeds[cur_seed]];
        edge_list = hg.getNode(seeds[cur_seed]);
        nodeMark[seeds[cur_seed]] = false;
        for (i = 0; i < edge_list.size(); i++) {
            if (edge_removed[edge_list[i]]) continue;
            node_list = hg.getEdge(edge_list[i]);
            for (j = 0; j < node_list.size(); j++) {
                if (nodeMark[node_list[j]])
                    coverage[node_list[j]]--;
            }
            edge_removed[edge_list[i]] = true;
        }
        cur_seed++;
    }
    return influence * inf_f / num_edge;
}

float better_fairness(Graph &g, HyperGraph &hg, unsigned int n, long long num_edge, double inf_f, vector<int> &seeds, vector<float> &group_loss) {
    unsigned int i, j;
    float coverage_weight;
    vector<int > edge_list, node_list;
    int c = g.getGroupSize();
    vector<vector<float>> node_group_w(n,vector<float>(c,0));
    vector<float> group_save(c,0.0f);
    vector<float> node_weight(n,0);
    for (i = 0; i < n; i++){
        for(j = 0; j < c; j++){
            node_group_w[i][j] = hg.getNodeGroupWeight(i,j);
        }
    }

    vector<bool> edge_removed(num_edge, false);
    vector<bool> nodeMark(n , true);
    unsigned int cur_seed = 0, group_id;

    while(cur_seed < seeds.size()) {
        for( i = 0; i < c; i++){
            group_save[i] += node_group_w[seeds[cur_seed]][i];
            node_group_w[seeds[cur_seed]][i] = 0;
        }
        edge_list = hg.getNode(seeds[cur_seed]);
        nodeMark[seeds[cur_seed]] = false;
        for (i = 0; i < edge_list.size(); i++) {
            if (edge_removed[edge_list[i]]) continue;
            node_list = hg.getEdge(edge_list[i]);
            coverage_weight = g.getNodeWeight_with_index(node_list[0]);
            group_id = g.getNodeGroup_with_index(node_list[0]);
            for (j = 0; j < node_list.size(); j++) {
                if (nodeMark[node_list[j]])
                    node_group_w[node_list[j]][group_id] -= coverage_weight;
            }
            edge_removed[edge_list[i]] = true;
        }
        cur_seed++;
    }
    float min_ratio = 1.0f, ratio;
    // * (1.0 - 1.0/exp(1))
    for(i = 0; i < c; i++){
        ratio = group_save[i] / num_edge * inf_f  / group_loss[i];
        if(ratio < min_ratio)
            min_ratio = ratio;
    }
    return min_ratio;
}

double computeLowerBound(Graph &g, HyperGraph &hg, unsigned int n, unsigned int k, long long num_edge, double epsilon, double delta, double inf_f, vector<int> &seeds, double &cov) {
    unsigned int i, j;
    float coverage_weight;
    vector<int > edge_list, node_list;

    vector<float> node_weight(n,0);
    for (i = 0; i < n; i++){
        node_weight[i] = hg.getNodeWeight(i);
    }

    vector<bool> edge_removed(num_edge, false);
    vector<bool> nodeMark(n , true);
    unsigned int cur_seed = 0;
    double coverage = 0;

    while(cur_seed < seeds.size()) {
        coverage += node_weight[seeds[cur_seed]];
        node_weight[seeds[cur_seed]] = 0;
        edge_list = hg.getNode(seeds[cur_seed]);
        nodeMark[seeds[cur_seed]] = false;
        for (i = 0; i < edge_list.size(); i++) {
            if (edge_removed[edge_list[i]]) continue;
            node_list = hg.getEdge(edge_list[i]);
            coverage_weight = g.getNodeWeight_with_index(node_list[0]);
            for (j = 0; j < node_list.size(); j++) {
                if (nodeMark[node_list[j]])
                    node_weight[node_list[j]] -= coverage_weight;
            }
            edge_removed[edge_list[i]] = true;
        }
        cur_seed++;
    }
//    cout << "Now Compute lower bound--------\n";
    double weighted_coverage = coverage * inf_f / n;
//    cout << "coverage = "<< coverage << " ,inf_f = " << inf_f << " ,coverage * inf_f / n = " << weighted_coverage << endl;
    cov = coverage / num_edge * inf_f;
    double term_one = log(1.0/delta);
//    cout << "term one = " << term_one << endl;
    double term_two = n / ((1+epsilon) * num_edge);
//    cout << "term two = " << term_two << endl;
    double root_one = sqrt(weighted_coverage + ((25.0/36.0) * term_one));
    double root_two = sqrt(term_one);
//    cout << "root one = " << root_one <<  ", root_two = " << root_two << endl;
    double term_three = root_one - root_two;
//    cout << "term three = " << term_three << endl;
    return (term_three * term_three - (term_one / 36)) * term_two;
}

double computeUpperBound(int n, int k, long long int num_edge, double epsilon, double delta, double inf_f, float cov_ub) {
//    float e = exp(1);
//    double weighted_coverage = cov_ub * inf_f /  n / ( 1 - 1/e );
    double weighted_coverage = cov_ub * inf_f /  n;
    double term_one = log(1.0/delta);
    double term_two = n / ((1-epsilon) * num_edge);
    double root_one = sqrt(weighted_coverage + term_one);
    double root_two = sqrt(term_one);
    double term_three = root_one + root_two;
    return term_three * term_three * term_two;
}


int main(int argc, char ** argv)
{
    srand(time(nullptr));
    bool time_generate_flag = false;

    OptionParser op(argc, argv);
    if (!op.validCheck()){
        printf("Parameters error, please check the readme.txt file for correct format!\n");
        return -1;
    }

    char * inFile = op.getPara("-i");
    if (inFile == nullptr){
        inFile = (char*)"network";
    }

    char * outFile = op.getPara("-o");
    if (outFile == nullptr){
        outFile = (char*)"results.txt";
    }

    char * fakeSeedsFile = op.getPara("-fakeseeds");
    if (fakeSeedsFile == nullptr){
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

    char * fakeInfFile = op.getPara("-fakeinf");
    if (fakeInfFile == NULL){
        fakeInfFile = (char*)"fake.inf";
    }

    char * tmp = op.getPara("-epsilon");
    float epsilon = 0.3;
    if (tmp != nullptr){
        epsilon = atof(tmp);
    }

    int aw = 30;
    tmp = op.getPara("-aw");
    if (tmp != NULL){
        aw = atoi(tmp);
    }

    float rp = 0.6;
    tmp = op.getPara("-rp");
    if (tmp != NULL){
        aw = atof(tmp);
    }

    int ml = 2;
    tmp = op.getPara("-ml");
    if (tmp != NULL){
        ml = atoi(tmp);
    }

    int k = 1;
    tmp = op.getPara("-k");
    if (tmp != nullptr){
        k = atoi(tmp);
    }

    float alpha = 0.8;
    tmp = op.getPara("-alpha");
    if (tmp != nullptr){
        alpha = atof(tmp);
    }

    int br = 200;
    tmp = op.getPara("-br");
    if (tmp != NULL){
        br = atoi(tmp);
    }

    bool ordinary = false;
    tmp = op.getPara("-ordinary");
    if (tmp != NULL){
        ordinary = atoi(tmp);
    }

    bool nub = false;
    tmp = op.getPara("-nub");
    if (tmp != NULL){
        nub = atoi(tmp);
    }

    bool ego = false;
    tmp = op.getPara("-ego");
    if (tmp != NULL){
        ego = atoi(tmp);
    }

    float ew = -1.0;
    tmp = op.getPara("-ew");
    if (tmp != nullptr){
        ew = atof(tmp);
    }
    bool fixed = (ew < 0.0) ? false : true;


    cout << "\n*******************" << endl;
    cout << "\tSTART" << endl;
    cout << "*******************\n" << endl;
//// read part begin
    Graph g(aw, ml, rp, br, ego);
    g.readGraph(inFile, fixed, ew);
    g.readFakeSeeds(fakeSeedsFile);
    g.readNodeWeight(nodeWeightFile);
    g.readNodeGroup(groupFile);
//// read part end

    int n = g.getNumNodes();
    cout << "n = " << n << endl;
    int c = g.getGroupSize();
    cout << "groupSize = " << c << endl;
    float maxWeight = g.getMaxWeight();
    cout << "maxWeight = " << maxWeight << endl;
    float delta = 1.0/n;
    tmp = op.getPara("-delta");
    if (tmp != NULL){
        delta = atof(tmp);
    }
    float precision = 1-1/exp(1);
    tmp = op.getPara("-precision");
    if (tmp != NULL){
        precision = atof(tmp);
    }

    double epsilon_prime = epsilon / 2.0;
    double delta_prime = delta / 9.0;
    double interval;
    double time_pre = 0.0;
    double start = omp_get_wtime();
    const vi &fs = g.getFakeSeeds();
    vector<float>group_loss(c,0.0f), group_inf(c,0.0f);
    double inf_f = readFakeInfluence(fakeInfFile, group_loss, group_inf);
    unsigned int GroupSize = g.getGroupSize();

    cout << "fake seed set ["<< fs.size()<<"]: ";
    for (unsigned int s = 0; s < g.getNumFakeSeeds(); s++) {
        cout << fs[s] << " ";

    }
    cout << endl;
    cout << "\nComputing mitigation lower bound." << endl;
    double mit_lower_bound = g.computeMitigationLowerBound(n, k);
    interval = omp_get_wtime()-start;
    time_pre += interval;
    cout << "Time to compute mitigation lower bound: " << interval << "s" << endl;
    cout << "k = " << k << "\tLB = " << mit_lower_bound << endl;
    double theta_max = computeThetaMax(n, epsilon, epsilon_prime, delta, delta_prime, mit_lower_bound, k);
//    double theta_nought = theta_max * (epsilon*epsilon) * mit_lower_bound / n * 1/4 * log(1.0/delta);
    double theta_nought = theta_max * (epsilon*epsilon) * mit_lower_bound / n;
    cout << "theta_max = " << theta_max << endl;
    cout << "theta_0 = " << theta_nought << endl;
    int i_max = ceil(log2(theta_max / theta_nought));
    cout << "i_max = " << i_max << endl;
    double delta_iter = (delta - delta_prime) / (3.0*i_max);
    start = omp_get_wtime();
    vector<int> seeds, better_last(k,0);
    long long int cur_samples;
    float best_lost_save, fair_lost_save_opt, fairness_ratio = 0.0f, unfairness_ratio = 0.0f;
    double lower_final = 0, upper_final = 0, cover_final = 0, coefficient_final = 0;
    vector<float> group_save_final(GroupSize, 0.0f);
    vector<float> group_save_better_final(GroupSize, 0.0f);
    int try_times;
    double better_time = 0.0;
    for(try_times = 0; try_times < 1; try_times++){
        cur_samples = (long long) theta_nought;
        HyperGraph coll_one(n, GroupSize);
        HyperGraph coll_two(n, GroupSize);
        vector<float> group_save_better(GroupSize, 0.0f);
        int i = 1;

        double lower;
        double upper;
        double cover;
        double coefficient;
        float ratio;
        ff cov_ub;
        cout << "\n*******************" << endl;
        cout << "LOWER BOUND FOR SA" << endl;
        cout << "*******************" << endl;
        addHyperedgeParallel(g, coll_one, cur_samples, false, ordinary);
        addHyperedgeParallel(g, coll_two, cur_samples, false, ordinary);
        while (true) {
            cout << "\nIteration " << i << endl;
            seeds.clear();
            for(int group = 0; group < GroupSize; group++){
                group_save_better[group] = 0.0f;
            }
            cov_ub = buildSeedSet(g, coll_one, n, k, seeds, group_save_better);

            lower = computeLowerBound(g, coll_two, n, k, cur_samples, epsilon_prime, delta_iter, inf_f, seeds, cover);
            cout << "lower is " << lower << endl;

            upper = computeUpperBound(n, k, cur_samples, epsilon_prime, delta_iter, inf_f, cov_ub.first);
            cout << "upper is " << upper << endl;
            coefficient = lower / upper;
            cout << "cof is " << coefficient << endl;
            if ((coefficient >= (precision - epsilon)) || (i == i_max)) {
                break;
            }
            cout << "Generating up to " << 2*cur_samples << " samples " << endl;
            addHyperedgeParallel(g, coll_one, cur_samples, false, ordinary);
            addHyperedgeParallel(g, coll_two, cur_samples, false, ordinary);
            cur_samples *= 2;
            i++;
        }
        vector<int>better(seeds.begin(),seeds.end());
        better_time += (omp_get_wtime() - start);
        seeds.clear();
        ff fair_cov;
        vector<float> group_save(GroupSize, 0.0f);
        fair_cov = buildSeedFair(g, coll_one, n, k, seeds, alpha, better, group_loss, inf_f, group_save) ;
        cout << "buildSeedFair finished, the seeds.size = " << seeds.size() << endl;
        if(fair_cov.first == -1.0f){
            cout << "WRONG WRONG WRONG\n";
            return -1;
        }
        lower = computeLowerBound(g, coll_two, n, k, cur_samples, epsilon_prime, delta_iter, inf_f, seeds, cover);
        upper = computeUpperBound(n, k, cur_samples, epsilon_prime, delta_iter, inf_f, fair_cov.first);
        ratio = better_fairness(g, coll_one, n, cur_samples, inf_f, better, group_loss);
        lower_final += lower;
        upper_final += upper;
        cover_final += cover;
        coefficient_final += coefficient;
        fairness_ratio += fair_cov.second;
        unfairness_ratio += ratio;
        if(ratio > fair_cov.second){
            best_lost_save += cov_ub.second / cur_samples * inf_f;
            fair_lost_save_opt += cov_ub.second / cur_samples * inf_f;
            for(i = 0; i < GroupSize; i++){
                group_save[i] = group_save_better[i];
            }
        }
        else{
            best_lost_save += cov_ub.second / cur_samples * inf_f;
            fair_lost_save_opt += fair_cov.first / cur_samples * inf_f;
        }
        for(i = 0; i < GroupSize; i++){
            group_save_final[i] += group_save[i];
            group_save_better_final[i] += group_save_better[i];
        }
        for(i = 0; i < k; i++)
            better_last[i] = better[i];
    }
    ofstream out(outFile);
    out << "now is the better set : \n";
    for (unsigned int s = 0; s < better_last.size(); s++) {
        out << better_last[s] << " ";
    }
    out << "\nnow is the fair set : \n";
    for (unsigned int s = 0; s < seeds.size(); s++) {
        out << seeds[s] << " ";
    }
    out << endl;
    out << "unfairness ratio is " << unfairness_ratio / (try_times * 1.0) << endl;
    out << "Maxmin ratio(fairness) is " << fairness_ratio / (try_times * 1.0) << endl;
    out << "Best lost save is " << best_lost_save / (try_times * 1.0) << " while fair lost save is " << fair_lost_save_opt / (try_times * 1.0) << ", the achieved ratio is " << fair_lost_save_opt / best_lost_save << endl;
    out << "RR set number = " << cur_samples << endl;
    out << "cover weight = " << cover_final / (try_times * 1.0) << endl;
    out << "lower bound  = " << lower_final / (try_times * 1.0) << endl;
    out << "upper bound  = " << upper_final / (try_times * 1.0) << endl;
    out << "cof bound  = " << coefficient_final / (try_times * 1.0) << endl;
    out << "The better time is " << better_time / (try_times * 1.0) << " s" << endl;
    out << "The total time is " << (omp_get_wtime() - start) / (try_times * 1.0) << " s" << endl;
    out << "Lost save to each group is :\n";
    for(int i = 0; i < GroupSize; i++){
        out << group_save_final[i] / (try_times * 1.0) / cur_samples * inf_f << " ";
        if( i == GroupSize -1 )
            out << endl;
    }
    out << "Better lost save to each group is :\n";
    for(int i = 0; i < GroupSize; i++){
        out << group_save_better_final[i] / (try_times * 1.0) / cur_samples * inf_f << " ";
        if( i == GroupSize -1 )
            out << endl;
    }
    out.close();
    cout << endl;
    cout << "************The fake seeds are belong to groups : \n";
    for(int i = 0; i < g.getNumFakeSeeds(); i++){
        cout << g.getNodeGroup_with_index(fs[i]) << " ";
    }
    cout << endl;
    cout << "\n*******************" << endl;
    cout << "\tALL DONE" << endl;
    cout << "*******************" << endl;
    return 0;
    //
    // ALL DONE
    //
}