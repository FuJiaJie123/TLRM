#include "graph.h"
#include <algorithm>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <queue>
#include <cstdlib>
#include <sstream>
#include <climits>
#include <cmath>
#include <unistd.h>

using namespace std;

typedef pair<int,int> ii;
typedef vector<int> vi;
typedef vector<bool> vb;
typedef vector<float> vf;
typedef vector<double> vd;
typedef vector<ii> vii;
typedef vector<vi> vvi;
typedef vector<vvi> vvvi;
typedef vector<vf> vvf;

Graph::Graph(int activation_window, int meeting_length, float reading_prob, int base_rate, bool ego_centric)
{
    sfmt_init_gen_rand(&sfmt_seed, rand());
    aw = activation_window;
    ml = meeting_length;
    rp = reading_prob;
    br = base_rate;
    ego = ego_centric;
}

const vector<int> & Graph::getOutNeighbours (int u) const
{
    return adj_list[u];
}

const vector<int> & Graph::getInNeighbours (int u) const
{
    return rev_adj_list[u];
}

const vector<float> & Graph::getOutWeights (int u) const
{
    return weights[u];
}

const vector<float> & Graph::getInWeights (int u) const
{
    return rev_weights[u];
}

const vector<int> & Graph::getFakeSeeds () const
{
    return fake_seeds_index;
}

unsigned int Graph::getOutDegree(int u) const
{
    return adj_list[u].size();
}

unsigned int Graph::getInDegree(int u) const
{
    return rev_adj_list[u].size();
}

/*
* get the number of nodes
*/
unsigned int Graph::getNumNodes() const
{
    return num_nodes;
}

/*
* get the number of edges
*/
unsigned int Graph::getNumEdges() const
{
    return num_edges;
}

/*
* get the number of fake seeds
*/
unsigned int Graph::getNumFakeSeeds() const
{
    return num_fake_seeds;
}

const float & Graph::getNodeWeight_with_index (int index) const
{
    return node_weight[index];
}

const vector<float> & Graph::getNodeWeight () const
{
    return node_weight;
}

const vector<int> & Graph::getNodeGroup () const
{
    return node_group;
}

const int & Graph::getNodeGroup_with_index (int index) const
{
    return node_group[index];
}
/*
* determine if node v is a fake seed
*/
bool Graph::isFakeSeed(int v) const
{
    return fake_seed[v];
}

/*
* read input graph
*/
void Graph::readGraph(const char* filename, bool fixed, float w)
{
    FILE * pFile1;
    FILE * pFile2;
    string filename1 = filename;
    filename1.append(".bin");
    string filename2 = filename;
    filename2.append("_rev.bin");
    pFile1 = fopen(filename1.c_str(), "rb");
    pFile2 = fopen(filename2.c_str(), "rb");
    fread(&num_nodes, sizeof(int), 1, pFile1);
    fread(&num_edges, sizeof(long long), 1, pFile1);
    node_deg = vi(num_nodes);
    fread(&node_deg[0], sizeof(int), num_nodes, pFile1);
    rev_node_deg = vi(num_nodes);
    fread(&rev_node_deg[0], sizeof(int), num_nodes, pFile2);

    for (unsigned int i = 0; i < num_nodes; i++){
        vi tmp1(node_deg[i]);
        fread(&tmp1[0], sizeof(int), node_deg[i], pFile1);
        adj_list.push_back(tmp1);

        vi tmp2(rev_node_deg[i]);
        fread(&tmp2[0], sizeof(int), rev_node_deg[i], pFile2);
        rev_adj_list.push_back(tmp2);
    }

    for (unsigned int i = 0; i < num_nodes; i++){
        vf tmp1(node_deg[i], w);
        if (!fixed) fread(&tmp1[0], sizeof(float), node_deg[i], pFile1);
        weights.push_back(tmp1);

        vf tmp2(rev_node_deg[i], w);
        if (!fixed) fread(&tmp2[0], sizeof(float), rev_node_deg[i], pFile2);
        rev_weights.push_back(tmp2);
    }
}

/*
* read fake seeds
*/
void Graph::readFakeSeeds(const char* filename)
{
    int fs;
    ifstream in(filename);
    in >> num_fake_seeds;
    fake_seeds_index = vi(num_fake_seeds);
    fake_seed = vb(getNumNodes(), false);
    for (unsigned int i = 0; i < num_fake_seeds; i++){
        in >> fs;
        fake_seeds_index[i] = fs;
        fake_seed[fs] = true;
    }
    in.close();
}

void Graph::readNodeWeight(const char* filename)
{
    int user_id;
    ifstream in(filename);
    float w;
    float maxWeight = 0.0f;
    for (unsigned int i = 0; i < num_nodes; i++){
        in >> user_id >> w;
        if (w > maxWeight){
            maxWeight = w;
        }
//        cout << w << endl;
        node_weight.emplace_back(w);
    }
    in.close();
    max_weight = maxWeight;
}

void Graph::setNodeWeight()
{
    int user_id;
    for (unsigned int i = 0; i < num_nodes; i++){
        node_weight.emplace_back(1.0f);
    }
    max_weight = 1.0f;
}

void Graph::readNodeGroup(const char* filename)
{
    ifstream in(filename);
    int number, group_size;
    in >> number >> group_size;
    int uid,group;
    node_group.resize(number);
    group_contain.resize(group_size);
    for (unsigned int i = 0; i < number; i++){
        in >> uid >> group;
        node_group[uid] = group;
        group_contain[group].emplace_back(uid);
    }
    in.close();
}

unsigned int Graph::getGroupSize() const
{
    return group_contain.size();
}

float Graph::getMaxWeight() const
{
    return max_weight;
}
/*
* generate a random activation window parameterized by rp & aw
*   first flip a coin to determine if sharing is reactionary (aw = 0)
*   if thoughtful --> generate geometric RV parameterized by aw
*/
int Graph::getActivationWindow()
{
    return 0;
//    float unif_one = sfmt_genrand_uint32(&sfmt_seed)/(float)(UI_MAX);
//    if (unif_one < rp) {
//        return 0;
//    } else {
//        float unif_two = sfmt_genrand_uint32(&sfmt_seed)/(float)(UI_MAX);
//        int geo = 1 + (int)( log(unif_two) / log( (aw - 1.0)/aw ) );
//        return (geo >= 0) ? geo / br : 0;
//    }
}

/*
* generate a random meeting length from geometric RV parameterized by ml
*/
int Graph::generateMeetingLength(int u)
{
    float m;
    if (ego) {
        m = (node_deg[u] + 5.0) / 5.0;
    } else {
        if (ml == 1) return 1;
        m = ml;
    }
    float unif = sfmt_genrand_uint32(&sfmt_seed)/(float)(UI_MAX);
    int val = log(unif) / log( (m - 1.0)/m );
    return (val >= 0) ? 1 + val : 1;
}

// compute mitigation lower bound based on top k nodes from a depth 1 MIA
double Graph::computeMitigationLowerBound(unsigned int n, unsigned int k)
{
    int i, j;
    int num_seen = 0;
    vi seen_index(n,0);
    vf ap(n,0);

    int num_fs = getNumFakeSeeds();
    const vi &fs = getFakeSeeds();
    for (i = 0; i < num_fs; i++) {
        const vf &w = getOutWeights(fs[i]);
        const vi &neigh = getOutNeighbours(fs[i]);
        for (j = 0; j < node_deg[fs[i]]; j++) {
//            if (ap[neigh[j]] < w[j]) {
                if (ap[neigh[j]] == 0) seen_index[num_seen++] = neigh[j];
                ap[neigh[j]] += w[j];
//            }
        }
    }

    vf sorted_ap;
    for (i = 0; i < num_seen; i++) {
        sorted_ap.push_back(ap[seen_index[i]]);
    }
    sort(sorted_ap.begin(), sorted_ap.end(), greater<int>());

    double sum = 0.0;
    int len = (sorted_ap.size() < k) ? sorted_ap.size() : k;
    for (i = 0; i < len; i++) {
        sum += sorted_ap[i];
    }
    float max_nw = getMaxWeight();
    return sum * max_nw;
}

// generate a single forward monte carlo estimate of the influence of input seed
int Graph::generateInfluenceSample(vb &visit, vi &visit_index, int root)
{
    int i, cur;
    float flip;

    int curPos = 0;
    int num_marked = 1;
    visit[root] = true;
    visit_index[0] = root;

    while(curPos < num_marked) {
        cur = visit_index[curPos];
        const vf &w = getOutWeights(cur);
        const vi &neigh = getOutNeighbours(cur);
        for (i = 0; i < node_deg[cur]; i++) {
            flip = sfmt_genrand_uint32(&sfmt_seed) / (float)UI_MAX;
            if (flip < w[i]) {
                if (!visit[neigh[i]]) {
                    visit[neigh[i]] = true;
                    visit_index[num_marked] = neigh[i];
                    num_marked++;
                }
            }
        }
        curPos++;
    }

    for(i = 0; i < num_marked; i++) {
        visit[visit_index[i]] = false;
    }

    return num_marked;
}

// generate a single forward monte carlo estimate of the outward influence of F
int Graph::generateFakeInfluenceSample(vb &visit, vi &visit_index, vf &loss, vi &g_inf )
{
    int i, cur;
    float flip;
    int curPos = 0;
    int num_marked = getNumFakeSeeds();

    const vi &fs = getFakeSeeds();
    for (i = 0; i < num_marked; i++) {
        visit[fs[i]] = true;
        visit_index[i] = fs[i];
    }

    while(curPos < num_marked) {
        cur = visit_index[curPos];
        const vf &w = getOutWeights(cur);
        const vi &neigh = getOutNeighbours(cur);
        for (i = 0; i < node_deg[cur]; i++) {
            flip = sfmt_genrand_uint32(&sfmt_seed) / (float)UI_MAX;
            if (flip < w[i]) {
                if (!visit[neigh[i]]) {
                    int g_id = node_group[neigh[i]];
                    float g_loss = node_weight[neigh[i]];
                    loss[g_id] += g_loss;
                    g_inf[g_id]++;
                    visit[neigh[i]] = true;
                    visit_index[num_marked] = neigh[i];
                    num_marked++;
                }
            }
        }
        curPos++;
    }

    for(i = 0; i < num_marked; i++) {
        visit[visit_index[i]] = false;
    }


    return num_marked - getNumFakeSeeds();
}

// generate a single forward monte carlo estimate of the mitigation of S_M
double Graph::generateMitigationSample(vi &seeds, vb &seed, vb &visit, vi &visit_index, vi &aw_length, vi &aw_close, vb &adoption, vb &fr_visit, vi &fr_index, vb &fake_reachable, vvi &visit_neighbours, vvi &parents, vvi &parent_arrivals,
    vvi &parent_permutation, priority_queue<ii, vii, greater<ii> > &pq, vd &g_loss)
{
    int cur, count, permutation_node, meet_len, new_close;
    unsigned int i, j, rand_pos;
    float flip;
    bool fake_reached, found;
    int num_marked = 0;
    queue<int>fs_pq;
    unsigned int num_fs = getNumFakeSeeds();
    const vi &fs = getFakeSeeds();
    for (i = 0; i < num_fs; i++) {
        const vf &w = getOutWeights(fs[i]);
        const vi &neigh = getOutNeighbours(fs[i]);
        for (j = 0; j < node_deg[fs[i]]; j++) {
            if (fake_seed[neigh[j]]) continue; // do not consider in-neighbours of nodes in S_F
            flip = sfmt_genrand_uint32(&sfmt_seed) / (float)UI_MAX;
            if (flip < w[j]) {
                visit_neighbours[fs[i]].push_back(neigh[j]);
                if (seed[neigh[j]]) continue; // do not consider in-neighbours of nodes in S_M
                if (aw_length[neigh[j]] < 0) aw_length[neigh[j]] = getActivationWindow();
                pq.push(make_pair(aw_length[neigh[j]]+1, neigh[j]));
                fs_pq.push(neigh[j]);
                parents[neigh[j]].push_back(fs[i]);
                parent_arrivals[neigh[j]].push_back(1);
                rand_pos = sfmt_genrand_uint32(&sfmt_seed)%(parent_permutation[neigh[j]].size() + 1);
                if (rand_pos == parent_permutation[neigh[j]].size()) {
                    parent_permutation[neigh[j]].push_back(fs[i]);
                } else {
                    parent_permutation[neigh[j]].push_back(parent_permutation[neigh[j]][rand_pos]);
                    parent_permutation[neigh[j]][rand_pos] = fs[i];
                }
            }
        }
    }

    for (i = 0; i < seeds.size(); i++) {
        adoption[seeds[i]] = true;
        const vf &w = getOutWeights(seeds[i]);
        const vi &neigh = getOutNeighbours(seeds[i]);
        for (j = 0; j < node_deg[seeds[i]]; j++) {
            if (fake_seed[neigh[j]]) continue; // do not consider in-neighbours of nodes in S_F
            flip = sfmt_genrand_uint32(&sfmt_seed) / (float)UI_MAX;
            if (flip < w[j]) {
                visit_neighbours[seeds[i]].push_back(neigh[j]);
                if (seed[neigh[j]]) continue; // do not consider in-neighbours of nodes in S_M
                meet_len = generateMeetingLength(seeds[i]);
                if (aw_length[neigh[j]] < 0) aw_length[neigh[j]] = getActivationWindow();
                pq.push(make_pair(aw_length[neigh[j]] + meet_len, neigh[j]));
                parents[neigh[j]].push_back(seeds[i]);
                parent_arrivals[neigh[j]].push_back(meet_len);
                rand_pos = sfmt_genrand_uint32(&sfmt_seed)%(parent_permutation[neigh[j]].size() + 1);
                if (rand_pos == parent_permutation[neigh[j]].size()) {
                    parent_permutation[neigh[j]].push_back(seeds[i]);
                } else {
                    parent_permutation[neigh[j]].push_back(parent_permutation[neigh[j]][rand_pos]);
                    parent_permutation[neigh[j]][rand_pos] = seeds[i];
                }
            }
        }
    }

    while (!pq.empty()) {
        cur = pq.top().second; 
        aw_close[cur] = pq.top().first; pq.pop();
        if (visit[cur]) continue; // duplicate entry in pq
        visit[cur] = true;
        visit_index[num_marked] = cur;
        num_marked++;

        // resolve adoption of cur via parent permutation
        count = 0;
        found = false;
        while(!found && count < parent_permutation[cur].size()) {
            permutation_node = parent_permutation[cur][count];
            for (i = 0; i < parents[cur].size(); i++) {
                if (parents[cur][i] == permutation_node && parent_arrivals[cur][i] <= aw_close[cur]) {
                    adoption[cur] = adoption[parents[cur][i]];
                    found = true;
                    break;
                }
            }
            count++;
        }

        const vf &w = getOutWeights(cur);
        const vi &neigh = getOutNeighbours(cur);
        for (i = 0; i < node_deg[cur]; i++) {
            if (fake_seed[neigh[i]]) continue; // do not consider in-neighbours of nodes in S_F
            flip = sfmt_genrand_uint32(&sfmt_seed) / (float)UI_MAX;
            if (flip < w[i]) {
                visit_neighbours[cur].push_back(neigh[i]);
                if (seed[neigh[i]]) continue; // do not consider in-neighbours of nodes in S_M
                if (!visit[neigh[i]]) {
                    meet_len = (adoption[cur]) ? generateMeetingLength(cur) : 1.0;
                    if (aw_length[neigh[i]] < 0) aw_length[neigh[i]] = getActivationWindow();
                    new_close = aw_close[cur] + meet_len + aw_length[neigh[i]];
                    pq.push(make_pair(new_close,neigh[i]));
                    parents[neigh[i]].push_back(cur);
                    parent_arrivals[neigh[i]].push_back(aw_close[cur] + meet_len);
                    rand_pos = sfmt_genrand_uint32(&sfmt_seed)%(parent_permutation[neigh[i]].size() + 1);
                    if (rand_pos == parent_permutation[neigh[i]].size()) {
                        parent_permutation[neigh[i]].push_back(cur);
                    } else {
                        parent_permutation[neigh[i]].push_back(parent_permutation[neigh[i]][rand_pos]);
                        parent_permutation[neigh[i]][rand_pos] = cur;
                    }
                }
            }
        }
    }

    // determine which nodes would have been reached by F
    int curPos = 0;
    int num_marked_fr = num_fs;
    for (i = 0; i < num_fs; i++) {
        fr_index[i] = fs[i];
        // my code
        fake_reachable[fs[i]] = true;
        fr_visit[fs[i]] = true;
        // my code
    }
    while(!fs_pq.empty()){
        int cur = fs_pq.front();
        fs_pq.pop();
        if(fr_visit[cur])
            continue;
        fr_index[num_marked_fr] = cur;
        fake_reachable[cur] = true;
        fr_visit[cur] = true;
        num_marked_fr++;
        const vf &w = getOutWeights(cur);
        const vi &neigh = getOutNeighbours(cur);
        for (j = 0; j < node_deg[cur]; j++) {
            if (fake_seed[neigh[j]]) continue; // do not consider in-neighbours of nodes in S_F
            flip = sfmt_genrand_uint32(&sfmt_seed) / (float)UI_MAX;
            if (flip < w[j]) {
                fs_pq.push(neigh[j]);
            }
        }
    }
//    while(curPos < num_marked_fr) {
//        cur = fr_index[curPos];
//        fake_reachable[cur] = true;
//        const vi &neigh = visit_neighbours[cur];
//        for (i = 0; i < neigh.size(); i++) {
//            if (fr_visit[neigh[i]]) continue;
//            fr_visit[neigh[i]] = true;
//            fr_index[num_marked_fr] = neigh[i];
//            num_marked_fr++;
//        }
//        curPos++;
//    }
    int n = getNumNodes();
    vector<bool> show(n,false);
    double reward = 0;
    for(i = 0; i < num_marked; i++) {
        cur = visit_index[i];
        if ( !fake_reachable[cur] || fake_seed[cur] || seed[cur]) continue;
        if (adoption[cur] && !show[cur]) { // cur adopts M
            show[cur] = true;
            reward += getNodeWeight_with_index(cur);
            int g_id = getNodeGroup_with_index(cur);
            g_loss[g_id] += getNodeWeight_with_index(cur);
        }
    }
    
    // add reawrd from seeds and reset data structures
    for(i = 0; i < seeds.size(); i++) {
        cur = seeds[i];
        if (fake_reachable[cur]) {
            reward += getNodeWeight_with_index(cur);
        }
        adoption[cur] = false;
        fake_reachable[cur] = false;
        vi().swap(visit_neighbours[cur]);
    }

     //reset data structures
    for (i = 0; i < num_marked; i++) {
        cur = visit_index[i];
        visit[cur] = false;
        aw_length[cur] = -1;
        aw_close[cur] = 0;
        adoption[cur] = false;
        vi().swap(parents[cur]);
        vi().swap(parent_arrivals[cur]);
        vi().swap(parent_permutation[cur]);
        vi().swap(visit_neighbours[cur]);
    }
//    for (i = 0; i < n; i++) {
//        visit[i] = false;
//        visit_index[i] = 0;
//        aw_length[i] = -1;
//        aw_close[i] = 0;
//        adoption[i] = false;
//        vi().swap(parents[i]);
//        vi().swap(parent_arrivals[i]);
//        vi().swap(parent_permutation[i]);
//        vi().swap(visit_neighbours[i]);
//        fr_visit[i] = false;
//        fr_index[i] = 0;
//        fake_reachable[i] = false;
//    }
    while(!pq.empty())
        pq.pop();
    for (i = 0; i < num_marked_fr; i++) {
        fr_visit[fr_index[i]] = false;
        fake_reachable[fr_index[i]] = false;
    }

    for (i = 0; i < num_fs; i++) {
        vi().swap(visit_neighbours[fs[i]]);
    }

    return reward;
}

HyperGraph::HyperGraph(unsigned int n, unsigned int c)
{
    sfmt_init_gen_rand(&sfmtSeed, rand());
    node_hyperedges = vvi(n);
    node_hyperedge_weight = vf(n);
    node_hyperedge_weight_group = vvf(n,vf(c));
    node_hyperedges_group = vvvi(n,vvi(c));
}

/*
* Add a hyperedge into the hypergraph
*/
void HyperGraph::addEdge(vi &edge, float w, int groupId)
{
    hyperedges.push_back(edge);
    unsigned int index = hyperedges.size() - 1;
    for (unsigned int i = 0; i < edge.size(); i++) {
        node_hyperedges[edge[i]].push_back(index);
        node_hyperedges_group[edge[i]][groupId].push_back(index);
        node_hyperedge_weight[edge[i]] += w;
        node_hyperedge_weight_group[edge[i]][groupId] += w;
    }
}

/*
* get an edge from the hypergraph
*/
const vector<int > & HyperGraph::getEdge(int e) const
{
    return hyperedges[e];
}

/*
* get an edge from the hypergraph_with_group
*/
const vector<int > & HyperGraph::getGroupNode(int e, int group) const
{
    return node_hyperedges_group[e][group];
}

/*
* get the list of hyperedges incident to node u
*/
const vector<int > & HyperGraph::getNode(int u) const
{
    return node_hyperedges[u];
}

/*
* get the weight covered by node u
*/
float HyperGraph::getNodeWeight(int u) const
{
    return node_hyperedge_weight[u];
}

/*
* get the weight covered by node u in group g
*/
float HyperGraph::getNodeGroupWeight(int u, int groupId) const
{
    return node_hyperedge_weight_group[u][groupId];
}

/*
* get the number of hyperedges
*/
int HyperGraph::getNumEdge() const
{
    return hyperedges.size();
}

/*
* remove all the hyperedges
*/
void HyperGraph::clearEdges()
{
    //hyperedges.clear();
    vvi().swap(hyperedges);
    node_hyperedges.clear();
    node_hyperedge_weight.clear();
    cout << "clear edges!" << endl;
}

// generating reachability set of fake campaign
bool HyperGraph::phaseOne(Graph &g, ii &root_data, int &traversal_data, priority_queue<ii, vii, greater<ii> > &pq, vb &visit, vi &visit_index, vi &dist, vi &aw_length)
{
    int cur, root_index;
    float flip;
    unsigned int i;
    int num_marked = 0;

    unsigned int num_fs = g.getNumFakeSeeds();
    const vi &fs = g.getFakeSeeds();
    for (i = 0; i < num_fs; i++) {
        dist[fs[i]] = 0;
        aw_length[fs[i]] = 0;
        pq.push(make_pair(0,fs[i]));
    }

    while ( !pq.empty() ) {
        cur = pq.top().second; pq.pop();
        if (visit[cur]) continue; // duplicate entry in pq
        visit[cur] = true;
        visit_index[num_marked] = cur;
        num_marked++;
        if (aw_length[cur] < 0) aw_length[cur] = g.getActivationWindow();
        const vf &w = g.getOutWeights(cur);
        const vi &neigh = g.getOutNeighbours(cur);
        for (i = 0; i < g.node_deg[cur]; i++) {
            if (!g.fake_seed[neigh[i]]) { // do not consider in-neighbours of nodes in S_F
                flip = sfmt_genrand_uint32(&sfmtSeed) / (float)(g.UI_MAX);
                if (flip < w[i]) {
                    if (dist[cur] + aw_length[cur] + 1 < dist[neigh[i]]) {
                        dist[neigh[i]] = dist[cur] + aw_length[cur] + 1;
                        pq.push(make_pair(dist[neigh[i]],neigh[i]));
                    }
                }
            }
        }
    }
    
    // if nodes reachable from F
    if (num_marked > num_fs) {
        // select root uniformly at random from nodes reached by F
        root_index = sfmt_genrand_uint32(&sfmtSeed)%(num_marked - num_fs);
        root_data.first = visit_index[root_index + num_fs];
        root_data.second = dist[root_data.first];
    }

    // reset local data structures
    for(i = 0; i < num_marked; i++) {
        dist[visit_index[i]] = INT_MAX;
    }

    // return num_marked for data reset
    traversal_data = num_marked;

    if (num_marked <= num_fs) return true; // F failed to activate any nodes


    return false;
}

// reverse Djikstra to identify nodes that can generate reward
void HyperGraph::phaseTwo(Graph &g, ii &root_data, ii &traversal_data, priority_queue<ii, vii, greater<ii> > &pq, vb &phase_one_visit, 
    vvi &parent_permutation, vb &visit, vi &visit_index, vi &delayed_dist, vi &aw_length, vb &overlap, vi &tb_nodes, vvi &visit_neighbours, 
    vvi &visit_neighbours_meet_len, vi &hyperedge, bool sa_upper)
{
    unsigned int  i, rand_pos;
    int new_dist, meet_len;
    float flip;
    bool crit_edge;
    /// root_data.first is id, root_data.second is dist
    int cur = root_data.first;
    
    int num_marked = 0;
    visit[cur] = true;

    const vf &w = g.getInWeights(cur);
    const vi &neigh = g.getInNeighbours(cur);
    if(w.size() == 0){
        hyperedge.push_back(root_data.first);
        traversal_data.first = 0;
        traversal_data.second = 0;
        return;
    }
    double p =  w[0];
    double log2Prob = Logarithm(1 - p);
    flip = sfmt_genrand_uint32(&sfmtSeed) / (float)(g.UI_MAX);
//    cout << "flip = " << flip << endl;
    int startPos = Logarithm(flip) / log2Prob;
//    cout << "startPos = " << startPos << endl;
    int endPos = g.rev_node_deg[cur];
//    cout << "endPos = " << endPos << endl;
    while (startPos < endPos)
    {
        crit_edge = false;
        if ( phase_one_visit[neigh[startPos]] && phase_one_visit[cur] ) {
            crit_edge = true;
        }
        rand_pos = sfmt_genrand_uint32(&sfmtSeed)%(parent_permutation[cur].size() + 1);
        if (rand_pos == parent_permutation[cur].size()) {
            parent_permutation[cur].push_back(neigh[startPos]);
        } else {
            parent_permutation[cur].push_back(parent_permutation[cur][rand_pos]);
            parent_permutation[cur][rand_pos] = neigh[startPos];
        }
        //meet_len = (crit_edge && sa_upper) ? 1.0 : g.generateMeetingLength(neigh[startPos]);
        meet_len = (sa_upper) ? 1.0 : g.generateMeetingLength(neigh[startPos]);
        visit_neighbours[neigh[startPos]].push_back(cur);
        visit_neighbours_meet_len[neigh[startPos]].push_back(meet_len);
        if ( !g.fake_seed[neigh[startPos]] ) {
            delayed_dist[neigh[startPos]] = meet_len;
            if ( phase_one_visit[neigh[startPos]] ) overlap[neigh[startPos]] = true;
            pq.push(make_pair(delayed_dist[neigh[startPos]],neigh[startPos]));
        }
        flip = sfmt_genrand_uint32(&sfmtSeed) / (float)(g.UI_MAX);
        int increment = Logarithm(flip) / log2Prob;
        startPos += (increment + 1);
    }

    while ( !pq.empty() ) {
//        cout << "pq.size = " << pq.size() << endl;
        /// need to think twice
        if (pq.top().first > root_data.second)
            break;
        /// need to think twice
        cur = pq.top().second; pq.pop();
//        cout << "cur = " << cur << endl;
        if (visit[cur]) continue; // duplicate entry in pq
        visit[cur] = true;
        visit_index[num_marked] = cur;
        num_marked++;
        // only genreate AW length if not previously computed in phase I
        if (aw_length[cur] < 0) aw_length[cur] = g.getActivationWindow();

        const vf &w = g.getInWeights(cur);
        const vi &neigh = g.getInNeighbours(cur);
        if(w.size() == 0)
            continue;
        p =  w[0];
        log2Prob = Logarithm(1 - p);
        flip = sfmt_genrand_uint32(&sfmtSeed) / (float)(g.UI_MAX);
        startPos = Logarithm(flip) / log2Prob;
        endPos = g.rev_node_deg[cur];
//        cout << "startPos = " << startPos << endl;
//        cout << "endPos = " << endPos << endl;
        while (startPos < endPos)
        {
//            cout << "startPos_while = " << startPos << endl;
            if (visit[neigh[startPos]]){
                flip = sfmt_genrand_uint32(&sfmtSeed) / (float)(g.UI_MAX);
                int increment = Logarithm(flip) / log2Prob;
                startPos += (increment + 1);
                continue;
            }
            crit_edge = false;
            if ( phase_one_visit[neigh[startPos]] && phase_one_visit[cur] ) {
                crit_edge = true; // otherwise live edge
            }
            rand_pos = sfmt_genrand_uint32(&sfmtSeed)%(parent_permutation[cur].size() + 1);
            if (rand_pos == parent_permutation[cur].size()) {
                parent_permutation[cur].push_back(neigh[startPos]);
            } else {
                parent_permutation[cur].push_back(parent_permutation[cur][rand_pos]);
                parent_permutation[cur][rand_pos] = neigh[startPos];
            }
            meet_len = (crit_edge && sa_upper) ? 1.0 : g.generateMeetingLength(neigh[startPos]);
            visit_neighbours[neigh[startPos]].push_back(cur);
            visit_neighbours_meet_len[neigh[startPos]].push_back(meet_len);
            if ( !g.fake_seed[neigh[startPos]] ) { // don't attempt to put fake seeds in pq
                if (phase_one_visit[cur] || overlap[cur]) overlap[neigh[startPos]] = true;
                new_dist = delayed_dist[cur] + aw_length[cur] + meet_len;
                if (new_dist < delayed_dist[neigh[startPos]]) { // new SP found --> add entry to PQ
                    delayed_dist[neigh[startPos]] = new_dist;
                    pq.push(make_pair(delayed_dist[neigh[startPos]],neigh[startPos]));
                }
            }
            flip = sfmt_genrand_uint32(&sfmtSeed) / (float)(g.UI_MAX);
            int increment = Logarithm(flip) / log2Prob;
            startPos += (increment + 1);
//            cout << "startPos_while = " << startPos << endl;
        }
    }
    // if not empty
    while ( !pq.empty() ){
//        cout << "Not empty()\n";
        pq.pop();
    }
    // put root in RR set
    hyperedge.push_back(root_data.first);

    // determine nodes that do not require TB'ing and put them in RR set
    int num_tb = 0;
    for(i = 0; i < num_marked; i++) {
        if ( !overlap[visit_index[i]] ) {
            if ( delayed_dist[visit_index[i]] < root_data.second ) {
                hyperedge.push_back(visit_index[i]);
            }
        } else {
            // add to set of tiebreak nodes
            tb_nodes[num_tb] = visit_index[i];
            num_tb++;
        }
    }

    // reset local data structures
    visit[root_data.first] = false;
    for(i = 0; i < num_marked; i++) {
        visit[visit_index[i]] = false;
        overlap[visit_index[i]] = false;
        delayed_dist[visit_index[i]] = INT_MAX;
    }

    traversal_data.first = num_tb;
    traversal_data.second = num_marked;
}

void HyperGraph::phaseTwoOrdinary(Graph &g, ii &root_data, ii &traversal_data, priority_queue<ii, vii, greater<ii> > &pq, vb &phase_one_visit,
                          vvi &parent_permutation, vb &visit, vi &visit_index, vi &delayed_dist, vi &aw_length, vb &overlap, vi &tb_nodes, vvi &visit_neighbours,
                          vvi &visit_neighbours_meet_len, vi &hyperedge, bool sa_upper)
{
    unsigned int  i, rand_pos;
    int new_dist, meet_len;
    float flip;
    bool crit_edge;
    /// root_data.first is id, root_data.second is dist
    int cur = root_data.first;

    int num_marked = 0;
    visit[cur] = true;

    const vf &w = g.getInWeights(cur);
    const vi &neigh = g.getInNeighbours(cur);
    if(w.size() == 0){
        hyperedge.push_back(root_data.first);
        traversal_data.first = 0;
        traversal_data.second = 0;
        return;
    }
    for (i = 0; i < g.rev_node_deg[cur]; i++)
    {
        crit_edge = false;
        if ( phase_one_visit[neigh[i]] && phase_one_visit[cur] ) {
            crit_edge = true;
        }
        rand_pos = sfmt_genrand_uint32(&sfmtSeed)%(parent_permutation[cur].size() + 1);
        if (rand_pos == parent_permutation[cur].size()) {
            parent_permutation[cur].push_back(neigh[i]);
        } else {
            parent_permutation[cur].push_back(parent_permutation[cur][rand_pos]);
            parent_permutation[cur][rand_pos] = neigh[i];
        }
        //meet_len = (crit_edge && sa_upper) ? 1.0 : g.generateMeetingLength(neigh[startPos]);
        meet_len = (sa_upper) ? 1.0 : g.generateMeetingLength(neigh[i]);
        visit_neighbours[neigh[i]].push_back(cur);
        visit_neighbours_meet_len[neigh[i]].push_back(meet_len);
        if ( !g.fake_seed[neigh[i]] ) {
            delayed_dist[neigh[i]] = meet_len;
            if ( phase_one_visit[neigh[i]] ) overlap[neigh[i]] = true;
            pq.push(make_pair(delayed_dist[neigh[i]],neigh[i]));
        }
    }

    while ( !pq.empty() ) {
//        cout << "pq.size = " << pq.size() << endl;
        /// need to think twice
        if (pq.top().first > root_data.second)
            break;
        /// need to think twice
        cur = pq.top().second; pq.pop();
//        cout << "cur = " << cur << endl;
        if (visit[cur]) continue; // duplicate entry in pq
        visit[cur] = true;
        visit_index[num_marked] = cur;
        num_marked++;
        // only genreate AW length if not previously computed in phase I
        if (aw_length[cur] < 0) aw_length[cur] = g.getActivationWindow();

        const vf &w = g.getInWeights(cur);
        const vi &neigh = g.getInNeighbours(cur);
        if(w.size() == 0)
            continue;
        for (i = 0; i < g.rev_node_deg[cur]; i++)
        {
//            cout << "startPos_while = " << startPos << endl;
            if (visit[neigh[i]]){
                continue;
            }
            crit_edge = false;
            if ( phase_one_visit[neigh[i]] && phase_one_visit[cur] ) {
                crit_edge = true; // otherwise live edge
            }
            rand_pos = sfmt_genrand_uint32(&sfmtSeed)%(parent_permutation[cur].size() + 1);
            if (rand_pos == parent_permutation[cur].size()) {
                parent_permutation[cur].push_back(neigh[i]);
            } else {
                parent_permutation[cur].push_back(parent_permutation[cur][rand_pos]);
                parent_permutation[cur][rand_pos] = neigh[i];
            }
            meet_len = (crit_edge && sa_upper) ? 1.0 : g.generateMeetingLength(neigh[i]);
            visit_neighbours[neigh[i]].push_back(cur);
            visit_neighbours_meet_len[neigh[i]].push_back(meet_len);
            if ( !g.fake_seed[neigh[i]] ) { // don't attempt to put fake seeds in pq
                if (phase_one_visit[cur] || overlap[cur]) overlap[neigh[i]] = true;
                new_dist = delayed_dist[cur] + aw_length[cur] + meet_len;
                if (new_dist < delayed_dist[neigh[i]]) { // new SP found --> add entry to PQ
                    delayed_dist[neigh[i]] = new_dist;
                    pq.push(make_pair(delayed_dist[neigh[i]],neigh[i]));
                }
            }
        }
    }
    // if not empty
    while ( !pq.empty() ){
//        cout << "Not empty()\n";
        pq.pop();
    }
    // put root in RR set
    hyperedge.push_back(root_data.first);

    // determine nodes that do not require TB'ing and put them in RR set
    int num_tb = 0;
    for(i = 0; i < num_marked; i++) {
        if ( !overlap[visit_index[i]] ) {
            if ( delayed_dist[visit_index[i]] < root_data.second ) {
                hyperedge.push_back(visit_index[i]);
            }
        } else {
            // add to set of tiebreak nodes
            tb_nodes[num_tb] = visit_index[i];
            num_tb++;
        }
    }

    // reset local data structures
    visit[root_data.first] = false;
    for(i = 0; i < num_marked; i++) {
        visit[visit_index[i]] = false;
        overlap[visit_index[i]] = false;
        delayed_dist[visit_index[i]] = INT_MAX;
    }

    traversal_data.first = num_tb;
    traversal_data.second = num_marked;
}

// dynamic programming routine for proportional tie-breaking
void HyperGraph::phaseThree(Graph &g, int root, int num_tb, priority_queue<ii, vii, greater<ii> > &pq, vi &tb_nodes, vi &aw_length, 
    vvi &visit_neighbours, vvi &visit_neighbours_meet_len, vvi &parent_permutation, vvi &parents, vvi &parent_arrivals, 
    vb &adopt_fake, vb &adopt_true, vb &visit, vi &tb_index, vi &hyperedge, bool sa_upper)
{
    unsigned int i, j, k;
    int candidate;
    int cur;
    int edge_len;
    int count;
    int permutation_node;
    int aw_close;
    int new_close;
    bool found;

    int num_marked_tb = 0;
    bool true_neighbour = false;

    unsigned int num_fs = g.getNumFakeSeeds();
    const vi &fs = g.getFakeSeeds();
    for (i = 0; i < num_fs; i++) {
        adopt_fake[fs[i]] = true;
    }

    for(i = 0; i < num_tb; i++) {
        candidate = tb_nodes[i];
        adopt_true[candidate] = true;

        visit[candidate] = true;

        // add fake neighbours into pq
        for (j = 0; j < num_fs; j++) {
            const vi &neigh = visit_neighbours[fs[j]];
            for (k = 0; k < neigh.size(); k++) {
                if (neigh[k] == candidate) continue;
                pq.push(make_pair(1+aw_length[neigh[k]],neigh[k]));
                parents[neigh[k]].push_back(fs[j]);
                parent_arrivals[neigh[k]].push_back(1);
            }
        }

        // add current candidate's neighbours into pq
        const vi &neigh = visit_neighbours[candidate];
        const vi &neigh_meet_len = visit_neighbours_meet_len[candidate];
        for (j = 0; j < neigh.size(); j++) {
            pq.push(make_pair(neigh_meet_len[j]+aw_length[neigh[j]],neigh[j]));
            parents[neigh[j]].push_back(candidate);
            parent_arrivals[neigh[j]].push_back(neigh_meet_len[j]);
        }

        // apply dynamic programming routine
        while (!pq.empty()) {
            cur = pq.top().second; 
            aw_close = pq.top().first; pq.pop();
            if (visit[cur]) continue; // duplicate entry in pq
            visit[cur] = true;
            tb_index[num_marked_tb] = cur;
            num_marked_tb++;

            // resolve adoption of cur via parent permutation
            count = 0;
            found = false;
            while(!found && count < parent_permutation[cur].size()) {
                permutation_node = parent_permutation[cur][count];
                for (j = 0; j < parents[cur].size(); j++) {
                    if (parents[cur][j] == permutation_node && parent_arrivals[cur][j] <= aw_close) {
                        if (adopt_fake[parents[cur][j]]) {
                            adopt_fake[cur] = true;
                        } else {
                            adopt_true[cur] = true;
                        }
                        found = true;
                        break;
                    }
                }
                count++;
            }

            // if root reached and adoption determined we can terminate early
            // NOTE: pq might still have elements in it
            if (cur == root) break;

            const vi &neigh = visit_neighbours[cur];
            const vi &neigh_meet_len = visit_neighbours_meet_len[cur];
            for (j = 0; j < neigh.size(); j++) {
                if (!visit[neigh[j]]) {
                    edge_len = adopt_fake[cur] ? 1 : neigh_meet_len[j];
                    new_close = aw_close + edge_len + aw_length[neigh[j]];
                    pq.push(make_pair(new_close,neigh[j]));
                    parents[neigh[j]].push_back(cur);
                    parent_arrivals[neigh[j]].push_back(aw_close + edge_len);
                }
            }
        }

        // add candidate to RDR set based on root and root neighbour adoptions
        const vi &par = parents[root];
        const vi &par_arrival = parent_arrivals[root];
        for (j = 0; j < par.size(); j++) {
            if (par_arrival[j] <= aw_close) {
                true_neighbour = true;
            }
        }
        if(sa_upper && true_neighbour)
            hyperedge.push_back(candidate);
        else if(adopt_true[root])
            hyperedge.push_back(candidate);

        // flush out pq
        while (!pq.empty()) {
            cur = pq.top().second; pq.pop();
            if (visit[cur]) continue; // duplicate entry in pq
            visit[cur] = true;
            tb_index[num_marked_tb] = cur;
            num_marked_tb++;
        }
        // reset data structures
        adopt_true[candidate] = false;
        visit[candidate] = false;
        for (j = 0; j < num_marked_tb; j++) {
            cur = tb_index[j];
            visit[cur] = false;
            adopt_fake[cur] = false;
            adopt_true[cur] = false;
            vi().swap(parents[cur]);
            vi().swap(parent_arrivals[cur]);
        }
        num_marked_tb = 0;
    }
}

// reset data structures
void HyperGraph::reset(Graph &g, int &phase_one_traversal_data, ii &phase_two_traversal_data, vi &phase_one_visit_index, vb &phase_one_visit,
     vi &phase_two_visit_index, vvi &visit_neighbours, vvi &visit_neighbours_meet_len, vvi &parent_permutation,
    vi &aw_length, vi &hyperedge)
{
    int i;

    vi().swap(hyperedge);

    // phase 1:
    for(i = 0; i < phase_one_traversal_data; i++) {
        phase_one_visit[phase_one_visit_index[i]] = false;
        aw_length[phase_one_visit_index[i]] = -1;
    }


    // phase 2:
    for(i = 0; i < phase_two_traversal_data.second; i++) {
        vi().swap(visit_neighbours[phase_two_visit_index[i]]);
        vi().swap(visit_neighbours_meet_len[phase_two_visit_index[i]]);
        vi().swap(parent_permutation[phase_two_visit_index[i]]);
        aw_length[phase_two_visit_index[i]] = -1;
    }

    int num_fs = g.getNumFakeSeeds();
    const vi &fs = g.getFakeSeeds();
    for (i = 0; i < num_fs; i++) {
        vi().swap(visit_neighbours[fs[i]]);
        vi().swap(visit_neighbours_meet_len[fs[i]]);
    }
}

/*
* calculate log(x)
*/
inline double HyperGraph:: Logarithm(const double x)
{
    return log2f(x);
}
/*
* convert from an integer to a string
*/
string intToStr(int i) {
    stringstream ss;
    ss << i;
    return ss.str();
}

/*
* convert from a strong to an integer
*/
unsigned int strToInt(string s) {
    unsigned int i;
    istringstream myStream(s);

    if (myStream>>i) {
        return i;
    } else {
        cout << "String " << s << " is not a number." << endl;
        return atoi(s.c_str());
    }
    return i;
}

/*
* measure the consumed memory
*/
float getCurrentMemoryUsage() {
    string pid = intToStr(unsigned(getpid()));
    string outfile = "tmp_" + pid + ".txt";
    string command = "pmap " + pid + " | grep -i Total | awk '{print $2}' > " + outfile;
    system(command.c_str());

    string mem_str;
    ifstream ifs(outfile.c_str());
    std::getline(ifs, mem_str);
    ifs.close();

    mem_str = mem_str.substr(0, mem_str.size()-1);
    float mem = (float)strToInt(mem_str);

    command = "rm " + outfile;
    system(command.c_str());

    return mem/1024;
}
