<h3>
	<center>Fairness-Aware Engagement Loss Reduction in Rumor Mitigation</center>
</h3>


### Information

Version 1.0: Implementation of Algorithm for  Fairness-Aware Engagement Loss Reduction in Rumor Mitigation. For more details about our code, please read our paper: "Jiajie F., Xueqin C., Xiangyu K., Lu C., Yunjun G., Fairness-Aware Engagement Loss Reduction in Rumor Mitigation"

### Introduction

1. This repository contains the full version of our paper.
2. This repository contains the codes and datasets used in our paper.
3. **Fairness-Aware Engagement Loss Reduction in Rumor Mitigation**.

Abstract: The rapid spread of negative information, such as rumors, poses significant challenges for social network operations and has considerable economic consequences.  Traditional rumor mitigation approaches focus on minimizing the number of rumor-affected users, often overlooking the varying social engagement levels and fairness concerns among user groups.
To address these gaps, we introduce the Fairness-aware Temporal Loss-based Rumor Mitigation (FAIR-TLRM) problem, which aims to identify a set of truth spreaders that maximize the expected reduction in economic loss while satisfying fairness constraints. FAIR-TLRM problem is **NP**-hard, monotone, and non-submodular. 
To solve this, we propose a dual-objective optimization algorithm, Fair-Greedy, for selecting truth spreaders by balancing loss reduction and group fairness. Given the $\#$P-hardness of computing mitigation rewards, we further devise a scalable algorithm, **FWS-RM**, that leverages group-aware weighted reverse influence sampling and a sandwich approximation technique. We empirically explore the trade-offs between rumor-induced losses and fairness concerning the network characteristics and introduce an effective **Joint-Greedy** framework. Compared with existing RM solutions, FWS-RM reduces economic losses by an average of 2$\times$, improves fairness by up to 8$\times$, and runs 5$\times$ faster. 

### Datasets

We use five publicly available real-world road networks, including EmailCore, Twitch, weibo, FaceBook and Digg datasets. 

EmailCore，FaceBook ,Twitch and be obtained from [1]. Weibo be obtained from [2]; Digg can be found in [3].

[1] Jure Leskovec and Andrej Krevl. 2014. SNAP Datasets: Stanford large network dataset collection. http://snap.stanford.edu/data.

[2] J. Zhang, B. Liu, J. Tang, T. Chen, and J. Li. “Social influence locality for modeling retweeting behaviors".

[3] K. Lerman, R. Ghosh, and T. Surachawala, “Social contagion: An empirical study of information spread on digg and twitter follower graphs”.

### Algorithms

The following files are the codes for our proposed algorithms. We implemented all the codes using C++ with CLion 2022.3.2.

1. First we use el2bin.cpp$^{[2]}$ (can be found in genSeed folder) to convert from a graph file in weighted edge list to two binary files that encode the graph and its transpose;

```shell
 ./el2bin <input file> <output file> <transpose output file>
```

2. Then we use fake_seeds.cpp to generate fake_seeds (random or influential). Specifically, when generating influential seeds, we set -m top, and the -f parameter means that the fake seeds will be randomly drawn from the top f-th fraction for generating influential fake seeds, the orders of the nodes are determined by the  singleton influence file. When choosing random seeds, just use four parameters (-n -o -k and -m), while setting -m random. The usages are listed as follows (the first line is for influential seeds and the second is for random seeds):

```shell
./fake_seeds -n <number of nodes> -o <seed output file>  -k <number of seeds> -m <top> -f <fraction> -s <singleton influence file>
```

```shell
./fake_seeds -n <number of nodes> -o <seed output file>  -k <number of seeds> -m <random> 
```

3. Then use **algorithm** Fair-TLSM:  A scalable algorithm, FWS-RM, that leverages group-aware weighted reverse influence sampling and a sandwich approximation technique to solve Fair-TLRM  to tackle our problem.

```shell
 ./Fair-TLSM -i <input networkFile> -o <result output file> -fakeseeds <fakeSeed file> -k <budget of blockSet> -epsilon <sample error parameter> -fakeinf <file contains estimate influence of rumor> -group <group affiliations> -nw <individual losses> -delta <failure probability> -alpha <fairness threshold>
```

[2] Michael Simpson, Farnoosh Hashemi, and Laks VS Lakshmanan. 2022. Misinformation mitigation under differential propagation rates and temporal penalties. Proceedings of the VLDB Endowment 15, 10 (2022), 2216–2229.

### Running Environment

A 64-bit Linux-based OS. 

GCC 4.7.2 and later.
