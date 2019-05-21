Code for the Interpolated Discretized (ID) Embedding
----------------------------------------------------
Ofir Pele, Alexey Kurbatsky
Contact: ofir.pele@g.ariel.ac.il
Version: 1, August 2016

This directory contains the source code for computing the Interpolated Discretized (ID) embedding.

Please cite this paper if you use this code:
 Interpolated Discretized Embedding of Single Vectors and Vector Pairs for Classification, Metric Learning and Distance Approximation
 Ofir Pele, Yakir Ben-Aliz
 arXiv 2016
bibTex:
@article{Pele-arXiv2016d,
  title={Interpolated Discretized Embedding of Single Vectors and Vector Pairs for Classification, Metric Learning and Distance Approximation},
  author = {Ofir Pele and Yakir Ben-Aliz},
  journal={arXiv},
  year={2016}
}

I plan to publish a new distribution that will include SVM learning with a stochastic subgradient decent 
algorithm with embedding on the fly in the future. I also hope to publish a python wrapper and a version that uses the GPU.

Easy startup
------------
See IDdemo.cpp

Compiling (the folder contains compiled binaries, thus you might not have to compile)
-------------------------------------------------------------------------------------
In a linux shell:
>> make

Licensing conditions
--------------------
See the file LICENSE.txt in this directory for conditions of use.
