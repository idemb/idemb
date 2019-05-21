Code for the Interpolated Discretized (ID) Embedding



Original code was taken from: Ofir Pele, Alexey Kurbatsky

Based on work by: Ofir Pele, Yakir Ben Aliz

contains the source code for computing the Interpolated Discretized (ID) embedding.

Please cite this paper if you use this code:
 Interpolated Discretized Embedding of Single Vectors and Vector Pairs for Classification, Metric Learning and Distance Approximation
 Ofir Pele, Yakir Ben-Aliz
 arXiv 2016
bibTex:
```
@article{Pele-arXiv2016d,

  title={Interpolated Discretized Embedding of Single Vectors and Vector Pairs for Classification, Metric Learning and Distance Approximation},

  author = {Ofir Pele and Yakir Ben-Aliz},

  journal={arXiv},

  year={2016}

}
```

Hopefully we'll get to the the stuff in TODO.txt in the near future.

Startup
------------
Run LearningMain as main. It runs a same/not-same (with various parameters) learning on colour pairs. The pairs were tagged by humans and appear in the files AnswersA*.csv. 

Prerequisites
--------------
(tested on ubuntu 16.04.3) 

cmake

armadillo-7.400.1



Compiling 
-----------------
In a linux shell:

```
>> make

>> ./LearningMain
```



Licensing conditions
--------------------
See the file LICENSE.txt for conditions of use.
