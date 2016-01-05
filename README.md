# NNclustering
The demo code for "Neural network-based clustering using pairwise constraints" (http://arxiv.org/abs/1511.06321)

## Prerequest
Install the [hungarian algorithm](https://github.com/yenchanghsu/hungarian.torch) for optimal cluster assignment.

## Demo
The demo code is modified from [szagoruyko's work](https://github.com/szagoruyko/cifar.torch).

Do classification on MNIST:
```
th demo.lua
```
Do clustering on MNIST:
```
th demo.lua --clustering 1 -b 32 -r 0.01
```
To lookup the available arguments:
```
th demo.lua -h
```

## Usage of BatchKLDivCriterion layer
It can handle 3 types of 'target'. The notation of n is the size of mini-batch.
 1. nx1 class label: It uses the same label as classification task. The pairwise relationship will be enumerated in BatchKLDivCriterion automatically.
 2. nxn relationship matrix: target\[i]\[j]= 1:similar pair; -1:dissimilar pair; 0:no relationship
 3. nx3 tuple: (i, j, relationship) where i and j are the index of sample inside the mini-batch and relationship indicates similar/dissimilar pair. relationship= 1:similar pair; 0: dissimilar pair

## Acknowledgments
This work was supported by the National Science Foundation and National Robotics Initiative (grant # IIS-1426998).