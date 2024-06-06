# Clustering Algorithms Implementation

This repository contains Python code for implementing two clustering algorithms: DBSCAN (Density-Based Spatial Clustering of Applications with Noise) and Agglomerative Clustering.

## Description

- **DBSCAN**: Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is a density-based clustering algorithm. It groups together points that are closely packed together (points with many nearby neighbors), marking as outliers those that lie alone in low-density regions (whose nearest neighbors are too far away). The algorithm requires two parameters: `epsilon` (a threshold distance) and `minpts` (the minimum number of points required to form a dense region).
  
- **Agglomerative Clustering**: Agglomerative clustering is a hierarchical clustering method that builds clusters by merging individual data points or small clusters successively. At each step, the two nearest clusters are merged into a larger cluster. The process continues until only a single cluster remains or until a stopping criterion is met. The `linkage` parameter determines the metric used for computing the distance between clusters.

## Features

- **DBSCAN Implementation**: Implementing the DBSCAN algorithm from scratch using NumPy.
- **Agglomerative Clustering**: Using scikit-learn to perform Agglomerative Clustering.
- **Evaluation**: Computing the Adjusted Rand Index (ARI) to evaluate the clustering performance.

## Directory Structure

- `dbscan.py`: Contains the implementation of the DBSCAN algorithm.
- `agglomerative_clustering.py`: Implements Agglomerative Clustering using scikit-learn.
- `data_points.py`: Contains example data points for testing the algorithms.
- `evaluation.py`: Evaluates the clustering results using the Adjusted Rand Index.
- `README.md`: Provides an overview of the repository and instructions for running the code.

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/your-username/clustering-algorithms.git
cd clustering-algorithms
