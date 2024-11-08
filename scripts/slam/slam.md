# Graph-Based SLAM and Graph Optimization

This document provides an overview and mathematical example of the graph optimization process in graph-based SLAM (Simultaneous Localization and Mapping).

## Overview

In graph-based SLAM, we aim to build a map of an environment while simultaneously determining the position of a robot within it. The "graph" represents both the robot's estimated trajectory and the environment map:
- **Nodes** represent the robot's poses (position and orientation) at different time steps, as well as the positions of observed landmarks.
- **Edges** represent constraints or measurements between these poses, derived from sensor data (e.g., odometry, lidar, or visual data).

### Goal of Optimization

The purpose of optimizing the graph is to refine the estimated trajectory and map. Optimization minimizes the errors in the estimated positions of the robot and landmarks by adjusting:
- **Nodes** (poses of the robot and landmarks)
- **Edges** (constraints or measurements between nodes)

## Mathematical Explanation

### Problem Setup

1. **Nodes**: We have a series of nodes x1,x2,…,xN where each node represents the pose of the robot at different time steps.
    * Each pose xixi​ can be represented by its position and orientation, e.g. 

    $$
    xi=(xi,yi,θi)
    $$

    where x<sub>i</sub> and y<sub>i</sub>​ are position coordinates, and θ<sub>i</sub>​ is the orientation.

2. **Edges (Constraints)**: Each edge represents a measurement z<sub>ij</sub>​ that constrains the relative position between nodes x<sub>i</sub>​ and x<sub>​j</sub>​.
    * For example, if the robot has an odometry measurement between poses x<sub>i</sub>​​ and x<sub>i+1</sub>​, this gives a relative position constraint between these nodes.

### Error Formulation

The goal of graph optimization is to adjust the nodes x<sub>i</sub>​ such that the constraints (edges) are satisfied as closely as possible. Mathematically, we aim to minimize the total error over all edges.

### Cost Function

For each edge z<sub>ij</sub> (measurement between nodes x<sub>i</sub>​  and xj<sub>j</sub>​ ), we define an error function e<sub>ij</sub>​:

$$
e​=zij​−h(xi​,xj​)
$$

where:

* z<sub>ij</sub>​ is the measured relative position between nodes <sub>i</sub> and <sub>j</sub>,
* h(x<sub>i</sub>,x<sub>j</sub>) is a function that computes the expected relative position between x<sub>i</sub> and x<sub>j</sub> based on their estimated positions.

The error e<sub>ij</sub>​ represents the discrepancy between the actual measurement z<sub>ij</sub>​ and the estimated position given by h(x<sub>i</sub>,x<sub>j</sub>).

## Example Calculation

Suppose we have three nodes x<sub>1</sub>​, x<sub>2</sub>, and x<sub>3</sub> with the following measurements (constraints):

Edge (1, 2): z<sub>12</sub>=(1,0,0)             z<sub>12</sub>=(1,0,0), meaning node x<sub>2</sub> is expected to be 1 unit to the right of x<sub>1</sub>​.

Edge (2, 3): z<sub>23</sub>=(1,0,0)             z<sub>23</sub>=(1,0,0), meaning node x<sub>3</sub> is expected to be 1 unit to the right of x<sub>2</sub>.

Loop Closure (1, 3): z<sub>13</sub>=(2,0,0)     z<sub>13</sub>=(2,0,0), indicating that node x<sub>3</sub> should be 2 units to the right of x<sub>1</sub>.

Now, let’s say the initial estimates for the nodes’ positions are:

$$
x1​=(0,0,0),x2​=(0.9,0,0),x3​=(1.8,0,0)
$$

1) Edge (1, 2):
$$
e12​=z12​−h(x1​,x2​)=(1,0,0)−(0.9,0,0)=(0.1,0,0)
$$

2) Edge (2, 3):
$$
e23​=z23​−h(x2​,x3​)=(1,0,0)−(0.9,0,0)=(0.1,0,0)
$$

3) Loop Closure (1, 3):
$$
e13​=z13​−h(x1​,x3​)=(2,0,0)−(1.8,0,0)=(0.2,0,0)
$$

### Error Calculation

Using the error function, we calculate the discrepancy between expected and actual measurements:

Assuming identity matrices for Ω<sub>ij</sub>​, the cost function E<sub>e</sub> becomes:

$$
E=e12Te12+e23Te23+e13Te13=(0.1)2+(0.1)2+(0.2)2=0.01+0.01+0.04=0.06
$$

### Optimization

To minimize this error E<sub>3</sub>​, we would adjust x<sub>1</sub>​, x<sub>2</sub>​, and x<sub>3</sub>​ to reduce the discrepancies in each edge’s error term. Through iterative optimization techniques like Gauss-Newton or Levenberg-Marquardt, the algorithm updates the node positions until E<sub>e</sub>​ is minimized, resulting in a more consistent and accurate map and trajectory.

## Tools for Optimization

Popular libraries for implementing graph-based SLAM include:
- **g2o (General Graph Optimization)**
- **Ceres Solver**
- **GTSAM (Georgia Tech Smoothing and Mapping)**

These libraries offer efficient nonlinear optimization methods suited for SLAM applications. In this project GTSAM library was selected.

## Summary

In graph-based SLAM, graph optimization is used to align the robot's trajectory and map by minimizing errors in the graph structure. This process refines both the estimated path and the environment map, yielding a more accurate and consistent model of the robot's surroundings.