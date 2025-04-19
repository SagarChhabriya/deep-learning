Performance of NN:
Previously we learned: How can we improve the Performance and Speed up the training of NN

Speed Up Techniques that we have learned so far:
Weigh initialization
Batch Normalization
Activation Function

and No we are going to learn optimizers

## 8.1. Intro to Optimizers

- **Role of Optimizer**:
The goal is to minimize the loss


## 8.2 Types of Optimizers
- Batch GD
- Stochastic  GD
- Mini Batch GD


## 8.3 Challenges
1. Learning Rate
2. Learning Rate Scheduling
3. If the Loss function is n-dimensional then we have n directions of slope movement and all the directions has same learning rate. That is also a challenge that we don't need the same learning rate in all dimesions
4. Local Minima: In Complex Scenarios your algo get stuck on local minima some how the Stochastic gradient solves this problem somehow but there is a huge probability that your algorithm get stuck. You might have to work with sub-optimim solution
5. Saddle Point: Where the slope of one point goes upper and another's to downward.

These are the challenges in the Conventional gradient descent although we can have a solution but it may have caused with slow training or its a sub optimal solution. That is why need to study some other optimizers.

1. Momentum
2. Adagrade
3. NAG
4. RMSprop
5. Adam

-----

## Exponentially Weighted Moving Average or Weighted Average
A technique to find the hidden trend in time series data
Formula:

Day1 --> Avg
Day2 --> Avg
Day3 --> Avg
Day4 --> Avg
Day5 --> Avg

The weight of avg 3 will greater than of 1,2, and weight of avg 5 will be greater than its previous. This is a point should be noted before studying EWMA

V_t = betaV_t-1 + (1-beta) theta_t

V_t weighted average at a instance like day3
beta is constant varying between 0 and 1



## 1. SGD with Momentum (Momentum Optimization)

Non-Convex Optimization | Why Momentum
- high curvature
- consitent gradient 
- Noisy gradient 


## What?
Common example:
If you have to go to point B from A. And on the way you ask from 4 person that what is the route of B and all those pointed to the same direction →. So you will move fast in the very direction →. But what if two of them said → and two said ← and you are moving in → direction. Although 2 vs 2 and you are still moving in → direction but not that fast because of doubt. 

From physics perspective: Consider a ball coming dowm from a parabola like shape as it moves toward down it speeds up. 

![](../assets/24-sgd-momentum.png)


## Nesterov Accelerated Gradient (NAG)

Solve the issue of 