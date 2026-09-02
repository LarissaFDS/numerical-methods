# Numerical Methods - Exercise List 1 

**Institution:** Universidade Federal de Alagoas (UFAL) - Instituto de Computação
**Course:** Computer Engineering - Numerical Methods 
**Professor:** Thales Vieira 
**Date:** August 19, 2026 

## Description
This repository contains the Python implementations and computational resolutions for the first list of exercises in the Numerical Methods course. The exercises focus on floating-point representations, polynomial approximations, and root-finding algorithms.

*Note: All algorithms are implemented from scratch without the use of external numerical libraries, as strictly required by the assignment instructions.*

## Topics Covered
1. **IEEE-754 Floating-Point Representation:** Analysis of 64-bit double-precision numbers, extracting the sign, exponent, and mantissa, and converting them to decimal[cite: 1].
2. **Maclaurin Series & Decimal Arithmetic:** Computing approximations of $\sin(0.5)$ and $\cos(x)$ using Maclaurin polynomials of degree 8, nested forms, and Python's `decimal` library with rounding.
3. **Bisection Method & Newton's Law of Cooling:** Formulating and solving $f(t) = 0$ to find the time a cooling device reaches $35^{\circ}C$, including an algorithm to automatically find valid intervals of length $C$.
4. **Function Plotting & Convergence Visualization:** Plotting functions (e.g., $\ln(x) - 2^x + x^2 - 1 = 0$) and visually tracking the sequence of root approximations.
5. **Fixed-Point Iteration:** Analyzing different iteration functions ($g_1(x)$ and $g_2(x)$) for $x^2 + x - 1 = 0$ to verify convergence based on the Fixed-Point Theorem.
6. **Newton's Method (Newton-Raphson):** Implementing Newton's method for $x^3 - 2x - 5 = 0$ and comparing its convergence rate (number of iterations) and accuracy with the Bisection method.

## Requirements
- Python 3.x
- Standard libraries only.
- External numerical libraries (e.g., NumPy/SciPy) are strictly prohibited for calculations.
- Python's `decimal` module is used for controlled arithmetic operations.

## Repository Structure
- `q1_ieee754.py`: Floating-point conversion and binary sequence analysis.
- `q2_maclaurin.py`: Maclaurin polynomial evaluation and nested forms.
- `q3_bisection.py`: Bisection method implementation and interval search algorithm.
- `q4_plotting.py`: Visualization of functions and convergence sequences.
- `q5_fixed_point.py`: Fixed-point iteration method with convergence testing.
- `q6_newton.py`: Newton's method vs Bisection comparison.

## Deadline
September 18, 2026.
