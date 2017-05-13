# Function Approximation

An example of how to use orthogonal projection to speed up some mathematical CUDA kernels. Checkout [this article](http://shuyangsun.com/portfolio/orthogonal-projection-and-parallel-computing.htm) I wrote for a detailed explanation of the math and implementation behind it.

## Introduction

Some CUDA kernels perform heavy scientific computing tasks, this sample project shows how to find alternative functions and speed up the kernel.

## Functionalities

### Orthogonal Projection Calculator

To approximate functions with polynomial, use [opc.py](https://github.com/shuyangsun/Function-Approximation-on-CUDA/blob/master/opc.py) (Orthogonal Projection Calculator). It's able to calculate the derivative, integral of a polynomial; get an orthonormal basis of an inner product space of polynomials; or find polynomial approximation of any continuous  real-valued function.

To get a detailed list of its commands, use `python opc.py --help`.

### CUDA Program

The CUDA program in this project is an implementation of a specific example I mentioned in [the article](http://shuyangsun.com/portfolio/orthogonal-projection-and-parallel-computing.htm). It contains many testing cases to prove the claims in the article.

### Supplementary Resources

The [supplementary_resources](https://github.com/shuyangsun/Function-Approximation-on-CUDA/tree/master/supplementary_resources) folder contains some additional materials to help you understand orthogonal projection visually (a macOS Grapher file), and a text file with a sample output of Orthogonal Projection Calculator when approximating a function.

