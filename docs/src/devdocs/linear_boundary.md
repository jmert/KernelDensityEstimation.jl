# Linear Boundary Correction

When the domain for the density estimation contains a closed edge, the boundary creates distortions in the density as
the kernel "falls off" the edge.
It is simple to correct for the normalization error, but that only improves the lowest-order bias.
[Lewis2019](@citet) Sec. III.A provides the so-called _linear boundary kernel_ correction — given a basic density
estimate $f(𝒙)$ (the Gaussian-convolved histogram), the corrected density $\hat f(𝒙)$ is defined to be
```math
    \hat f(𝒙) = A_0 f(𝒙) + \sum_{i} A_i f_i^{(1)}(𝒙)
```
where $i$ ranges over each dimension in $𝒙 ∈ ℝ^n$ and $f_i^{(1)} = \frac{∂}{∂x_i} f(𝒙)$ is the first derivative with
respect to each coordinate direction.

In the following sections, the $W$ variables are various convolutions of the domain indicator function —
see [Lewis2019](@citet) for more details.

## Univariate

[Lewis2019](@citet) derives the coefficients $A_0$ and $A_1$ in general, with Eqn. 13 providing the solutions for
the univariate case when $i = j = 1$:
```math
\begin{align*}
    A_0 &= \frac{1}{W_0 - \left(W_1\right)^2 \left(W_2\right)^{-1}}
    &
    A_1 &= -\left(W_2\right)^{-1} W_1 A_0
\end{align*}
```
or combining with the corrected density above and rearranging terms:
```math
\begin{align*}
    \hat f(x) &= \frac{W_2 f(x) - W_1 f^{(1)}(x)}
        {W_0 W_2 - \left(W_1\right)^2}
\end{align*}
```

## Bivariate

For higher dimensions, it was easier to step back to Eqn. 12 which provides the constraints written in an implicit summation style:
```math
    \langle \hat f(x) \rangle = \left[ A_0 W_0 + A_1^i W_1^i \right] f(x)
        - \left[ A_0 W_1^i + A_1^j W_2^{ij} \right] f_i^{(1)}(x)
```
For 2D densities where $(i,j) ∈ \{1,2\} × \{1,2\}$ over two dimensions, there are three terms and therefore
three conditions to be satisfied.
The first term on the right is a single expression with 3 inner terms, while the second expands to two equations with
three inner terms each.
The condition is that the first condition evaluates to $1$, whereas the latter two should evaluate to $0$.
```math
\begin{eqnarray*}
    A_0 W_0   &+& A_1^1 W_1^1    &+& A_1^2 W_1^2    &=& 1 \\
    A_0 W_1^1 &+& A_1^1 W_2^{11} &+& A_1^2 W_2^{12} &=& 0 \\
    A_0 W_1^2 &+& A_1^1 W_2^{12} &+& A_1^2 W_2^{22} &=& 0 \\
\end{eqnarray*}
```
This is maybe more easily recognizable as a linear system of equations by rewriting it in matrix form.
```math
    \begin{bmatrix}
    W_0   & W_1^1    & W_1^2    \\
    W_1^1 & W_2^{11} & W_2^{12} \\
    W_1^2 & W_2^{12} & W_2^{22} \\
    \end{bmatrix}
    \begin{bmatrix} A_0 \\ A_1^1 \\ A_1^2 \end{bmatrix}
    =
    \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
```
Solving this by hand is tedious, but we can turn to any computer algebra system to solve it symbolically.
In the following example, we use Julia's [Symbolics.jl](https://docs.sciml.ai/Symbolics/stable/) package to derive
the set of expressions.

```julia
using Latexify
using LinearAlgebra
using Symbolics

A, Wi = @variables A[0:2] W[0:2]
Wij, = @variables W[1:2, 1:2]
Wij = Symmetric(Wij)

# construct the system of equations
W = [transpose(Wi);
     Wi[1:2]  Wij ]
eqns = Symbolics.scalarize(W * A ~ [1, 0, 0])

# solve the system of equations symbolically
soln = symbolic_linear_solve(eqns, A, simplify = true)

# and for better printing, extract the common denominator and rewrite
# the expressions a bit
D, = @variables D
denom = denominator(soln[1])
soln = simplify.(substitute(soln, Dict(denom => D, -denom => -D)))
exprs = [[D ~ denom]; [A...] .~ soln]
latexify(exprs)
```
```math
\begin{align*}
D &= W_{0} ~ W_{1,1} ~ W_{2,2} - \left( W_{1,2} \right)^{2} ~ W_{0} - \left( W_{1} \right)^{2} ~ W_{2,2} + 2 ~ W_{1} ~ W_{1,2} ~ W_{2} - \left( W_{2} \right)^{2} ~ W_{1,1} \\
A_{0} &= \frac{W_{1,1} ~ W_{2,2} - \left( W_{1,2} \right)^{2}}{D} \\
A_{1} &= \frac{W_{1} ~ W_{2,2} - W_{1,2} ~ W_{2}}{ - D} \\
A_{2} &= \frac{W_{1} ~ W_{1,2} - W_{1,1} ~ W_{2}}{D}
\end{align*}
```
