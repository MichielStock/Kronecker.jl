# Kronecker.jl Documentation

```@contents
Pages = [
    "man/basic.md",
    "man/types.md",
    "man/linalgebra.md",
    "man/multiplication.md"
    "man/factorization.md",
    "man/kroneckersums.md",
    "man/kroneckerpowers.md"
]
```

*A general-purpose toolbox for efficient Kronecker-based algebra.*

`Kronecker.jl` is a Julia package for working with large-scale Kronecker systems. The main feature of `Kronecker.jl` is providing a function `kronecker(A, B)` used to obtain an instance of the lazy `GeneralizedKroneckerProduct` type. In contrast to the native Julia function `kron(A, B)`, this does not compute the Kronecker product but instead stores the matrices in a specialized structure. Commonly-used mathematical functions are overloaded to provide the most efficient methods to work with Kronecker products. We also provide an equivalent binary operator `⊗` which can be used directly as a Kronecker product in statements, i.e., A ⊗ B`.

## Package features

- `tr`, `det`, `size`, `eltype`, `inv`, ... are efficient functions to work with Kronecker products. Either the result is a numeric value, or returns a new `KroneckerProduct` type.
- Kronecker product - vector multiplications are performed using the vec trick.
- Working with incomplete systems using the [sampled vec trick](https://arxiv.org/pdf/1601.01507.pdf).
- overloading of the the function `eigen` to compute eigenvalue decompostions of Kronecker products. Can be used to efficiently solve systems of the form `(A ⊗ B +λI) \ v`.
- Higher-order Kronecker systems are supported: most functions work on `A ⊗ B ⊗ C` or systems of arbitrary order.
  - Efficient sampling of [Kronecker graphs](https://cs.stanford.edu/~jure/pubs/kronecker-jmlr10.pdf) is supported.
- Kronecker powers are supported: `kronecker(A, 3)` or `A⊗3`.
- A `KroneckerSum` can be constructed with `A ⊕ B` (typed using `\oplus TAB`) or `kroneckersum(A,B)`.
  - Multiplication with vectors uses  a specialized version of the vec trick
  - Higher-order sums are supported, e.g. `A ⊕ B ⊕ C` or `kroneckersum(A,4)`.

## Example use

```julia
julia> A = randn(4, 4);

julia> B = Array{Float64, 2}([1 2 3;
            4 5 6;
            7 -2 9]);

julia> K = A ⊗ B
12×12 Kronecker.KroneckerProduct{Float64,Array{Float64,2},Array{Float64,2}}:
 -1.33612   -2.67224    -4.00836    1.34896    2.69793   4.04689  …  -1.71266   -2.56898   0.959048    1.9181      2.87714
 -5.34448   -6.6806     -8.01672    5.39586    6.74482   8.09378     -4.28164   -5.13797   3.83619     4.79524     5.75429
 -9.35284    2.67224   -12.0251     9.44275   -2.69793  12.1407       1.71266   -7.70695   6.71333    -1.9181      8.63143
  0.182498   0.364996    0.547493   0.775931   1.55186   2.32779     -1.14276   -1.71414   0.0317447   0.0634894   0.0952341
  0.729991   0.912489    1.09499    3.10373    3.87966   4.65559     -2.8569    -3.42828   0.126979    0.158724    0.190468
  1.27748   -0.364996    1.64248    5.43152   -1.55186   6.98338  …   1.14276   -5.14241   0.222213   -0.0634894   0.285702
 -1.0181    -2.03621    -3.05431   -0.770815  -1.54163  -2.31245      0.803382   1.20507  -0.945128   -1.89026    -2.83538
 -4.07241   -5.09051    -6.10862   -3.08326   -3.85408  -4.62489      2.00846    2.41015  -3.78051    -4.72564    -5.67077
 -7.12672    2.03621    -9.16292   -5.39571    1.54163  -6.93734     -0.803382   3.61522  -6.61589     1.89026    -8.50615
  0.665022   1.33004     1.99507    2.29976    4.59953   6.89929      4.01682    6.02524   1.37186     2.74373     4.11559
  2.66009    3.32511     3.99013    9.19906   11.4988   13.7986   …  10.0421    12.0505    5.48746     6.85932     8.23119
  4.65516   -1.33004     5.9852    16.0983    -4.59953  20.6979      -4.01682   18.0757    9.60305    -2.74373    12.3468

julia> tr(K)
18.200512768440117

julia> det(K)
-2.9756571382358265e9

julia> K'
12×12 Kronecker.KroneckerProduct{Float64,Adjoint{Float64,Array{Float64,2}},Adjoint{Float64,Array{Float64,2}}}:
 -1.33612   -5.34448   -9.35284   0.182498    0.729991   1.27748    -1.0181    -4.07241  -7.12672   0.665022   2.66009   4.65516
 -2.67224   -6.6806     2.67224   0.364996    0.912489  -0.364996   -2.03621   -5.09051   2.03621   1.33004    3.32511  -1.33004
 -4.00836   -8.01672  -12.0251    0.547493    1.09499    1.64248    -3.05431   -6.10862  -9.16292   1.99507    3.99013   5.9852
  1.34896    5.39586    9.44275   0.775931    3.10373    5.43152    -0.770815  -3.08326  -5.39571   2.29976    9.19906  16.0983
  2.69793    6.74482   -2.69793   1.55186     3.87966   -1.55186    -1.54163   -3.85408   1.54163   4.59953   11.4988   -4.59953
  4.04689    8.09378   12.1407    2.32779     4.65559    6.98338    -2.31245   -4.62489  -6.93734   6.89929   13.7986   20.6979
 -0.856328  -3.42531   -5.99429  -0.571379   -2.28552   -3.99966     0.401691   1.60676   2.81184   2.00841    8.03365  14.0589
 -1.71266   -4.28164    1.71266  -1.14276    -2.8569     1.14276     0.803382   2.00846  -0.803382  4.01682   10.0421   -4.01682
 -2.56898   -5.13797   -7.70695  -1.71414    -3.42828   -5.14241     1.20507    2.41015   3.61522   6.02524   12.0505   18.0757
  0.959048   3.83619    6.71333   0.0317447   0.126979   0.222213   -0.945128  -3.78051  -6.61589   1.37186    5.48746   9.60305
  1.9181     4.79524   -1.9181    0.0634894   0.158724  -0.0634894  -1.89026   -4.72564   1.89026   2.74373    6.85932  -2.74373
  2.87714    5.75429    8.63143   0.0952341   0.190468   0.285702   -2.83538   -5.67077  -8.50615   4.11559    8.23119  12.3468

julia> K * K  # (A * A) ⊗ (B * B)
12×12 Kronecker.KroneckerProduct{Float64,Array{Float64,2},Array{Float64,2}}:
  106.23      21.2461    148.723      63.2994   12.6599     88.6192  …   11.7334    82.1339    26.5933    5.31866    37.2306
  233.707     74.3613    339.937     139.259    44.3096    202.558       41.067    187.735     58.5052   18.6153     85.0985
  219.543    -49.5742    318.691     130.819   -29.5397    189.898      -27.378    176.001     54.9595  -12.4102     79.7799
   15.018      3.0036     21.0252     40.8506    8.17011    57.1908      -4.59234  -32.1464    23.4969    4.69939    32.8957
   33.0396    10.5126     48.0577     89.8713   28.5954    130.722      -16.0732   -73.4775    51.6933   16.4479     75.1902
   31.0372    -7.00841    45.0541     84.4245  -19.0636    122.552   …   10.7155   -68.8852    48.5604  -10.9652     70.4908
    5.46423    1.09285     7.64992  -133.641   -26.7281   -187.097       -2.54756  -17.8329   -80.3134  -16.0627   -112.439
   12.0213     3.82496    17.4855   -294.009   -93.5484   -427.65        -8.91645  -40.7609  -176.69    -56.2194   -257.003
   11.2927    -2.54997    16.3927   -276.19     62.3656   -400.922        5.9443   -38.2134  -165.981    37.4796   -240.94
  -48.0389    -9.60778   -67.2545    128.652    25.7304    180.113       10.0711    70.4977    20.838     4.1676     29.1732
 -105.686    -33.6272   -153.724     283.034    90.0564    411.686   …   35.2488   161.138     45.8436   14.5866     66.6815
  -99.2804    22.4182   -144.117     265.881   -60.0376    385.956      -23.4992   151.066     43.0652   -9.72439    62.5139

julia> v=rand(12);

julia> K * v
12-element Array{Float64,1}:
  -1.1613188421895284
  -3.3614138427520146
  -0.9837276548605985
   0.3822864526552396
   1.6826148613226284
   6.9235360184821
  -4.990000955828004
 -13.208741091523903
 -20.122579401560937
  13.988391494685379
  37.22488616176438
  40.186818343992286
```
