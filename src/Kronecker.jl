module Kronecker

export GeneralizedKroneckerProduct, AbstractKroneckerProduct, AbstractSquareKronecker
export AbstractKroneckerSum, KroneckerSum
export SquareKroneckerProduct, EigenKroneckerProduct, ShiftedKroneckerProduct
export issquare, getmatrices, size, getindices, order, issymmetric, isposdef
export ⊗, kronecker, Matrix, ⊕, kroneckersum
export tr, det, logdet, collect, inv, +, *, mult!, eigen, /, adjoint, transpose, conj, solve, exp
export getindex
export cholesky, CholeskyKronecker

export genvectrick, genvectrick!

using LinearAlgebra
import LinearAlgebra: mul!
import Base: collect

include("base.jl")
include("indexedkroncker.jl")
include("eigen.jl")
include("factorization.jl")
include("kroneckersum.jl")

end # module
