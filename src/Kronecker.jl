module Kronecker

export GeneralizedKroneckerProduct, AbstractKroneckerProduct, AbstractSquareKronecker
export SquareKroneckerProduct, EigenKroneckerProduct, ShiftedKroneckerProduct
export AbstractKroneckerSum, KroneckerSum
export issquare, getmatrices, size, getindices, order, issymmetric, isposdef
export ⊗, kronecker, Matrix, ⊕, kroneckersum
export tr, det, logdet, collect, inv, +, *, mult!, eigen, \, /, adjoint, transpose, conj, solve
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
