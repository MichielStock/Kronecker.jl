module Kronecker

export GeneralizedKroneckerProduct, AbstractKroneckerProduct
export AbstractKroneckerSum, KroneckerSum
export EigenKroneckerProduct
export issquare, order, issymmetric, isposdef, getmatrices
export ⊗, kronecker, ⊕, kroneckersum
export CholeskyKronecker

using LinearAlgebra
import LinearAlgebra: mul!
import Base: collect
using SparseArrays: sparse

include("base.jl")
include("vectrick.jl")
include("indexedkroncker.jl")
include("eigen.jl")
include("factorization.jl")
include("kroneckersum.jl")

end # module
