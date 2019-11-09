module Kronecker

export GeneralizedKroneckerProduct, AbstractKroneckerProduct
export AbstractKroneckerSum, KroneckerSum
export KroneckerPower
export issquare, order, issymmetric, isposdef, getmatrices
export ⊗, kronecker, ⊕, kroneckersum
export CholeskyKronecker

export isprob, naivesample, fastsample, sampleindices

using LinearAlgebra, FillArrays
import LinearAlgebra: mul!, lmul!, rmul!
import Base: collect, *, getindex, size, eltype, inv, adjoint
using SparseArrays: sparse

include("base.jl")
include("kroneckerpowers.jl")
include("vectrick.jl")
include("indexedkroncker.jl")
#include("eigen.jl")
include("factorization.jl")
#include("kroneckersum.jl")
include("kroneckergraphs.jl")

end # module
