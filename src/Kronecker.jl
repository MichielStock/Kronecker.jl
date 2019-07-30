module Kronecker

export GeneralizedKroneckerProduct, AbstractKroneckerProduct, AbstractSquareKronecker
export AbstractKroneckerSum, KroneckerSum
export SquareKroneckerProduct, EigenKroneckerProduct, ShiftedKroneckerProduct
export issquare, order, issymmetric, isposdef, getmatrices
export ⊗, kronecker, ⊕, kroneckersum
export CholeskyKronecker

export genvectrick, genvectrick!

using LinearAlgebra
import LinearAlgebra: mul!
import Base: collect
using SparseArrays: sparse

include("base.jl")
include("indexedkroncker.jl")
include("eigen.jl")
include("factorization.jl")
include("kroneckersum.jl")

end # module
