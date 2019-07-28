module Kronecker

# TODO types!
export GeneralizedKroneckerProduct, AbstractKroneckerProduct, KroneckerProduct, SquareKroneckerProduct, EigenKroneckerProduct, ShiftedKroneckerProduct
export AbstractKroneckerSum, KroneckerSum
export issquare, getmatrices, size, getindices, order, issymmetric
export ⊗, kronecker, Matrix, ⊕, kroneckersum
export tr, det, collect, inv, *, mult!, eigen, \, /, adjoint, transpose, conj, solve
export getindex

export genvectrick, genvectrick!

using LinearAlgebra
import LinearAlgebra: mul!

include("base.jl")
include("indexedkroncker.jl")
include("shiftedkronecker.jl")
include("eigen.jl")
include("kroneckersum.jl")

end # module
