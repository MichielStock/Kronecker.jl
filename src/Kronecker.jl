module Kronecker

# TODO types!
export GeneralizedKroneckerProduct, AbstractKroneckerProduct, SquareKroneckerProduct, EigenKroneckerProduct, ShiftedKroneckerProduct
export issquare, getmatrices, size, getindices, order, issymmetric
export ⊗, kronecker, Matrix
export tr, det, collect, inv, *, mult!, eigen, \, /, adjoint, transpose, solve
export getindex

export genvectrick, genvectrick!

using LinearAlgebra

include("base.jl")
include("indexedkroncker.jl")
include("shiftedkronecker.jl")

end # module
