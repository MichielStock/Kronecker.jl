module Kronecker

# TODO types!
export GeneralizedKroneckerProduct, AbstractKroneckerProduct, SquareKroneckerProduct, EigenKroneckerProduct, ShiftedKroneckerProduct
export issquare, getmatrices, size, getindices, order, issymmetric
export ⊗, kronecker
export tr, det, collect, inv, *, mult!, eigen, \, /, adjoint
export getindex

export genvectrick, genvectrick!

using LinearAlgebra

include("base.jl")
include("indexedkroncker.jl")
include("shiftedkronecker.jl")

end # module
