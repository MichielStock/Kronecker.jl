module Kronecker

# TODO types!
export GeneralizedKroneckerProduct, AbstractKroneckerProduct, SquareKroneckerProduct, EigenKroneckerProduct, ShiftedKroneckerProduct
export issquare, getmatrices, size, getindices, order, issymmetric
export ⊗, kronecker, Matrix
export tr, det, logdet, collect, inv, *, mult!, eigen, \, /, adjoint, transpose, conj, solve
export getindex

export genvectrick, genvectrick!

using LinearAlgebra
import LinearAlgebra: mul!

include("base.jl")
include("indexedkroncker.jl")
include("shiftedkronecker.jl")

end # module
