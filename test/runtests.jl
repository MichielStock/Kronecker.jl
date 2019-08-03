using Kronecker, Test, LinearAlgebra
using SparseArrays: SparseMatrixCSC

@testset "Kronecker" begin
    include("testbase.jl")
    include("testvectrick.jl")
    include("testindexed.jl")
    include("testeigen.jl")
    include("testkroneckersum.jl")
    include("testfactorization.jl")
end
