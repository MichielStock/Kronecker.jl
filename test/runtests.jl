using Kronecker, Test, LinearAlgebra, Random, FillArrays
using SparseArrays: AbstractSparseMatrix, SparseMatrixCSC, sprand,
    sparse, issparse

@testset "Kronecker" begin
    include("testbase.jl")
    include("testkroneckerpowers.jl")
    include("testnames.jl")
    include("testvectrick.jl")
    include("testindexed.jl")
    include("testeigen.jl")
    include("testkroneckersum.jl")
    include("testfactorization.jl")
    include("testkroneckergraphs.jl")
end
