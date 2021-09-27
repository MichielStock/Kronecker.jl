using Kronecker, Test, LinearAlgebra, Random
using SparseArrays: AbstractSparseMatrix, SparseMatrixCSC, sprand,
    sparse, issparse

using Aqua
@testset "project quality" begin
    Aqua.test_all(Kronecker, ambiguities=false)
end

@testset "Kronecker" begin

    include("testbase.jl")
    include("testkroneckerpowers.jl")
    include("testnames.jl")
    include("testmultiply.jl")
    include("testindexed.jl")
    include("testeigen.jl")
    include("testkroneckersum.jl")
    include("testfactorization.jl")
    include("testkroneckergraphs.jl")
end
