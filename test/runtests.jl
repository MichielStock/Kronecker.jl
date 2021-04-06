using Kronecker, Test, LinearAlgebra, Random, FillArrays
using SparseArrays: AbstractSparseMatrix, SparseMatrixCSC, sprand,
    sparse, issparse
using Zygote: gradient

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
    include("testchainrules.jl")
end
