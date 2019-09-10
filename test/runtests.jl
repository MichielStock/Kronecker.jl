using Kronecker, Test, LinearAlgebra, Random, FillArrays
using SparseArrays: SparseMatrixCSC, sprand, AbstractSparseMatrix

@testset "Kronecker" begin
    # include("base.jl")
    # include("kroneckerpowers.jl")
    # include("vectrick.jl")
    # include("indexed.jl")
    # include("eigen.jl")
    include("cholesky.jl")
    # include("kroneckersum.jl")
    # include("factorization.jl")
    # include("kroneckergraphs.jl")
end
