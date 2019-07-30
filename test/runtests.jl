using Kronecker, Test, LinearAlgebra

@testset "Kronecker" begin
    include("testbase.jl")
    include("testindexed.jl")
    include("testeigen.jl")
    include("testkroneckersum.jl")
    include("testfactorization.jl")
end
