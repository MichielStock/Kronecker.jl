using NamedDims
@testset "named" begin
    A = rand(Int8, 2,2)
    B = rand(Int8, 3,3)
    ndA = NamedDimsArray(A, (:ia, :ja))
    ndB = NamedDimsArray(B, (:ib, :jb))

    @test NamedDims.names(kronecker(ndA, ndB)) == (Symbol("iaᵡib"), Symbol("jaᵡjb"))
    @test NamedDims.names(kronecker(ndA, B)) == (Symbol("iaᵡ_"), Symbol("jaᵡ_"))
    @test NamedDims.names(kronecker(A, ndB)) == (Symbol("_ᵡib"), Symbol("_ᵡjb"))

    sum(abs2, kronecker(ndA, ndB) .- kron(A, B)) == 0

    @test 0 == @allocated Kronecker._join(:i, :j)
    @test 0 == @allocated Kronecker.kron_names((:ia, :ja), (:ib, :jb))

    @test NamedDims.names(kronecker(ndA, 3)) == (Symbol("iaᵡiaᵡiaᵡia"), Symbol("jaᵡjaᵡjaᵡja"))
    @test 0 == @allocated Kronecker.kron_names((:i, :j), Val(3))
end
