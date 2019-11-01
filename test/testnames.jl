using NamedDims
@testset "named" begin
    A = rand(Int8, 2,2)
    B = rand(Int8, 3,3)
    ndA = NamedDimsArray(A, (:xA, :yA))
    ndB = NamedDimsArray(B, (:xB, :yB))

    @test NamedDims.names(kronecker(ndA, ndB)) == (Symbol("xA⊗xB"), Symbol("yA⊗yB"))
    @test NamedDims.names(kronecker(ndA, B)) == (Symbol("xA⊗_"), Symbol("yA⊗_"))
    @test NamedDims.names(kronecker(A, ndB)) == (Symbol("_⊗xB"), Symbol("_⊗yB"))

    sum(abs2, kronecker(ndA, ndB) .- kron(A, B)) == 0

    @test 0 == @allocated Kronecker._join(:i, :j)
    @test 0 == @allocated Kronecker.kron_names((:xA, :yA), (:xB, :yB))

    @test NamedDims.names(kronecker(ndA, 3)) == (Symbol("xA⊗xA⊗xA⊗xA"), Symbol("yA⊗yA⊗yA⊗yA"))
    @test 0 == @allocated Kronecker.kron_names((:i, :j), Val(3))
end
