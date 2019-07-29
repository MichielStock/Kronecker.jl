@testset "Kronecker sums" begin


    A = rand(4,4); IA = oneunit(A)
    B = rand(3,3); IB = oneunit(B)

    KS = A ⊕ B
    kronsum = kron(A,IB) + kron(IA,B)

    @test collect(KS) == kronsum

    @test issquare(KS)


    C = rand(5,5); IC = oneunit(C)
    KS3 = A ⊕ B ⊕ C
    kronsum3 = kron(A,IB,IC) + kron(IA,B,IC) + kron(IA,IB,C)

    @test collect(KS3) ≈ kronsum3

    @test order(KS) == 2
    @test order(KS3) == 3

    @test getmatrices(KS) == (A,B)

    @test getindex(KS,2,3) == kronsum[2,3]
    @test getindex(KS3,2,3) == kronsum3[2,3]

    D = rand(ComplexF64,6,6); ID = oneunit(D)

    @testset "Structure of sums" begin
        @test size(A ⊕ D) == size(A) .* size(D,1)
        @test size(B ⊕ C ⊕ D) == size(B) .* size(C) .* size(D)
        @test eltype(A ⊕ B) == Float64
        @test eltype(C ⊕ D) == ComplexF64
    end

    @testset "Basic linear algebra for sums" begin
        @test tr(KS) ≈ tr(kronsum)
        @test KS' == kronsum'
        @test transpose(KS) == transpose(kronsum)
        @test conj(KS) == conj(kronsum)
    end

    A = rand(10,10); B = rand(10,10); V = Diagonal(rand(10))
    @testset "Vec trick for sums" begin
        @test (A ⊕ B) * vec(V) == vec(B*V + V*transpose(A))
    end

    A = rand(3, 3); IA = oneunit(A)
    B = rand(4, 4); IB = oneunit(B)
    @testset "exp for Kronecker sum" begin
        EKS = exp(A ⊕ B)
        @test EKS isa SquareKroneckerProduct
        @test EKS ≈ exp(kron(A, IB) + kron(IA, B))
    end
end
