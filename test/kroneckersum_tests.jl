@testset "Kronecker sums" begin


    A = rand(4,4); IA = oneunit(A)
    B = rand(3,3); IB = oneunit(B)

    KS = A ⊕ B
    kronsum = kron(A,IB) + kron(IA,B)

    @testset "A ⊕ B factors" begin
        @test KS.A isa SquareKroneckerProduct && KS.B isa SquareKroneckerProduct
        @test KS.A == kronecker(A,IB) && KS.B == kronecker(IA,B)
    end
    @test collect(KS) == kronsum



    C = rand(5,5); IC = oneunit(C)
    KS3 = A ⊕ B ⊕ C
    kronsum3 = kron(A,IB,IC) + kron(IA,B,IC) + kron(IA,IB,C)

    @testset "A ⊕ B ⊕ C factors" begin
        @test KS3.A.A.A.A == A
        @test KS3.A.A.B.B == B
        @test KS3.B.B == C
    end

    # Can't collect higher order sums
    @test_broken collect(KS3) = kronsum


    @test order(KS) == 2
    @test order(KS3) == 3

    @test getmatrices(KS) == (A,B)
    @test_broken getmatrices(KS3) == (A,B,C)

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
end
