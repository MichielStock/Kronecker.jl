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

end
