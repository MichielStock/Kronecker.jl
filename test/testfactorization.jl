using LinearAlgebra: cholesky

@testset "Factorization" begin
    @testset "Cholesky" begin
        A = [1 0 0.5;
             0 2 0;
             0.5 0 3]

        B = rand(4)
        B *= B'
        B += I

        K = A ⊗ B

        @test isposdef(K)

        KC = cholesky(K)
        @test isposdef(K)
        @test size(K) == (12, 12)

        C = cholesky(collect(K))

        @test det(KC) ≈ det(C)
        @test logdet(KC) ≈ logdet(C)
        @test collect(inv(KC)) ≈ inv(C)
    end
end
