@testset "Kronecker products" begin

    A = randn(4, 4)
    B = Array{Float64, 2}([1 2 3;
         4 5 6;
         7 -2 9])
    C = rand(5, 6)

    v = rand(12)

    K = A ⊗ B

    X = kron(A, B)  # true result

    @testset "Types and basic properties" begin
        @test issquare(A)
        @test !issquare(C)

        @test issquare(K)

        @test K ≈ X

        @test order(A) == 1
        @test order(K) == 2

        for j in 1:12
            for i in 1:12
                @test K[i,j] ≈ X[i,j]
            end
        end
    end

    @testset "Linear algebra" begin
        @test tr(K) ≈ tr(X)
        @test det(K) ≈ det(X)
        @test !isposdef(K)
        @test transpose(K) ≈ transpose(X)
        @test conj(K) ≈ conj(X)
        @test K' ≈ X'
        @test inv(K) ≈ inv(X)

        # test on pos def functions
        As = A' * A
        Bs = B * B'
        @test logdet(As ⊗ Bs) ≈ log(det(As ⊗ Bs)) ≈ log(det(kron(As, Bs)))
    end

    @testset "Vec trick" begin
        @test K * v ≈ X * v
        K3 = A ⊗ B ⊗ C
        v3 = randn(size(K3, 2))
        @test K3 * v3 ≈ collect(K3) * v3
    end

    @testset "Higher order" begin
        @test order(K ⊗ A) == 3
        K3 = kronecker(A, B, C)

        @test order(K3) == 3
        @test collect(K3) ≈ kron(X, C)

    end

    @testset "Kronecker powers" begin
        Kpow = ⊗(A, 5)
        @test order(Kpow) == 5
        @test size(Kpow, 1) == 4^5
        @test Kpow[1,1] ≈ A[1,1]^5
    end

    @testset "Mixed product" begin
        A = rand(5, 4)
        B = rand(2, 3)
        C = rand(4, 6)
        D = rand(3, 4)

        K1 = (A ⊗ B)
        K2 = (C ⊗ D)

        @test K1 * K2 ≈ collect(K1) * collect(K2)
    end

end
