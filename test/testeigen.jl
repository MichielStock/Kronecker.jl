using Random, LinearAlgebra

# Standardised tests for the eigen decomposition of a square kronecker product
function eigen_tests(rng, C::AbstractKroneckerProduct)

    # Check approximate correctness of decomposition
    λ, Γ = eigen(C)
    @test Γ * Diagonal(λ) * Γ' ≈ C
    @test Γ * (Diagonal(λ) * Γ') ≈ C
    @test (Γ * Diagonal(λ)) * Γ' ≈ C

    # Check approximate correctness of shifted decomposition
    σ² = abs2(randn(rng))
    λ′, Γ′ = (eigen(C) + σ² * I)
    @test Γ′ * Diagonal(λ′) * Γ′' ≈ Matrix(C + σ² * I)
    @test σ² * I + C == C + σ² * I

    # Test with decomposition
    v = randn(rng, length(λ))
    @test eigen(C) \ v ≈ Float64.(Matrix(C)) \ v

    # Test various linear algebra operations with decomposition
    C_dense = Float64.(Matrix(C))
    @test det(eigen(C)) ≈ det(C_dense)
    @test logdet(eigen(C)) ≈ logdet(C_dense)
    @test Matrix(inv(eigen(C))) ≈ inv(C_dense)
end

@testset "eigen" begin
    rng, P, Q, R = MersenneTwister(123456), 3, 5, 7

    # Generate some positive definite matrices so that logdet can be tested.
    A_, B_, C_ = randn(rng, P, P), randn(rng, Q, Q), randn(rng, R, R)
    A, B, C = Symmetric(A_ * A_' + I), Symmetric(B_ * B_' + I), Symmetric(C_ * C_' + I)

    D = kronecker(A, B)
    eigen_tests(rng, D)
    eigen_tests(rng, kronecker(D, C))
    eigen_tests(rng, kronecker(C, D))
    eigen_tests(rng, kronecker(kronecker(A, B), kronecker(B, A)))
    eigen_tests(rng, kronecker(A, 4))

    function test_non_square_extension()
        local n, m, A, B, K, M
        n, m = 3, 5
        # 1. K is square, while A, B aren't
        A = randn(n, m)
        B = randn(m, n)
        K = A ⊗ B
        M = Matrix(K)
        E = eigen(K)
        F = eigen(M)
        @test E.values ≈ F.values
        @test det(E) ≈ det(F)
        @test Matrix(E) ≈ K

        # 2. K is not square
        A = randn(n, m)
        B = randn(m, m)
        K = A ⊗ B
        @test_throws DimensionMismatch eigen(K)
    end
    test_non_square_extension()
end

@testset "eigvals" begin
    A = rand(3,3); B = rand(4, 4);
    K = kronecker(A, B);
    Kd = kron(A, B);
    @test eigvals(K) ≈ eigvals(Kd)
    if VERSION >= v"1.6"
        @test real(eigvals(K, sortby = real)) ≈ real(eigvals(Kd, sortby = real)) atol=1e-14 rtol=1e-8
    end
    A = rand(2,3); B = rand(3, 2);
    K = kronecker(A, B);
    Kd = kron(A, B);
    @test eigvals(K) ≈ eigvals(Kd)
end
