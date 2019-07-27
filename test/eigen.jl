using Random, LinearAlgebra

@testset "eigen" begin
    rng, P, Q = MersenneTwister(123456), 3, 5

    # Generate some positive definite matrices so that logdet can be tested.
    A_, B_ = randn(rng, P, P), randn(rng, Q, Q)
    A, B = Symmetric(A_ * A_' + I), Symmetric(B_ * B_' + I)

    # Check approximate correctness of decomposition
    C = kronecker(A, B)
    λ, Γ = eigen(C)
    @test Γ * Diagonal(λ) * Γ' ≈ C

    # Check approximate correctness of shifted decomposition
    σ² = abs2(randn(rng))
    λ′, Γ′ = (eigen(C) + σ² * I)
    @test Γ′ * Diagonal(λ′) * Γ′' ≈ Matrix(C + σ² * I)

    # Test with decomposition
    v = randn(rng, P * Q)
    @test eigen(C) \ v ≈ Matrix(C) \ v

    # Test various linear algebra operations with decomposition
    @test det(eigen(C)) ≈ det(C)
    @test logdet(eigen(C)) ≈ log(det(C))
    @test Matrix(inv(eigen(C))) ≈ inv(C)
end
