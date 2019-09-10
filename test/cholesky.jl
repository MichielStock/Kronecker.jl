to_psd(A) = A * A' + I

@testset "cholesky" begin
    rng = MersenneTwister(123456)
    M, N = 7, 3
    A, B = to_psd(randn(rng, M, M)), to_psd(randn(rng, N, N))

    # Construct Kronecker-factored Cholesky
    A_kron_B = A ⊗ B
    chol_A_kron_B = cholesky(A_kron_B)

    # Construct equivalent dense Cholesky.
    A_kron_B_dense = kron(A, B)
    chol_A_kron_B_dense = cholesky(A_kron_B_dense)

    # Check for agreement in user-facing properties of Kronecker-factored and dense.
    @test chol_A_kron_B.U ≈ chol_A_kron_B_dense.U
    @test chol_A_kron_B.L ≈ chol_A_kron_B_dense.L
    @test det(chol_A_kron_B) ≈ det(chol_A_kron_B_dense)
    @test logdet(chol_A_kron_B) ≈ logdet(chol_A_kron_B_dense)


    # Test backsolve vs dense vector from the left.
    x = randn(rng, M * N)
    @test chol_A_kron_B \ x ≈ chol_A_kron_B_dense \ x

    # Test backsolve vs dense matrix from the left.
    X = randn(rng, M * N, 11)
    @test chol_A_kron_B \ X ≈ chol_A_kron_B_dense \ X
end
