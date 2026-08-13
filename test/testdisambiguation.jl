using NamedDims

# Tests exercising calls that used to be method-ambiguous (Aqua's ambiguity
# check is now fully enabled; every fix below pins the resolved behaviour).
@testset "method disambiguation" begin
    A = randn(4, 4)
    B = randn(3, 3)
    K = A ⊗ B
    Kdense = kron(A, B)

    @testset "lazy Kronecker times lazy Kronecker fallback" begin
        S = A ⊕ B  # KroneckerSum is a GeneralizedKroneckerProduct
        Sdense = collect(S)
        @test K * S ≈ Kdense * Sdense
        @test collect(S * S) ≈ Sdense * Sdense
    end

    @testset "mul! of KroneckerSum with triangular matrix" begin
        S = A ⊕ B
        T = UpperTriangular(randn(12, 12))
        C = zeros(12, 12)
        mul!(C, S, T)
        @test C ≈ collect(S) * T
    end

    @testset "adjoint and transpose row vectors" begin
        v = randn(ComplexF64, 12)
        @test v' * K ≈ v' * Kdense
        @test transpose(v) * K ≈ transpose(v) * Kdense
    end

    @testset "Eigen solve with complex right-hand side" begin
        P = A * A' + 4I  # positive definite, real spectrum
        Q = B * B' + 4I
        E = eigen(P ⊗ Q)
        v = randn(ComplexF64, 12)
        @test E \ v ≈ kron(P, Q) \ v
    end

    @testset "copyto! into PermutedDimsArray" begin
        dest = PermutedDimsArray(zeros(12, 12), (2, 1))
        copyto!(dest, K)
        @test dest ≈ Kdense
        destc = PermutedDimsArray(zeros(ComplexF64, 12, 12), (2, 1))
        copyto!(destc, K)
        @test destc ≈ Kdense
    end

    @testset "kron of product and sum" begin
        S = A ⊕ B
        @test kron(S, K) ≈ kron(collect(S), Kdense)
        @test kron(K, S) ≈ kron(Kdense, collect(S))
    end

    @testset "NamedDims operands" begin
        v = NamedDimsArray(randn(12), (:i,))
        @test K * v ≈ Kdense * parent(v)

        M = NamedDimsArray(randn(12, 2), (:i, :j))
        @test K * M ≈ Kdense * parent(M)
        @test dimnames(K * M) == (:_, :j)
        N = NamedDimsArray(randn(2, 12), (:i, :j))
        @test N * K ≈ parent(N) * Kdense
        @test dimnames(N * K) == (:i, :_)

        D = Diagonal(randn(4)) ⊗ Diagonal(randn(3))
        @test D * v ≈ collect(D) * parent(v)

        p, q = rand(1:4, 6), rand(1:3, 6)
        r, t = rand(1:4, 5), rand(1:3, 5)
        ikp = (B ⊗ A)[p, q, r, t]
        u = NamedDimsArray(randn(5), (:i,))
        @test ikp * u ≈ collect(ikp) * parent(u)
    end
end
