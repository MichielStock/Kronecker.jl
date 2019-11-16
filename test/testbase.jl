@testset "Kronecker products" begin

    A = randn(4, 4)
    B = Array{Float64, 2}([1 2 3;
         4 5 6;
         7 -2 9])
    C = rand(5, 6)
    D = rand(4, 4)

    v = rand(12)

    K = A ⊗ B
    K3 = kronecker(A, B, C)

    X = kron(A, B)  # true result

    @testset "Types and basic properties" begin
        @test issquare(A)
        @test !issquare(C)

        @test getmatrices(A)[1] === A

        @test !issymmetric(A)

        @test issquare(K)
        @test !issymmetric(K)

        @test K ≈ X

        @test order(A) == 1
        @test order(K) == 2

        @test eltype(K) <: Float64

        @test K isa AbstractKroneckerProduct
        @test K isa AbstractKroneckerProduct{Float64}

        @test K isa AbstractMatrix{Float64}
    end

    # testing all linear algebra functions' behavior on square and non-square matrices
    function test_non_square_extensions()
        local n, m, A, B, K, M
        n, m = 3, 5
        # 1. K is square, while A, B aren't
        A = randn(n, m)
        B = randn(m, n)
        K = A ⊗ B
        M = Matrix(K)
        @test tr(K) ≈ tr(M)
        @test det(K) ≈ 0
        @test logdet(K) ≈ -Inf
        @test !isposdef(K)
        @test !issymmetric(K)
        @test LinearAlgebra.checksquare(K) == size(K, 1) # should not throw on square matrix
        @test_throws SingularException inv(K) # if K is square but singular

        # 2. K is not square
        A = randn(n, m)
        B = randn(m, m)
        K = A ⊗ B
        @test !isposdef(K)
        @test !issymmetric(K)
        @test_throws DimensionMismatch tr(K)
        @test_throws DimensionMismatch det(K)
        @test_throws DimensionMismatch logdet(K)
        @test_throws DimensionMismatch LinearAlgebra.checksquare(K)
        @test_throws DimensionMismatch inv(K)
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

        test_non_square_extensions()
    end

    @testset "Mismatch errors" begin
        P, Q = rand(10, 4), rand(4, 5)
        Kns = P ⊗ Q
        @test_throws DimensionMismatch inv(Kns)
        @test_throws DimensionMismatch det(Kns)
        @test_throws DimensionMismatch Kns * [1, 2, 3]
    end

    @testset "Higher order" begin
        @test order(K ⊗ A) == 3
        @test order(K3) == 3
        @test collect(K3) ≈ kron(X, C)

    end

    @testset "Kronecker powers" begin
        Kpow = ⊗(A, 5)
        @test order(Kpow) == 5
        @test size(Kpow, 1) == 4^5
        @test Kpow[1,1] ≈ A[1,1]^5
    end

    @testset "kron" begin
        @test kron(A ⊗ B, C) ≈ kron(A, B, C)
        @test kron(A, B ⊗ C) ≈ kron(A, B, C)
        @test kron(A ⊗ B, C ⊗ D) ≈ kron(A, B, C, D)
        @test collect((A⊗B) ⊗ (C⊗D)) ≈ kron(A, B, C, D)
    end

    @testset "Mixed product" begin
        A = rand(5, 4)
        B = rand(2, 3)
        C = rand(4, 6)
        D = rand(3, 4)

        K1 = (A ⊗ B)
        K2 = (C ⊗ D)

        @test K1 * K2 ≈ collect(K1) * collect(K2)

        @test_throws DimensionMismatch (A ⊗ C) * (B ⊗ D)
        @test_throws DimensionMismatch (A ⊗ D) * (C ⊗ B)
    end

    @testset "Add to dense" begin
        @test K + X ≈ Matrix(K) + X
        @test X + K ≈ X + Matrix(K)
    end

    @testset "Scalar multiplication" begin
        @test 3.0K ≈ 3.0X
        @test K * 2 ≈ 2X
        @test π * K3 ≈ π * collect(K3)
        @test 3.0K isa AbstractKroneckerProduct
        @test K * 2 isa AbstractKroneckerProduct
        @test 2(K ⊗ K) isa AbstractKroneckerProduct
    end

    @testset "Inplace scalar multiplication" begin
        A = rand(2, 2)
        B = rand(3, 4)

        K = copy(A) ⊗ copy(B)
        lmul!(3, K)
        rmul!(K, 2)
        @test K.A ≈ 3A
        @test K.B ≈ 2B
        @test K ≈ 6kron(A, B)
    end

    @testset "Solving Linear Systems" begin
        n = 8
        m = 16
        # 1) testing square kronecker product with square A, B
        A = randn(n, n)
        B = randn(m, m)
        K = A ⊗ B
        x = randn(n * m)
        b = K*x
        @test K\b ≈ x
        @test (x'*K)/K ≈ x'

        # testing least squares solution
        function test_ls_solve(size_A, size_B)
            A = randn(size_A)
            B = randn(size_B)
            x = randn(size(A, 2) * size(B, 2))
            K = kronecker(A, B)
            b = K*x
            b .+= randn(size(b)) # this moves b out of range(K), necessitating least-squares
            xls = K\b
            return K'*(K*xls) ≈ K'b # test via normal equations
        end

        # 2) kronecker product is square, but A, B aren't
        size_A = (m, n)
        size_B = (n, m)
        @test test_ls_solve(size_A, size_B)

        # 3) kronecker product is not square, dimension of x is larger than b
        size_A = (n, n)
        size_B = (n, m)
        @test test_ls_solve(size_A, size_B)

        # 4) kronecker product is not square, dimension of x is smaller than b
        size_A = (m, n)
        size_B = (m, m)
        @test test_ls_solve(size_A, size_B)
    end
end
