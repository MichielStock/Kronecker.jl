function test_gradients_for_KroneckerProduct(M_out, N_out)
    M, N = 3, 2
    n_samples = 3
    A = rand(N_out, N)
    B = rand(M_out, M)
    x = rand(M*N, n_samples)
    y = rand(M_out*N_out, n_samples)

    eager_model(A, B, X) = kron(A, B) * X

    function loss(A, B, X)
        Z = eager_model(A, B, X) - y
        L = 0.5 * tr(Z' * Z)
        return L
    end

    lazy_model(A, B, X) = (A ⊗ B) * X

    function lazy_loss(A, B, X)
        Z = lazy_model(A, B, X) - y
        L = 0.5 * tr(Z' * Z)
        return L
    end

    function gradient_A(A, B, x)
        Z = eager_model(A, B, x) - y
        m, n = size(A)
        IA = Diagonal(ones(m*n))
        return Z * (kron(IA', B) * x)'
    end

    function gradient_B(A, B, x)
        Z = eager_model(A, B, x) - y
        m, n = size(B)
        IB = Diagonal(ones(m*n))
        return  Z * (kron(A, IB) * x)'
    end

    function gradient_x(A, B, x)
        Z = eager_model(A, B, x) - y
        return kron(A, B)'*Z
    end

    if (M_out, N_out) == (1,1)
        @testset "Gradients for M_out=$M_out, N_out=$N_out" begin
            gA, gB, gx = gradient(loss, A, B, x)
            # Compare hand-written gradients with running Zygote.gradient on the loss function
            @test gradient_A(A, B, x) ≈ gA
            @test gradient_B(A, B, x) ≈ gB
            @test gradient_x(A, B, x) ≈ gx
            # Compare `Base.kron` with `Kronecker.kronecker` in Zygote
            @test all(gradient(loss, A, B, x) .≈ gradient(lazy_loss, A, B, x))
        end
    else
        @testset "Gradients for M_out=$M_out, N_out=$N_out" begin
            gA, gB, gx = gradient(loss, A, B, x)
            # Compare hand-written gradients with running Zygote.gradient on the loss function
            @test_broken gradient_A(A, B, x) ≈ gA
            @test_broken gradient_B(A, B, x) ≈ gB
            @test gradient_x(A, B, x) ≈ gx
            # Compare `Base.kron` with `Kronecker.kronecker` in Zygote
            @test_broken all(gradient(loss, A, B, x) .≈ gradient(lazy_loss, A, B, x))
        end
    end
end

# factors A and B in (A⊗B)*x : [M_out*N_out, n_samples=3]
for (Mo, No) in ((1,1), (2, 3))
    test_gradients_for_KroneckerProduct(Mo, No)
end
