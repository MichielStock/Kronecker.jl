@testset "sums" begin

    A = rand(10, 10)
    B = rand(5, 5)
    C = randn(10, 10)
    D = rand(1:10.0, 5, 5)

    KA = A ⊗ B
    KB = C ⊗ D

    K = KA + KB
    Kdense = kron(A, B) .+ kron(C, D)

    v = randn(50)
    V = randn(50, 3)

    @test K ≈ Kdense
    @test size(K) == size(Kdense)
    @test collect(K) ≈ Kdense
    @test sum(K) ≈ sum(Kdense)
    @test sum(K, dims=1) ≈ sum(Kdense, dims=1)
    @test K' ≈ Kdense'
    @test transpose(K) ≈ transpose(Kdense)
    @test conj(K) ≈ conj(Kdense)

    @test K * v ≈ Kdense * v
    @test K * V ≈ Kdense * V

    @test 5 * K ≈ 5 * Kdense
    @test K * 2.1 ≈ Kdense * 2.1

    # recursive, kronecker products of Kronecker sums


    
    Kl = (KA + KB) ⊗ C
    Kr = C ⊗ (KA + KB)
    Kb = ((B ⊗ C) + (B ⊗ C)) ⊗ ((B ⊗ C) + (B ⊗ C))
    
    #=
    Kl = kronecker(KA + KB, C)
    Kr = kronecker(C, KA + KB) 
    Kb = kronecker(kronecker(B, C) + kronecker(B, C), kronecker(B, C) + kronecker(B, C)) 
    =#

    n = size(Kl, 2)
    m = size(Kb, 2)

    v = randn(n)
    u = randn(m)

    
    @test Kl * v ≈ collect(Kl) * v
    @test Kr * v ≈ collect(Kr) * v
    @test Kb * u ≈ collect(Kb) * u
    


end