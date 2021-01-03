@testset "sums" begin

    A = rand(10, 10)
    B = rand(Bool, 5, 5)
    C = randn(10, 10)
    D = rand(1:100, 5, 5)

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


end