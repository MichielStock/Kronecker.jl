@testset "Vec trick" begin
    A = rand(3, 3)
    B = ones(4, 4)
    C = randn(5, 6)
    K = A ⊗ B
    X = collect(K)
    v = rand(12)

    @test K * v ≈ X * v

    V = rand(4, 3)
    @test K * vec(V) ≈ X * vec(V)
    @test_throws DimensionMismatch K * V
    @test_throws DimensionMismatch K * reshape(V, 2, 6)
    K3 = A ⊗ B ⊗ C
    v3 = randn(size(K3, 2))
    @test K3 * v3 ≈ collect(K3) * v3
    u = similar(v)
    @test mul!(u, K, v) ≈ X * v
end
