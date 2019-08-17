A = rand(3, 3)
B = ones(4, 4)
C = randn(5, 6)
K = A ⊗ B
X = collect(K)
v = rand(12)

K3 = A ⊗ B ⊗ C

@testset "Vec trick" begin

    @test K * v ≈ X * v

    V = rand(4, 3)
    @test K * vec(V) ≈ X * vec(V)
    @test_throws DimensionMismatch K * V
    @test_throws DimensionMismatch K * reshape(V, 2, 6)

    v3 = randn(size(K3, 2))
    @test K3 * v3 ≈ collect(K3) * v3
    u = similar(v)
    @test mul!(u, K, v) ≈ X * v
end

@testset "Reshaped vec trick" begin


    @test K * v ≈ X * v

    V = sprand(4, 3, 1.0)
    @test K * vec(V) ≈ X * vec(V)
    @test_throws DimensionMismatch K * V
    @test_throws DimensionMismatch K * reshape(V, 2, 6)
    K3 = A ⊗ B ⊗ C
    v3 = sprand(size(K3, 2),1.0)
    @test K3 * vec(v3) ≈ collect(K3) * v3
    u = similar(v)
    @test mul!(u, K, v) ≈ X * v

end

@testset "sum" begin
    @test sum(K) ≈ sum(X)
    @test sum(K3) ≈ sum(collect(K3))

    @test sum(K, dims=1) ≈ sum(X, dims=1)
    @test sum(K3, dims=2) ≈ sum(collect(K3), dims=2)
    @test sum(K3, dims=2) isa AbstractKroneckerProduct

    @test sum(kronecker(A, 3)) ≈ sum(kron(A, A, A))
end
