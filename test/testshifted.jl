A = Symmetric(rand(10, 10))
B = Symmetric(rand(30, 30))
v = rand(300)

K = A ⊗ B

@test issymmetric(K)

rnaive = (kron(A, B) + 2I) \ v

@test rnaive ≈ (A ⊗ B + 2I) \ v
@test rnaive ≈ v / (A ⊗ B + 2I)

# also works with non-symmetric matrices

A = rand(10, 10) + 2I
B = rand(30, 30) + 2I
v = rand(300)

K = A ⊗ B

@test !issymmetric(K)

rnaive = (kron(A, B) + 5I) \ v

@test rnaive ≈ (A ⊗ B + 5I) \ v
@test rnaive ≈ v / (A ⊗ B + 5I)
