A = randn(4, 4)
B = Array{Float64, 2}([1 2 3;
     4 5 6;
     7 -2 9])
C = rand(5, 6)

v = rand(12)

@test issquare(A)
@test !issquare(C)

kronprod = A ⊗ B

@test issquare(kronprod)

X = kron(A, B)  # true result

@test tr(kronprod) ≈ tr(X)
@test det(kronprod) ≈ det(X)
@test collect(transpose(kronprod)) ≈ transpose(X)
@test collect(conj(kronprod)) ≈ conj(X)
@test collect(kronprod') ≈ X'
@test collect(inv(kronprod)) ≈ inv(X)
@test all(kronprod * v .≈ X * v)

# test on pos def functions
As = A' * A
Bs = B * B'

@test logdet(As ⊗ Bs) ≈ logdet(kron(As, Bs))


@test order(A) == 1
@test order(kronprod) == 2
@test order(kronprod ⊗ A) == 3

K3 = kronecker(A, B, C)

@test order(K3) == 3
@test collect(K3) ≈ kron(X, C)

Kpow = ⊗(A, 5)
@test order(Kpow) == 5
@test size(Kpow, 1) == 4^5
@test Kpow[1,1] ≈ A[1,1]^5

for j in 1:12
    for i in 1:12
        @test kronprod[i,j] ≈ X[i,j]
    end
end

# test mixed-product property

A = rand(5, 4)
B = rand(2, 3)
C = rand(4, 6)
D = rand(3, 4)

K1 = (A ⊗ B)
K2 = (C ⊗ D)

@test collect(K1 * K2) ≈ collect(K1) * collect(K2)
