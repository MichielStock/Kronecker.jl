using Kronecker
using Test
using LinearAlgebra


A = randn(4, 4)
B = Array{Float64, 2}([1 2 3;
     4 5 6;
     7 -2 9])
C = rand(5, 6)

v = rand(12)

@test issquare(A)
@test !issquare(C)

kronprod = A ⊗ B

X = kron(A, B)  # true result

@test tr(kronprod) ≈ tr(X)
@test det(kronprod) ≈ det(X)
@test collect(inv(kronprod)) ≈ inv(X)
@test all(kronprod * v .≈ X * v)

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


# testing indexed systems

v = rand(10)

a, b = 4, 8
c, d = 5, 9

M = randn(a, b)
N = rand(c, d)

p = rand(1:a, 6)
q = rand(1:c, 6)

r = rand(1:b, 10)
t = rand(1:d, 10)

kronprod = N ⊗ M
ikp = kronprod[p,q,r,t]

subsystem = kron(N, M)[a * (q .- 1) .+ p, b * (t .- 1) .+ r]
@test subsystem ≈ collect(ikp)

# result shortcut
#u = genvectrick(M, N, v, p, q, r, t)
u = ikp * v
# result naive
unaive = kron(N, M)[a * (q .- 1) .+ p, b * (t .- 1) .+ r] * v
@test all(u .≈ unaive)


# Shifted Kronecker systems

A = Symmetric(rand(10, 10))
B = Symmetric(rand(30, 30))
v = rand(300)

rnaive = (kron(A, B) + 2I) \ v

@test rnaive ≈ (A ⊗ B + 2I) \ v
@test rnaive ≈ v / (A ⊗ B + 2I)
