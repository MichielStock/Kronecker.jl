@testset "Indexed" begin
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

    @test_throws DimensionMismatch kronprod[p, rand(1:b, 11), r,t]
    @test_throws BoundsError kronprod[p, -q, r,t]
    @test_throws BoundsError kronprod[p, q.+b, r,t]
    @test_throws BoundsError kronprod[p,q,r,t.+d]

    ikp = kronprod[p,q,r,t]

    @test eltype(ikp) == Float64

    subsystem = kron(N, M)[a * (q .- 1) .+ p, b * (t .- 1) .+ r]
    @test subsystem ≈ collect(ikp)
    @test subsystem ≈ ikp

    @test (N, M) == getmatrices(ikp)

    # result shortcut
    #u = genvectrick(M, N, v, p, q, r, t)
    u = ikp * v
    # result naive
    unaive = kron(N, M)[a * (q .- 1) .+ p, b * (t .- 1) .+ r] * v
    @test all(u .≈ unaive)

    @test_throws DimensionMismatch ikp * rand(8)
end
