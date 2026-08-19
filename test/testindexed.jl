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

    # regression tests for genvectrick!: the scratch array was not
    # zero-initialised (garbage/NaN results depending on heap state) and the
    # second branch used wrong dimensions and indices; run both branches on
    # many seeded datasets and compare against the naive computation
    let rng = MersenneTwister(42)
        for trial in 1:100
            # sizes with a*e + d*f < c*e + b*f: first branch (T = VM')
            M2 = randn(rng, 4, 8); N2 = randn(rng, 5, 9)
            p2, q2 = rand(rng, 1:4, 6), rand(rng, 1:5, 6)
            r2, t2 = rand(rng, 1:8, 10), rand(rng, 1:9, 10)
            v2 = randn(rng, 10)
            ikp2 = (N2 ⊗ M2)[p2, q2, r2, t2]
            @test ikp2 * v2 ≈ collect(ikp2) * v2

            # sizes with a*e + d*f >= c*e + b*f: second branch (S = NV)
            M3 = randn(rng, 10, 2); N3 = randn(rng, 2, 3)
            p3, q3 = rand(rng, 1:10, 4), rand(rng, 1:2, 4)
            r3, t3 = rand(rng, 1:2, 5), rand(rng, 1:3, 5)
            v3 = randn(rng, 5)
            ikp3 = (N3 ⊗ M3)[p3, q3, r3, t3]
            @test ikp3 * v3 ≈ collect(ikp3) * v3
        end
    end

    @test_throws DimensionMismatch ikp * rand(8)
end
