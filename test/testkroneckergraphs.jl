@test "Kronecker graphs" begin
    P1 = [0.1 0.0 0.0;
          0.3 0.4 0.5;
          1.0 0.7 0.2]

    P2 = [0.1 0.1 0.0;
          0.3 0.4 0.6;
          1.0 0.2 0.2]

    P = P1 ⊗ P2

    @test isprob(P1)
    @test !isprob([2 .01; 0.2 0])
    @test isprob(P)

    G = naivesample(P)
    @test G isa AbstractSparseMatrix
    @test size(G) == size(P)
    @test G[9, 1]  # probability of 1
    @test !G[2, 5]  # probability of 0

end
