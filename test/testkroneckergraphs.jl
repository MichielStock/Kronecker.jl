@test "Kronecker graphs" begin
    P1 = [0.1 0.0 0.0;
          0.3 0.4 0.5;
          1.0 0.7 0.2]

    P2 = [0.1 0.1 0.0;
          0.3 0.4 0.6;
          1.0 0.2 0.2]

    P = P1 ⊗ P2

    P4order = kronecker(P1, 4)

    @testset "Naive sample" begin
          G = naivesample(P)
          @test G isa AbstractSparseMatrix
          @test size(G) == size(P)
          @test G[9, 1]  # probability of 1
          @test !G[2, 5]  # probability of 0
    end

    @testset "Fast sample" begin
          G = fastsample(P)
          @test G isa AbstractSparseMatrix
          @test size(G) == size(P)
          @test G[9, 1]  # probability of 1
          @test !G[2, 5]
    end

    @testset "helpers" begin
         @test isprob(P1)
         @test !isprob([2 .01; 0.2 0])
         @test isprob(P)
         @test isprob(P4order)

         I = sampleindices(P1, 100)
         @test all(maximum(I) .<= size(P1))
         for ind in I
               @test P1[I] > 0
         end

         I = sampleindices(P, 100)
         @test all(maximum(I) .<= size(P))
         for ind in I
               @test P[I] > 0
         end

         I = sampleindices(P4order, 1000)
         @test all(maximum(I) .<= size(P4order))
         for ind in I
               @test P4order[I] > 0
         end
   end

end
