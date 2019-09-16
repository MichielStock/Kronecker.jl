@testset "Kronecker graphs" begin
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
          # exact sample does not always generate this, so don't test
          #@test G[9, 1]  # probability of 1
          @test !G[2, 5]  # probability of 0
    end

    @testset "helpers" begin
         @test isprob(P1)
         @test !isprob([2 .01; 0.2 0])
         @test isprob(P)
         @test isprob(P4order)

         indices = sampleindices(P1, 100)
         @test all(maximum(indices) .<= size(P1))
         for ind in indices
               @test P1[ind...] > 0
         end

         indices = sampleindices(P, 100)
         @test all(maximum(indices) .<= size(P))
         for ind in indices
               @test P[ind...] > 0
         end

         indices = sampleindices(P4order, 1000)
         @test all(maximum(indices) .<= size(P4order))
         for ind in indices
               @test P4order[ind...] > 0
         end
   end

end
