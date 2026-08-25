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
          # collisions are re-sampled, so the number of edges is exactly the
          # expected edge count of the probability matrix
          @test sum(G) == round(Int, sum(P))
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

    @testset "seeded sampling is deterministic" begin
          @test naivesample(MersenneTwister(42), P) == naivesample(MersenneTwister(42), P)
          @test fastsample(MersenneTwister(42), P) == fastsample(MersenneTwister(42), P)
          @test sampleindices(MersenneTwister(42), P, 50) == sampleindices(MersenneTwister(42), P, 50)
          @test sampleindices(MersenneTwister(42), P1, 50) == sampleindices(MersenneTwister(42), P1, 50)
    end

    @testset "sampleindices matches the kron distribution" begin
          rng = MersenneTwister(1)
          n = 200_000
          A = [0.5 0.1; 0.2 0.7]
          B = [0.3 0.6 0.1; 0.2 0.05 0.4]  # rectangular factor

          for (K, W) in ((A ⊗ B, kron(A, B)), (kronecker(A, 3), kron(A, A, A)))
                counts = zeros(size(K))
                for (i, j) in sampleindices(rng, K, n)
                      counts[i, j] += 1
                end
                @test all(abs.(counts ./ n .- W ./ sum(W)) .< 0.01)
          end
    end

    @testset "weighted sampling frequencies" begin
          rng = MersenneTwister(0)
          weights = [0.5, 0.3, 0.2]
          n = 100_000
          draws = Kronecker._sample_weighted(rng, 1:3, weights, n)
          freqs = [count(==(i), draws) / n for i in 1:3]
          @test all(abs.(freqs .- weights) .< 0.01)
          @test_throws ArgumentError Kronecker._sample_weighted(rng, 1:3, zeros(3), 5)
          @test_throws ArgumentError Kronecker._sample_weighted(rng, 1:3, [0.5, -0.1, 0.6], 5)
          @test_throws ArgumentError sampleindices(rng, [0.5 -0.1; 0.2 0.4] ⊗ [0.3 0.2; 0.1 0.4], 5)
    end

end
