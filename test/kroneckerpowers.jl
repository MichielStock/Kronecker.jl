@testset "Kronecker powers" begin
    A = [0.1 0.4; 0.6 2.1]
    B = [1 2 3 4; 5 6 7 8]

    K1 = kronecker(A, 3)
    K2 = kronecker(B, 3)

    K1dense = kron(A, A, A)
    K2dense = kron(B, B, B)

    @testset "Types and basic properties" begin

        @test K1 isa AbstractKroneckerProduct
        @test K1 isa KroneckerPower

        @test issquare(K1)
        @test !issquare(K2)

        @test order(K1) == 3

        @test getmatrices(K1)[1] === A
        @test getmatrices(K1)[2] isa KroneckerPower
        @test order(getmatrices(K1)[2]) == 2

        @test !issymmetric(K1)

        @test collect(K1) ≈ K1dense
    end

    @testset "Linear algebra" begin
        @test tr(K1) ≈ tr(K1dense)
        @test !isposdef(K1)
        @test transpose(K1) ≈ transpose(K1dense)
        @test conj(K1) ≈ conj(K1dense)
        @test K1' ≈ K1dense'
        @test inv(K1) ≈ inv(K1dense)

        # test on pos def functions
        As = A' * A
        @test det(⊗(As, 2)) ≈ det(kron(As, As))
        @test logdet(⊗(As, 2)) ≈ log(det(kron(As, As)))
    end

@testset "Mixed product" begin

    @test K1 * K2 ≈ K1dense * K2dense
end

end
