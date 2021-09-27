@testset "Kronecker powers" begin
    A = [0.1 0.4; 0.6 2.1]
    B = [1 2 3 4; 5 6 7 8]

    K1 = kronecker(A, 3)
    K2 = kronecker(B, 3)

    K1dense = kron(A, A, A)
    K2dense = kron(B, B, B)

    C = [1 2; 3 4]
    KC = kronecker(C, 3)
    KCdense = kron(C, C, C)

    @testset "Types and basic properties" begin

        @test K1 isa AbstractKroneckerProduct
        @test K1 isa KroneckerPower

        @test issquare(K1)
        @test !issquare(K2)

        @test eltype(K1) <: Real
        @test eltype(K2) <: Integer

        @test order(K1) == 3

        @test getmatrices(K1)[1] isa KroneckerPower
        @test getmatrices(K1)[2] === A
        @test order(getmatrices(K1)[1]) == 2

        # scaling
        @test 3K1 ≈ 3K1dense ≈ K1 * 3
        @test 9.2 * K2 ≈ 9.2K2dense ≈ K2 * 9.2

        @test !issymmetric(K1)

        @test collect(K1) ≈ K1dense
        @test collect!(similar(K2dense), K2) ≈ K2dense

        @test sum(K1) ≈ sum(K1dense)

    end

    @testset "Inplace scaling" begin
        using LinearAlgebra: lmul!
        K1copy = kronecker(copy(A), 3)
        K2copy = kronecker(copy(B), 3)

        lmul!(2.7, K1copy)
        @test_throws InexactError lmul!(2.3, K2copy)

        @test K1copy ≈ 2.7K1dense

    end

    @testset "Linear algebra" begin
        @test tr(K1) ≈ tr(K1dense)
        @test !isposdef(K1)
        @test transpose(K1) ≈ transpose(K1dense)
        @test conj(K1) ≈ conj(K1dense)
        @test K1' ≈ K1dense'
        @test inv(K1) ≈ inv(K1dense)

        @test diag(K1) ≈ diag(K1dense)
        @test diag(K2) == diag(K2dense)
        @test diag(KC) == diag(KCdense)


        # test on pos def functions
        As = A' * A
        @test det(⊗(As, 2)) ≈ det(kron(As, As))
        @test logdet(⊗(As, 2)) ≈ log(det(kron(As, As)))
    end


    @testset "Inference in indexing" begin
        @test (@inferred K2[1,1]) == K2dense[1,1]
    end

    @testset "Arithmetic" begin
        @test K2 + K2 == 2 * K2 == 2 * K2dense
        @test K2 + K2 + K2 == 3 * K2 == 3 * K2dense
        @test K2 - K2 == 0 * K2 == 0 * K2dense
        @test K2 - K2 - K2 == -K2 == -K2dense
    end

    @testset "Broadcasting" begin
        @test K2 .+ K2 == 2 .* K2 == 2 .* K2dense
        @test K2 .- K2 == 0 .* K2 == 0 .* K2dense
        @test sin.(K2) == sin.(K2dense)
    end

    @testset "Multiplication" begin
        # product between two Kronecker powers
        @test K1 * K2 ≈ K1dense * K2dense
        # power
        @test K1^4 ≈ K1dense^4
        # mixed product
        @test K1 * (A ⊗ B ⊗ A) ≈ (A ⊗ A ⊗ A) * (A ⊗ B ⊗ A)
        @test (A ⊗ B' ⊗ A) * K1 ≈ (A ⊗ B' ⊗ A) * (A ⊗ A ⊗ A)
    end


@testset "Mixed product" begin

    @test K1 * K2 ≈ K1dense * K2dense
end

end
