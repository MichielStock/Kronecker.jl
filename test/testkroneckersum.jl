@testset "Kronecker sums" begin

    As = (rand(4,4), sprand(4,4,1.0))
    Bs = (rand(3,3), sprand(3,3,1.0))
    Cs = (rand(5,5), sprand(5,5,1.0))
    Ds = (rand(ComplexF64,6,6), sprand(ComplexF64,6,6,1.0))
    arraytypes = (Matrix, SparseMatrixCSC)
    for (A, B, C, D, arraytype) in zip(As, Bs, Cs, Ds, arraytypes)
        IA = oneunit(A)
        IB = oneunit(B)

        KS = A ⊕ B
        kronsum = kron(A, IB) + kron(IA, B)

        @test eltype(KS) <: Float64
        @test KS isa AbstractMatrix{Float64}
        @test KS isa GeneralizedKroneckerProduct{Float64}
        @test !isa(KS, AbstractKroneckerProduct)

        @test collect(KS) ≈ kronsum
        @test collect(KS) isa AbstractSparseMatrix

        @test issquare(KS)

        IC = oneunit(C)
        KS3 = A ⊕ B ⊕ C
        KS3AB = (A ⊕ B) ⊕ C
        KS3BC = A ⊕ (B ⊕ C)

        kronsum3 = kron(A,IB,IC) + kron(IA,B,IC) + kron(IA,IB,C)

        for ks3 in [KS3, KS3AB, KS3BC]
            @test collect(ks3) ≈ kronsum3
            @test collect(ks3) isa AbstractSparseMatrix
            @test order(ks3) == 3
            @test getindex(ks3,2,3) == kronsum3[2,3]
        end

        @test order(KS) == 2
        @test getmatrices(KS) == (A,B)
        @test getindex(KS,2,3) == kronsum[2,3]
        
        ID = oneunit(D)

        @testset "Structure of sums" begin
            @test size(A ⊕ D) == size(A) .* size(D,1)
            @test size(B ⊕ C ⊕ D) == size(B) .* size(C) .* size(D)
            @test eltype(A ⊕ B) == Float64
            @test eltype(C ⊕ D) == ComplexF64
        end

        @testset "Basic linear algebra for sums" begin
            @test tr(KS) ≈ tr(kronsum)
            @test KS' == kronsum'
            @test transpose(KS) == transpose(kronsum)
            @test conj(KS) == conj(kronsum)
        end

        A = rand(10,10); B = rand(10,10); V = Diagonal(rand(10))
        @testset "Vec trick for sums" begin
            @test (A ⊕ B) * vec(V) == vec(B*V + V*transpose(A))
        end

        A = rand(3, 3); IA = oneunit(A)
        B = rand(4, 4); IB = oneunit(B)
        @testset "exp for Kronecker sum" begin
            EKS = exp(A ⊕ B)
            @test EKS isa AbstractKroneckerProduct
            @test EKS ≈ exp(kron(A, IB) + kron(IA, B))
        end

        @testset "sum over Kronecker sum" begin
            @test sum(KS) ≈ sum(kronsum)
            @test sum(KS3) ≈ sum(kronsum3)

            for dims in 1:2
                @test sum(KS, dims=dims) ≈ sum(kronsum, dims=dims)
                @test sum(KS3, dims=dims) ≈ sum(kronsum3, dims=dims)
            end

            @test_throws ArgumentError sum(KS, dims=-1)
            @test_throws ArgumentError sum(KS3, dims=3)
        end

        @testset "sparse Kronecker sum" begin
            for ks in (KS, KS3, KS3AB, KS3BC)
                sks = sparse(ks)
                @test sks isa AbstractKroneckerSum
                @test issparse(sks.A)
                @test issparse(sks.B)

                if arraytype <: Array
                    @test !issparse(ks)
                elseif arraytype <: AbstractSparseMatrix
                    @test issparse(ks)
                end
            end
        end

        @testset "call kron on Kronecker sums" begin
            for ks in (KS, KS3, KS3AB, KS3BC)
                cks = collect(ks)
                @test kron(ks,D) ≈ kron(cks, D)
                @test kron(D,ks) ≈ kron(D,cks)
                @test kron(ks,ks) ≈ kron(cks, cks)
            end
        end
    end

end
