#=
Created on Saturday 3 August 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Vec trick: multiplying vectors with Kronecker systems.
=#


import Base.ReshapedArray

vecmulR!(X,N,V,M) = X .= N * (V * transpose(M))
vecmulL!(X,N,V,M) = X .= (N * V) * transpose(M)
vecmul!(X,N,V,M) = X .= N * V * transpose(M)

vectrick_reshape(v::AbstractVector, d::Int, b::Int) = reshape(v, d, b)
function vectrick_reshape(v::ReshapedArray{<:Any, 1}, d::Int, b::Int)
    return size(v.parent) == (d, b) ? v.parent : reshape(v, d, b)
end


"""
    mul_vec_trick!(x::AbstractVector, K::AbstractKroneckerProduct, v::AbstractVector)

Calculates the vector-matrix multiplication `K * v` and stores the result in
`x`, overwriting its existing value.
"""
function mul_vec_trick!(x::AbstractVector, M::AbstractMatrix, N::AbstractMatrix, v::AbstractVector)
    a, b = size(M)
    c, d = size(N)
    e = length(v)
    f = length(x)

    V = vectrick_reshape(v, d, b)
    X = vectrick_reshape(x, c, a)
    if b * c * (a + d) < a * d * (b + c)
        mul!(X, N, V * transpose(M))
    else
        mul!(X, N * V, transpose(M))
    end
    return x
end

function mul_vec_trick!(X::AbstractMatrix, M::AbstractMatrix, N::AbstractMatrix, V::AbstractMatrix)
    @inbounds for i in eachindex(axes(X, 2), axes(V, 2))
        @views mul_vec_trick!(X[:, i], M, N, V[:, i])
    end
    return X
end


# SOLVING
function ldiv_vec_trick!(x::AbstractVector, M::MatrixOrFactorization, N::MatrixOrFactorization, v::AbstractVector)
    b, a = size(M)
    d, c = size(N)
    e = length(v)
    f = length(x)

    V = vectrick_reshape(v, d, b)
    X = vectrick_reshape(x, c, a)
    if b * c * (a + d) < a * d * (b + c)
        S = N \ V
        copyto!(transpose(X), M \ transpose(S))
    else
        S = M \ transpose(V)
        copyto!(X, N \ transpose(S))
    end
    return x

    # size(K, 1) != length(v) && throw(DimensionMismatch("size(K, 1) != length(v)"))
    # C = reshape(v, size(K.B, 1), size(K.A, 1)) # matricify
    # return vec((K.B \ C) / K.A') #(A ⊗ B)vec(X) = vec(C) <=> BXA' = C => X = B^{-1} C A'^{-1}
end

function ldiv_vec_trick!(X::AbstractMatrix, M::MatrixOrFactorization, N::MatrixOrFactorization, V::AbstractMatrix)
    @inbounds for i in eachindex(axes(X, 2), axes(V, 2))
        @views ldiv_vec_trick!(X[:, i], M, N, V[:, i])
    end
    return X
end


function check_compatible_sizes(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, mul=true)
    m, n = mul ? size(A) : reverse(size(A))

    if n != size(B, 1)
        throw(DimensionMismatch(
            "A has size $(size(A)), B has size $(size(B))"
        ))
    elseif size(C) != (m, size(B)[2:end]...)
        throw(DimensionMismatch(
            "A has size $(size(A)), B has size $(size(B)), C has size $(size(C))"
        ))
    end
end

function mul!(C::AbstractMatrix, A::AbstractKroneckerProduct, D::Diagonal)
    check_compatible_sizes(C, A, D)
    @inbounds for j in axes(C, 2)
        @views C[:, j] = A[:, j] * D[j, j]
    end
    return C
end
function mul!(C::AbstractMatrix, D::Diagonal, A::AbstractKroneckerProduct)
    check_compatible_sizes(C, D, A)
    @inbounds for i in axes(C, 1)
        @views C[i, :] = D[i, i] * A[i, :]
    end
    return C
end

for VM in [:AbstractVector, :AbstractMatrix]
    @eval function mul!(C::$VM, A::AbstractKroneckerProduct, B::$VM)
        check_compatible_sizes(C, A, B)

        matrices = getallfactors(A)

        if length(matrices) == 2
            return mul_vec_trick!(C, matrices[1], matrices[2], B)
        elseif all(issquare, matrices)
            return kron_mul_fast_square!(C, B, matrices)
        else
            return kron_mul_fast_rect!(C, B, matrices)
        end
    end

    @eval function ldiv!(C::$VM, A::AbstractKroneckerProduct, B::$VM)
        check_compatible_sizes(C, A, B, false)

        matrices = getallfactors(A)

        factors = ntuple(length(matrices)) do i
            m = matrices[i]
            return (m isa Factorization) ? m : factorize(m)
        end

        if length(matrices) == 2
            return ldiv_vec_trick!(C, factors[1], factors[2], B)
        elseif all(issquare, matrices)
            return kron_ldiv_fast_square!(C, B, factors)
        else
            return kron_ldiv_fast_rect!(C, B, factors)
        end
    end
end


function Base.:*(K::AbstractKroneckerProduct, v::AbstractVector)
    return mul!(Vector{promote_type(eltype(v), eltype(K))}(undef, first(size(K))), K, v)
end

function Base.:*(K::AbstractKroneckerProduct, v::AbstractMatrix)
    return mul!(Matrix{promote_type(eltype(v), eltype(K))}(undef, first(size(K)), last(size(v))), K, v)
end

function Base.:*(K::AbstractKroneckerProduct, D::Diagonal)
    return mul!(Matrix{promote_type(eltype(K), eltype(D))}(undef, size(K)...), K, D)
end

function Base.:*(v::Adjoint{<:Number, <:AbstractVector}, K::AbstractKroneckerProduct)
    return mul!(Vector{promote_type(eltype(v), eltype(K))}(undef, first(size(K))), K', v.parent)'
end

function Base.:*(v::Transpose{<:Number, <:AbstractVector}, K::AbstractKroneckerProduct)
    return transpose(mul!(Vector{promote_type(eltype(v), eltype(K))}(undef, first(size(K))), transpose(K), v.parent))
end

function Base.:*(v::AbstractMatrix, K::AbstractKroneckerProduct)
    return transpose(mul!(Matrix{promote_type(eltype(v), eltype(K))}(undef, first(size(K)), last(size(v))), transpose(K), transpose(v)))
end

function LinearAlgebra.:\(K::AbstractKroneckerProduct, v::AbstractVector)
    return ldiv!(Vector{promote_type(eltype(v), eltype(K))}(undef, last(size(K))), K, v)
end

function LinearAlgebra.:\(K::AbstractKroneckerProduct, v::AbstractMatrix)
    return ldiv!(Matrix{promote_type(eltype(v), eltype(K))}(undef, last(size(K)), last(size(v))), K, v)
end

function Base.sum(K::AbstractKroneckerProduct; dims::Union{Nothing,Int}=nothing)
    A, B = getmatrices(K)
    if dims === nothing
        s = zero(eltype(K))
        sumB = sum(B)
        return sum(sumB * A)
    else
        return kronecker(sum(A, dims=dims), sum(B, dims=dims))
    end
end
