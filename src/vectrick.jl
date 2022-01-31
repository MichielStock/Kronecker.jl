#=
Created on Saturday 3 August 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Vec trick: multiplying vectors with Kronecker systems.
=#


import Base.ReshapedArray

vectrick_reshape(v::AbstractVector, d::Int, b::Int) = reshape(v, d, b)
function vectrick_reshape(v::ReshapedArray{<:Any,1}, d::Int, b::Int)
    return size(v.parent) == (d, b) ? v.parent : reshape(v, d, b)
end


"""
    mul_vec_trick!(x::AbstractVector, K::AbstractKroneckerProduct, v::AbstractVector)

Calculates the vector-matrix multiplication `K * v` and stores the result in
`x`, overwriting its existing value.
"""
function mul_vec_trick!(x::AbstractVector, A::AbstractKroneckerProduct, v::AbstractVector)
    M, N = getmatrices(A)
    a, b = size(M)
    c, d = size(N)

    V = vectrick_reshape(v, d, b)
    X = vectrick_reshape(x, c, a)
    if b * c * (a + d) < a * d * (b + c)
        mul!(X, N, V * transpose(M))
    else
        mul!(X, N * V, transpose(M))
    end
    return x
end

function mul_vec_trick!(x::AbstractVector, K::AbstractKroneckerSum, v::AbstractVector)
    A, B = getmatrices(K)
    a, b = size(A)
    c, d = size(B)

    V = reshape(v, d, b)
    X = reshape(x, c, a)
    mul!(X, V, transpose(A))
    _mul5!(X, B, V, true, true)
    return x
end

if VERSION < v"1.3.0-alpha.115"
    function _mul5!(X, B, V, α, β)
        if β && α
            X .= (B * V) .* α .+ X .* β
        elseif α
            X .= (B * V) .* α
        elseif β
            X .*= β
        else
            X .= zero(eltype(X))
        end
    end
else # 5-arg mul! is available
    _mul5! = mul!
end # VERSION

function mul_vec_trick!(X::AbstractMatrix, A::GeneralizedKroneckerProduct, V::AbstractMatrix)
    @inbounds for i in eachindex(axes(X, 2), axes(V, 2))
        @views mul_vec_trick!(X[:, i], A, V[:, i])
    end
    return X
end


# SOLVING
function ldiv_vec_trick!(x::AbstractVector, A::AbstractKroneckerProduct, v::AbstractVector)
    M, N = getmatrices(A)
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

function ldiv_vec_trick!(X::AbstractMatrix, A::AbstractKroneckerProduct, V::AbstractMatrix)
    @inbounds for i in eachindex(axes(X, 2), axes(V, 2))
        @views ldiv_vec_trick!(X[:, i], A, V[:, i])
    end
    return X
end


function check_compatible_sizes(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, mul = true)
    # when performing a division (mul=false), A acts as a matrix with reversed dimensions
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

for TC in [:AbstractVector, :AbstractMatrix],
    TB in [:($TC), :(Transpose{T,<:$TC{T}} where {T}), :(Adjoint{T,<:$TC{T}} where {T})]

    @eval function mul!(C::$TC, A::AbstractKroneckerProduct, B::$TB)
        check_compatible_sizes(C, A, B)

        factors = getallfactors(A)

        if length(factors) == 2
            return mul_vec_trick!(C, A, B)
        elseif all(issquare, factors)
            return _kron_mul_fast_square!(C, B, factors)
        else
            return _kron_mul_fast_rect!(C, B, factors)
        end
    end

    @eval function ldiv!(C::$TC, A::AbstractKroneckerProduct, B::$TB)
        check_compatible_sizes(C, A, B, false)

        matrices = getallfactors(A)

        # TODO: should figure out how to avoid this if possible, leave
        #       factorization up to the user/Julia
        factors = ntuple(length(matrices)) do i
            m = matrices[i]
            return (m isa Factorization) ? m : factorize(m)
        end

        if length(factors) == 2
            return ldiv_vec_trick!(C, A, B)
        elseif all(issquare, factors)
            return _kron_ldiv_fast_square!(C, B, factors)
        else
            return _kron_ldiv_fast_rect!(C, B, factors)
        end
    end

    @eval function mul!(C::$TC, A::AbstractKroneckerSum, B::$TB)
        check_compatible_sizes(C, A, B)

        summands = getallsummands(A)

        if length(summands) == 2
            return mul_vec_trick!(C, A, B)
        else
            return _kronsum_mul_fast!(C, B, summands)
        end
    end
end

function Base.:*(K::GeneralizedKroneckerProduct, v::AbstractVector)
    return mul!(Vector{promote_type(eltype(v), eltype(K))}(undef, first(size(K))), K, v)
end

# explicit list of types that need to be disambiguated against
const MulMatTypes = [:Diagonal, :AbstractTriangular]

for T in [MulMatTypes; :AbstractMatrix]
    @eval function Base.:*(K::GeneralizedKroneckerProduct, M::$T)
        return mul!(Matrix{promote_type(eltype(M), eltype(K))}(undef, size(K, 1), size(M, 2)), K, M)
    end
end
for T in MulMatTypes
    @eval function Base.:*(M::$T, K::GeneralizedKroneckerProduct)
        return mul!(Matrix{promote_type(eltype(K), eltype(M))}(undef, size(M, 1), size(K, 2)), M, K)
    end
end

function Base.:*(v::AbstractMatrix, K::GeneralizedKroneckerProduct)
    out = Matrix{promote_type(eltype(v), eltype(K))}(undef, last(size(K)), first(size(v)))
    return transpose(mul!(out, transpose(K), collect(transpose(v))))
end

function Base.:*(v::Adjoint{<:Number,<:AbstractVector}, K::GeneralizedKroneckerProduct)
    out = Vector{promote_type(eltype(v), eltype(K))}(undef, last(size(K)))
    return mul!(out, K', v.parent)'
end

function Base.:*(v::Transpose{<:Number,<:AbstractVector}, K::GeneralizedKroneckerProduct)
    out = Vector{promote_type(eltype(v), eltype(K))}(undef, last(size(K)))
    return transpose(mul!(out, transpose(K), v.parent))
end

# special multiplication methods for Kronecker products of Diagonal matrices
# It's usually better to convert these to Diagonal to use optimized multiplication methods
# instead of using the vec trick
const KroneckerDiagonal = Union{KronProdDiagonal,KronPowDiagonal}
Base.:*(K::KroneckerDiagonal, v::AbstractVector) = Diagonal(K) * v
for T in [MulMatTypes; :AbstractMatrix; :AbstractKroneckerProduct]
    @eval Base.:*(K::KroneckerDiagonal, D::$T) = Diagonal(K) * D
    @eval Base.:*(D::$T, K::KroneckerDiagonal) = D * Diagonal(K)
end
# ambiguity fix
Base.:*(K1::KroneckerDiagonal, K2::KroneckerDiagonal) = Diagonal(K1) * Diagonal(K2)

for T in [:Adjoint, :Transpose]
    @eval Base.:*(A::$T{<:Number,<:AbstractVector}, K::KroneckerDiagonal) = A * Diagonal(K)
end

function LinearAlgebra.:\(K::AbstractKroneckerProduct, v::AbstractVector)
    return ldiv!(Vector{promote_type(eltype(v), eltype(K))}(undef, last(size(K))), K, v)
end

function LinearAlgebra.:\(K::AbstractKroneckerProduct, v::AbstractMatrix)
    return ldiv!(Matrix{promote_type(eltype(v), eltype(K))}(undef, last(size(K)), last(size(v))), K, v)
end

function Base.sum(K::AbstractKroneckerProduct; dims::Union{Nothing,Int} = nothing)
    A, B = getmatrices(K)
    if dims === nothing
        s = zero(eltype(K))
        sumB = sum(B)
        return sum(sumB * A)
    else
        return kronecker(sum(A, dims = dims), sum(B, dims = dims))
    end
end
