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

"""
    mul!(x::AbstractVector, K::AbstractKroneckerProduct, v::AbstractVector)

Calculates the vector-matrix multiplication `K * v` and stores the result in
`x`, overwriting its existing value.
"""
function mul!(x::AbstractVector, K::AbstractKroneckerProduct, v::AbstractVector)
    M, N = getmatrices(K)
    a, b = size(M)
    c, d = size(N)
    e = length(v)
    f = length(x)
    f == a * c || throw(DimensionMismatch(
        "Dimension missmatch between kronecker system and result placeholder"))
    e == b * d || throw(DimensionMismatch(
        "Dimension missmatch between kronecker system and vector"))
    V = reshape(v, d, b)
    if (d + a) * b < (b + c) * d
        x .= vec(N * (V * transpose(M)))
    else
        x .= vec((N * V) * transpose(M))
    end
    return x
end

"""
    mul!(x::AbstractVector, K::AbstractKroneckerProduct, v::ReshapedArray)

Calculates the vector-matrix multiplication `K * v` and stores the result in
`x`, overwriting its existing value. Retains any special structure in the case
that `v` is a reshaped matrix.
"""
function mul!(x::AbstractVector, K::AbstractKroneckerProduct,
                                                    v::ReshapedArray)
    M, N = getmatrices(K)
    a, b = size(M)
    c, d = size(N)
    e = length(v)
    f = length(x)
    f == a * c || throw(DimensionMismatch(
        "Dimension missmatch between kronecker system and result placeholder"))
    e == b * d || throw(DimensionMismatch(
        "Dimension missmatch between kronecker system and vector"))
    if size(v.parent) == (d, b)
        V = v.parent
    else
        V = reshape(v, d, b)
    end
    if (d + a) * b < (b + c) * d
        x .= vec(N * (V * transpose(M)))
    else
        x .= vec((N * V) * transpose(M))
    end
    return x
end

function Base.:*(K::AbstractKroneckerProduct, v::AbstractVector)
    return mul!(Vector{promote_type(eltype(v), eltype(K))}(undef,
                                                        first(size(K))), K, v)
end

function Base.sum(K::AbstractKroneckerProduct; dims::Union{Nothing,Int}=nothing)
    A, B = getmatrices(K)
    if dims == nothing
        s = zero(eltype(K))
        sumB = sum(B)
        return sum(sum(B) * A)
    else
        return kronecker(sum(A, dims=dims), sum(B, dims=dims))
    end
end
