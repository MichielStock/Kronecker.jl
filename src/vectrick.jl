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
    V = vectrick_reshape(v, d, b)
    if (d + a) * b < (b + c) * d
        x .= vec(N * (V * transpose(M)))
    else
        x .= vec((N * V) * transpose(M))
    end
    return x
end

function Base.:*(K::AbstractKroneckerProduct, v::AbstractVector)
    return mul!(Vector{promote_type(eltype(v), eltype(K))}(undef, first(size(K))), K, v)
end

reshape_cols(x, sizes...) = reshape(x, sizes...)
reshape_rows(x, sizes...) = transpose(reshape(collect(transpose(x)), reverse(sizes)...))

kron_id_a(a, x) = reshape_cols(a * reshape_cols(x, size(a)[2], :), :, size(x)[2])
kron_a_id(a, x) = reshape_rows(a * reshape_rows(x, size(a)[2], :), :, size(x)[2])
function kron_a_b(A, B, x)
    a, b = size(A)
    c, d = size(B)
    if (d + a) * b < (b + c) * d
        return kron_a_id(A, kron_id_a(B, x))
    else
        return kron_id_a(B, kron_a_id(A, x))
    end
end

function Base.:*(A::AbstractKroneckerProduct, B::StridedMatrix)
    size(A, 2) != size(B, 1) && throw(DimensionMismatch("size(A, 2) != size(B, 1)"))
    return kron_a_b(getmatrices(A)..., B)
end

function Base.:*(A::KroneckerProduct{<:Any, <:Eye, <:AbstractMatrix}, B::StridedMatrix)
    size(A, 2) != size(B, 1) && throw(DimensionMismatch("size(A, 2) != size(B, 1)"))
    return kron_id_a(A.B, B)
end

function Base.:*(A::KroneckerProduct{<:Any, <:AbstractMatrix, <:Eye}, B::StridedMatrix)
    size(A, 2) != size(B, 1) && throw(DimensionMismatch("size(A, 2) != size(B, 1)"))
    return kron_a_id(A.A, B)
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
