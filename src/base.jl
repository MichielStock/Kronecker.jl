abstract type GeneralizedKroneckerProduct{T} <: AbstractMatrix{T} end

Base.eltype(K::GeneralizedKroneckerProduct{T}) where {T} = T

abstract type AbstractKroneckerProduct{T} <: GeneralizedKroneckerProduct{T} end

Base.IndexStyle(::Type{<:GeneralizedKroneckerProduct}) = IndexCartesian()

"""
    collect!(C::AbstractMatrix, K::GeneralizedKroneckerProduct)

In-place collection of `K` in `C`. If possible, specialized routines are used to
speed up the computation. The fallback is an element-wise iteration. In this case,
this function might be slow.
"""
function collect!(C::AbstractMatrix, K::GeneralizedKroneckerProduct)
    size(C) == size(K) || throw(DimensionMismatch("`K` $(size(K)) cannot be collected in `C` $(size(C))"))
    @inbounds for I in CartesianIndices(K)
        C[I] = K[I]
    end
    return C
end


"""
    collect(K::GeneralizedKroneckerProduct)

Collects a lazy instance of the `GeneralizedKroneckerProduct` type into a dense,
native matrix. Falls back to the element-wise case when not specialized method
is defined.
"""
function collect(K::GeneralizedKroneckerProduct{T}) where {T}
    C = Matrix{T}(undef, size(K)...)
    collect!(C, K)
    return C
end


"""
    KroneckerProduct{T,TA<:AbstractMatrix, TB<:AbstractMatrix} <: AbstractKroneckerProduct{T}

Concrete Kronecker product between two matrices `A` and `B`.
"""
struct KroneckerProduct{T<:Any, TA<:AbstractMatrix, TB<:AbstractMatrix} <: AbstractKroneckerProduct{T}
    A::TA
    B::TB
    function KroneckerProduct(A::AbstractMatrix{T}, B::AbstractMatrix{V}) where {T, V}
        return new{promote_type(T, V), typeof(A), typeof(B)}(A, B)
    end
end

# conversion
Base.convert(::Type{T}, K::KroneckerProduct) where {T<:KroneckerProduct} = K isa T ? K : T(K)
function KroneckerProduct{T,TA,TB}(K::KroneckerProduct) where {T,TA,TB}
    A, B = getmatrices(K)
    KroneckerProduct(convert(TA, A), convert(TB, B))
end

"""
    kronecker(A::AbstractMatrix, B::AbstractMatrix)

Construct a Kronecker product object between two arrays. Does not evaluate the
Kronecker product explictly.
"""
kronecker(A::AbstractMatrix, B::AbstractMatrix) = KroneckerProduct(A, B)

# version that have a vector as input reshape to matrices
kronecker(A::AbstractVector, B::AbstractMatrix) = KroneckerProduct(reshape(A,:,1), B)
kronecker(A::AbstractMatrix, B::AbstractVector) = KroneckerProduct(A, reshape(B,:,1))
kronecker(A::AbstractVector, B::AbstractVector) = KroneckerProduct(reshape(A,:,1), reshape(B,:,1))

"""
    kronecker(A::AbstractMatrix, B::AbstractMatrix)

Higher-order Kronecker lazy kronecker product, e.g.
```
kronecker(A, B, C, D)
```
"""
kronecker(A::AbstractVecOrMat, B::AbstractVecOrMat...) = kronecker(A,
                                                            kronecker(B...))

"""
    ⊗(A::AbstractMatrix, B::AbstractMatrix)

Binary operator for `kronecker`, computes as Lazy Kronecker product. See
`kronecker` for documentation.
"""
⊗(A::AbstractVecOrMat, B::AbstractVecOrMat) = kronecker(A, B)
⊗(A::AbstractVecOrMat...) = kronecker(A...)

"""
    getindex(K::AbstractKroneckerProduct, i1::Integer, i2::Integer)

Computes and returns the (i,j)-th element of an `AbstractKroneckerProduct` K.
Uses recursion if `K` is of an order greater than two.
"""
function getindex(K::AbstractKroneckerProduct, i1::Integer, i2::Integer)
    A, B = getmatrices(K)
    k, l = size(B)
    return (A[cld(i1, k), cld(i2, l)]::eltype(A)) * (B[(i1 - 1) % k + 1, (i2 - 1) % l + 1]::eltype(B))
end


"""
    getallfactors(K::AbstractKroneckerProduct)

Obtain all factors in an `AbstractKroneckerProduct` object.
"""
getallfactors(K::AbstractKroneckerProduct) = (getallfactors(K.A)..., getallfactors(K.B)...)
getallfactors(K::AbstractMatrix) = (K,)


"""
    getmatrices(K::AbstractKroneckerProduct)

Obtain the two matrices of an `AbstractKroneckerProduct` object.
"""
getmatrices(K::AbstractKroneckerProduct) = (K.A, K.B)

"""
    getmatrices(A::AbstractArray)

Returns a matrix itself. Needed for recursion.
"""
getmatrices(A::AbstractArray) = (A,)

firstmatrix(K::AbstractKroneckerProduct) = first(getmatrices(K))
lastmatrix(K::AbstractKroneckerProduct) = last(getmatrices(K))

"""
    size(K::AbstractKroneckerProduct)

Returns a the size of an `AbstractKroneckerProduct` instance.
"""
function size(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    (m, n) = size(A)
    (k, l) = size(B)
    return m * k, n * l
end

"""
    size(K::GeneralizedKroneckerProduct)

Returns a the size of an `GeneralizedKroneckerProduct` instance.
"""
size(K::GeneralizedKroneckerProduct, dim::Integer) = size(K)[dim]

# CHECKS

"""
    issquare(A::AbstractMatrix)

Checks if an array is a square matrix.
"""
function issquare(A::AbstractMatrix)
    m, n = size(A)
    return m == n
end


"""
    issquare(A::Factorization)

Checks if a Factorization struct represents a square matrix.
"""
function issquare(A::Factorization)
    m, n = size(A)
    return m == n
end

"""
    issymmetric(K::AbstractKroneckerProduct)

Checks if a Kronecker product is symmetric.
"""
function LinearAlgebra.issymmetric(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return issymmetric(A) && issymmetric(B)
end


"""
    ishermitian(K::AbstractKroneckerProduct)

Checks if a Kronecker product is Hermitian.
"""
function LinearAlgebra.ishermitian(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return ishermitian(A) && ishermitian(B)
end

"""
    isposdef(K::AbstractKroneckerProduct)

Test whether a Kronecker product is positive definite (and Hermitian) by trying
to perform a Cholesky factorization of K.
"""
function LinearAlgebra.isposdef(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return isposdef(A) && isposdef(B)
end

# SCALAR PROPERTIES

"""
    order(M::AbstractMatrix)

Returns the order of a matrix, i.e. how many matrices are involved in the
Kronecker product (default to 1 for general matrices).
"""
order(M::AbstractMatrix) = 1
order(M::AbstractKroneckerProduct) = order(M.A) + order(M.B)

"""
    det(K::AbstractKroneckerProduct)

Compute the determinant of a Kronecker product.
"""
function LinearAlgebra.det(K::AbstractKroneckerProduct{T}) where {T}
    checksquare(K)
    A, B = getmatrices(K)
    if issquare(A) && issquare(B)
        m = size(A)[1]
        n = size(B)[1]
        return det(A)^n * det(B)^m
    else
        return zero(T)
    end
end

"""
    logdet(K::AbstractKroneckerProduct)

Compute the logarithm of the determinant of a Kronecker product.
"""
function LinearAlgebra.logdet(K::AbstractKroneckerProduct{T}) where {T}
    checksquare(K)
    A, B = getmatrices(K)
    if issquare(A) && issquare(B)
        m = size(A)[1]
        n = size(B)[1]
        return n * logdet(A) + m * logdet(B)
    else
        return real(T)(-Inf)
    end
end

"""
    tr(K::AbstractKroneckerProduct)

Compute the trace of a Kronecker product.
"""
function LinearAlgebra.tr(K::AbstractKroneckerProduct)
    checksquare(K)
    A, B = getmatrices(K)
    if issquare(A) && issquare(B)
        return tr(A) * tr(B)
    else
        return sum(diag(K))
    end
end

"""
    inv(K::AbstractKroneckerProduct)

Compute the inverse of a Kronecker product.
"""
function inv(K::AbstractKroneckerProduct)
    checksquare(K)
    A, B = getmatrices(K)
    if issquare(A) && issquare(B)
        return KroneckerProduct(inv(A), inv(B))
    else
        throw(SingularException(1))
    end
end

"""
    pinv(K::AbstractKroneckerProduct)

Compute the Moore-Penrose pseudo-inverse of a Kronecker product.
"""
function LinearAlgebra.pinv(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return KroneckerProduct(pinv(A), pinv(B))
end

"""
    adjoint(K::AbstractKroneckerProduct)

Compute the adjoint of a Kronecker product.
"""
function adjoint(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kronecker(A', B')
end

"""
    transpose(K::AbstractKroneckerProduct)

Compute the transpose of a Kronecker product.
"""
function Base.transpose(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kronecker(transpose(A), transpose(B))
end

function Base.conj(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kronecker(conj(A), conj(B))
end

function LinearAlgebra.diag(K::KroneckerProduct)
    A, B = getmatrices(K)
    if issquare(A) && issquare(B)
        return kron(diag(K.A), diag(K.B))
    end
    return K[diagind(K)]
end

# COLLECTING

#=
"""
    collect(K::AbstractKroneckerProduct)

Collects a lazy instance of the `AbstractKroneckerProduct` type into a full,
native matrix. Equivalent with `Matrix(K::AbstractKroneckerProduct)`.
"""
function collect(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kron(A, B)
end
=#

_maybecollect(A::GeneralizedKroneckerProduct) = collect(A)
_maybecollect(A::AbstractArray) = A

# function for in-place Kronecker product
function _kron!(C::AbstractArray, A::AbstractArray, B::AbstractArray)
    m = first(LinearIndices(C)) - 1
    @inbounds for j = axes(A,2), l = axes(B,2), i = axes(A,1)
        Aij = A[i,j]
        for k = axes(B,1)
            C[m += 1] = Aij * B[k,l]
        end
    end
    return C
end

_kron!(C::AbstractArray, A::GeneralizedKroneckerProduct, B::AbstractArray) = _kron!(C, collect(A), B)
_kron!(C::AbstractArray, A::AbstractArray, B::GeneralizedKroneckerProduct) = _kron!(C, A, collect(B))
_kron!(C::AbstractArray, A::GeneralizedKroneckerProduct, B::GeneralizedKroneckerProduct) = _kron!(C, collect(A), collect(B))

@inline function _kron!(C::AbstractArray, As::Tuple{AbstractArray, Vararg{AbstractArray}}, Bs::Tuple{AbstractArray, Vararg{AbstractArray}}, f = identity)
    m = first(LinearIndices(C)) - 1
    A1 = first(As)
    B1 = first(Bs)
    for j = axes(A1,2), l = axes(B1,2), i = axes(A1,1)
        Aijs = map(A -> A[i,j], As)
        for k = 1:size(Bs[1],1)
            Bkls = map(B -> B[k,l], Bs)
            Aijs_times_Bkls = map(*, Aijs, Bkls)
            C[m += 1] = f(Aijs_times_Bkls...)
        end
    end
    return C
end

"""
    collect!(C::AbstractMatrix, K::AbstractKroneckerProduct)

In-place collection of `K` in `C` where `K` is an `AbstractKroneckerProduct`, i.e.,
`K = A ⊗ B`. This is equivalent to the broadcasted assignment `C .= K`.

    collect!(f, C::AbstractMatrix, K1::AbstractKroneckerProduct, Ks::AbstractKroneckerProduct...)

Evaluate `f.(K1, Ks...)` and assign it in-place to `C`. This is equivalent to the broadcasted
operation `C .= f.(K1, Ks...)`.
"""
function collect!(C::AbstractMatrix, K::AbstractKroneckerProduct)
    size(C) == size(K) || throw(DimensionMismatch("`K` $(size(K)) cannot be collected in `C` $(size(C))"))
    A, B = getmatrices(K)
    return _kron!(C, A, B)
end

@inline function collect!(f, C::AbstractMatrix, K1::AbstractKroneckerProduct, Ks::AbstractKroneckerProduct...)
    @noinline throwdm(K1sz, Csz) = throw(DimensionMismatch("`K` $K1sz cannot be collected in `C` $Csz"))
    size(C) == size(K1) || throwdm(size(K1), size(C))
    for K in Ks
        size(C) == size(K) || throwdm(size(K), size(C))
    end
    Ks_all = (K1, Ks...)
    As = map(x -> first(getmatrices(x)), Ks_all)
    Bs = map(x -> last(getmatrices(x)), Ks_all)
    return _kron!(C, As, Bs, f)
end


"""
    Matrix(K::GeneralizedKroneckerProduct)

Converts a `GeneralizedKroneckerProduct` instance to a Matrix type.
"""
Base.Matrix(K::GeneralizedKroneckerProduct) = collect(K)

function Base.:+(A::AbstractKroneckerProduct, B::StridedMatrix)
    T = promote_type(eltype(A), eltype(B))
    C = similar(Array{T}, size(A))
    C .= A
    C .+= B
    return C
end
Base.:+(A::StridedMatrix, B::AbstractKroneckerProduct) = B + A

function Base.:+(A::AbstractKroneckerProduct, B1::AbstractKroneckerProduct, Brest::AbstractKroneckerProduct...)
    Bs = (B1, Brest...)
    for B in Bs
        Base.promote_shape(A, B) # check size compatibility
    end
    # special methods to handle kronecker products with singleton matrices
    # if one matrix is common to all products, we only need to add the other matrix
    if all(x -> firstmatrix(x) === firstmatrix(A), Bs)
        K1 = kronecker(firstmatrix(A), +(lastmatrix(A), map(lastmatrix, Bs)...))
        return collect(K1)
    elseif all(x -> lastmatrix(x) === lastmatrix(A), Bs)
        K2 = kronecker(+(firstmatrix(A), map(firstmatrix, Bs)...), lastmatrix(A))
        return collect(K2)
    end
    # if the sizes of the component matrices are compatible, the operation may be
    # short-circuited
    sa = map(size, getmatrices(A))
    if all(x -> map(size, getmatrices(x)) == sa, Bs)
        return broadcast(+, A, Bs...)
    end
    # collect the arrrays before adding to avoid indexing into the kronecker products
    return +(collect(A), map(collect, Bs)...)
end

function Base.:-(A::AbstractKroneckerProduct, B::StridedMatrix)
    T = promote_type(eltype(A), eltype(B))
    C = similar(Array{T}, size(A))
    C .= A
    C .-= B
    return C
end
function Base.:-(A::StridedMatrix, B::AbstractKroneckerProduct)
    T = promote_type(eltype(A), eltype(B))
    C = similar(Array{T}, size(A))
    @. C = -B
    C .+= A
    return C
end

const KronProdDiagonal = KroneckerProduct{<:Any, <:Diagonal, <:Diagonal}
for T in [:Diagonal, :UniformScaling]
    @eval Base.:+(K::KronProdDiagonal, D::$T) = Diagonal(K) + D
    @eval Base.:+(D::$T, K::KronProdDiagonal) = D + Diagonal(K)
    @eval Base.:-(K::KronProdDiagonal, D::$T) = Diagonal(K) - D
    @eval Base.:-(D::$T, K::KronProdDiagonal) = D - Diagonal(K)
end

function Base.:-(A::AbstractKroneckerProduct, B::AbstractKroneckerProduct)
    # special methods to handle kronecker products with singleton matrices
    # if one matrix is common to all products, we only need to add the other matrix
    if firstmatrix(B) === firstmatrix(A)
        K1 = kronecker(firstmatrix(A), lastmatrix(A) - lastmatrix(B))
        return collect(K1)
    elseif lastmatrix(B) === lastmatrix(A)
        K2 = kronecker(firstmatrix(A) - firstmatrix(B), lastmatrix(A))
        return collect(K2)
    end
    # if the sizes of the component matrices are compatible, the operation may be
    # short-circuited
    if map(size, getmatrices(A)) == map(size, getmatrices(B))
        return A .- B
    end
    # collect the arrrays before subtracting to avoid indexing into the kronecker products
    return collect(A) - collect(B)
end

function Base.kron(K::AbstractKroneckerProduct, C::AbstractMatrix)
    A, B = getmatrices(K)
    return kron(kron(A, B), C)
end

function Base.kron(A::AbstractMatrix, K::AbstractKroneckerProduct)
    B, C = getmatrices(K)
    return kron(A, kron(B, C))
end

Base.kron(K1::AbstractKroneckerProduct,
            K2::AbstractKroneckerProduct) = kron(collect(K1), collect(K2))

# mixed-product property
function Base.:*(K1::AbstractKroneckerProduct, K2::AbstractKroneckerProduct)
    A, B = getmatrices(K1)
    C, D = getmatrices(K2)
    # check for size
    if size(A, 2) != size(C, 1)
        throw(DimensionMismatch("Mismatch between A and C in (A ⊗ B)(C ⊗ D)"))
    end
    if size(B, 2) != size(D, 1)
        throw(DimensionMismatch("Mismatch between B and D in (A ⊗ B)(C ⊗ D)"))
    end
    return (A * C) ⊗ (B * D)
end


"""
    lmul!(a::Number, K::AbstractKroneckerProduct)

Scale an `AbstractKroneckerProduct` `K` inplace by a factor `a` by rescaling the
**left** matrix.
"""
function LinearAlgebra.lmul!(a::Number, K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    lmul!(a, A)
end

"""
    rmul!(K::AbstractKroneckerProduct, a::Number)

Scale an `AbstractKroneckerProduct` `K` inplace by a factor `a` by rescaling the
**right** matrix.
"""
function LinearAlgebra.rmul!(K::AbstractKroneckerProduct, a::Number)
    A, B = getmatrices(K)
    rmul!(B, a)
end

function Base.:*(a::Number, K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    kronecker(a * A, B)
end


function Base.:*(K::AbstractKroneckerProduct, a::Number)
    A, B = getmatrices(K)
    kronecker(A, B * a)
end

function LinearAlgebra.power_by_squaring(K::KroneckerProduct, p::Integer)
    A, B = getmatrices(K)
    kronecker(A^p, B^p)
end

function LinearAlgebra.svdvals(K::KroneckerProduct)
    A, B = getmatrices(K)
    σA = svdvals(A)
    σB = svdvals(B)
    σ = sort!(kron(σA, σB), rev = true)
    n = minimum(size(K))
    if length(σ) < n
        append!(σ, zeros(eltype(σ), n - length(σ)))
    end
    return σ
end

# Broadcasting machinery

Base.copyto!(dest::AbstractMatrix, K::AbstractKroneckerProduct) = collect!(dest, K)

struct AbsKronProdStyle <: Broadcast.AbstractArrayStyle{2} end
AbsKronProdStyle(::Val{N}) where {N} = Broadcast.DefaultArrayStyle{N}()
AbsKronProdStyle(::Val{2}) = AbsKronProdStyle()

Base.BroadcastStyle(::Type{<:AbstractKroneckerProduct}) = AbsKronProdStyle()

function Base.similar(bc::Broadcast.Broadcasted{AbsKronProdStyle}, ::Type{T}) where {T}
    similar(Array{T}, axes(bc))
end

@inline function Base.copyto!(dest::AbstractArray, bc::Broadcast.Broadcasted{AbsKronProdStyle})
    @noinline throwdm(axdest, axsrc) =
        throw(DimensionMismatch("destination axes $axdest are not compatible with source axes $axsrc"))
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    # Some common cases that may be short-circuited
    # Case 1, example: A .= B
    if bc.args isa Tuple{AbstractKroneckerProduct}
        A = bc.args[1]
        collect!(bc.f, dest, A)
        return dest
    # Case 2, example: 2 .* B
    elseif bc.args isa Tuple{Number, AbstractKroneckerProduct}
        A = last(bc.args)
        n = first(bc.args)
        collect!(let n = n; x -> bc.f(n, x); end, dest, A)
        return dest
    # Case 3, example: B .* 2
    elseif bc.args isa Tuple{AbstractKroneckerProduct, Number}
        A = first(bc.args)
        n = last(bc.args)
        collect!(let n = n; x -> bc.f(x, n); end, dest, A)
        return dest
    end
    # An operation like K1 .+ K2 may be short-circuited if the component matrices
    # have the same size.
    bcf = Broadcast.flatten(bc)
    if all(x -> x isa AbstractKroneckerProduct, bcf.args)
        sz1 = map(size, getmatrices(bcf.args[1]))
        if all(x -> map(size, getmatrices(x)) == sz1, bcf.args[2:end])
            collect!(bcf.f, dest, bcf.args...)
            return dest
        end
    elseif all(x -> x isa Union{AbstractKroneckerProduct, StridedArray}, bcf.args)
        # Performance is better if the kronecker products are collected before the
        # broadcasted operation is performed, although this incurs allocations
        broadcast!(bcf.f, dest, map(_maybecollect, bcf.args)...)
        return dest
    end
    # The general case that indexes into each array.
    # This may be slow as indexing into an AbstractKroneckerProduct is expensive
    bc′ = Broadcast.preprocess(dest, bc)
    # Performance may vary depending on whether `@inbounds` is placed outside the
    # for loop or not. (cf. https://github.com/JuliaLang/julia/issues/38086)
    @inbounds @simd for I in eachindex(bc′)
        dest[I] = bc′[I]
    end
    return dest
end
