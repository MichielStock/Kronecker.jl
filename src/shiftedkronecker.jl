"""
    struct EigenKroneckerProduct <: KroneckerProduct

Eigen value decomposition of a Kronecker product.
"""
struct EigenKroneckerProduct <: KroneckerProduct
    A
    B
    Aeigen::Eigen
    Beigen::Eigen
end

"""
    struct ShiftedKroneckerProduct <: GeneralizedKroneckerProduct

System of the form

    (A ⊗ B) + cI.

Automatically computes the eigenvalue decompostion to
speed up some computations.
"""
struct ShiftedKroneckerProduct <: GeneralizedKroneckerProduct
    K::EigenKroneckerProduct
    D::UniformScaling
end

function Base.:show(io::IO, K::T) where T <: ShiftedKroneckerProduct
    print(io, "A ⊗ B + cI")
end


"""
    eigen(K::KroneckerProductArray)

Compute the eigenvalue decomposition of system of the from (A ⊗ B). Returns
an instance of the type `EigenKroneckerProduct`.
"""
function LinearAlgebra.:eigen(K::KroneckerProductArray)
    A, B = getmatrices(K)
    (typeof(A) <: Symmetric && typeof(B) <:Symmetric) || TypeError("function only implemented and relevant for symmetric matrices")
    return EigenKroneckerProduct(A, B, eigen(A), eigen(B))
end

function Base.:+(K::EigenKroneckerProduct, D::UniformScaling)
    return ShiftedKroneckerProduct(K, D)
end

function Base.:+(K::KroneckerProductArray, D::UniformScaling)
    return eigen(K) + D
end

function Base.:+(SK::ShiftedKroneckerProduct, D::UniformScaling)
    return SK.K + (SK.D + D)  # just update the weights
end

function Base.:collect(SK::ShiftedKroneckerProduct)
    return collect(SK.K) + SK.D
end

function Base.:\(SK::ShiftedKroneckerProduct, v::V where V <: AbstractVector)
    K = SK.K
    λ, V = K.Aeigen
    σ, U = K.Beigen
    D = SK.D
    # note that this should go fast
    return V ⊗ U * ((Diagonal(kron(λ, σ))+ D)^-1 * ((V ⊗ U)' * v))
end

Base.:/(v::V where V <: AbstractVector, SK::ShiftedKroneckerProduct) = \(SK, v)

"""
    solve(SK::ShiftedKroneckerProduct, v::V where V <: AbstractVector)

Solves a linear system of the form

(A ⊗ B + cI) x = v,

where (A ⊗ B + cI) is given by an instance of `ShiftedKroneckerProduct`.
"""
solve(SK::ShiftedKroneckerProduct, v::V where V <: AbstractVector) = \(SK, v)
