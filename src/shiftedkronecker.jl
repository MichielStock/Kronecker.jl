#TODO: document

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

    (A ⊗ B) + D

with D a diagonal matrix. Automatically computes the eigenvalue decompostion to
speed up some computations.
"""
struct ShiftedKroneckerProduct <: GeneralizedKroneckerProduct
    K::EigenKroneckerProduct
    D::Union{Diagonal, UniformScaling}
end

function Base.:show(io::IO, K::T) where T <: ShiftedKroneckerProduct
    println("Kronecker system of the form A ⊗ B + diagonal matrix")
end

function LinearAlgebra.:eigen(K::KroneckerProductArray)
    A, B = getmatrices(K)
    (typeof(A) <: Symmetric && typeof(B) <:Symmetric) || TypeError("function only implemented and relevant for symmetric matrices")
    return EigenKroneckerProduct(A, B, eigen(A), eigen(B))
end

#function getmatrices(K::EigenKroneckerProduct)
#    return getmatrices(K.K)
#end

#function Base.:inv(kronprod::T) where T <: EigenKroneckerProduct
#    return inv(kronprod.A) ⊗ inv(kronprod.B)
#end

function Base.:+(K::EigenKroneckerProduct, D::Union{Diagonal, UniformScaling})
    return ShiftedKroneckerProduct(K, D)
end

function Base.:+(K::KroneckerProductArray, D::Union{Diagonal, UniformScaling})
    return eigen(K) + D
end

function Base.:+(SK::ShiftedKroneckerProduct, D::Union{Diagonal, UniformScaling})
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
    return V ⊗ U * ((D + Diagonal(kron(λ, σ)))^-1 * ((V ⊗ U)' * v))
end
