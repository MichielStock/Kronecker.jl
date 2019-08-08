#=
Created on Saturday 3 August 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Some methods to simulate Kronecker graphs.

Leskovec, J., Chakrabarti, D., Kleinberg, J., Faloutsos, C., & Ghahramani,
Z. (2008). Kronecker graphs: an approach to modeling networks.
Journal of Machine Learning Research, 11, 985–1042.
Retrieved from https://cs.stanford.edu/~jure/pubs/kronecker-jmlr10.pdf
=#

using SparseArrays: spzeros

"""
    isprob(A::AbstractArray)

Test if a matrix can be interpeted as a probability matrix, i.e., all elements
are between 0 and 1.
"""
isprob(A::AbstractArray) = all(0 .<= A .<= 1)

"""
    isprob(K::AbstractKroneckerProduct)

Test if a Kronecker product can be interpeted as a probability matrix,
i.e., all elements are between 0 and 1.
"""
isprob(K::AbstractKroneckerProduct) = isprob(K.A) && isprob(K.B)

"""
    naivesample(P::AbstractKroneckerProduct)

Sample a Kronecker graph from a probabilistic Kronecker product P using the
naive method. This method has a time complexity in the size of the Kronecker
product (but is still light in memory use). Consider using `fastsample`.
"""
function naivesample(P::AbstractKroneckerProduct)
    @assert isprob(P) throw(DomainError(
                            "All values of K should be between 0 and 1"))
    G = spzeros(Bool, size(P)...)
    for I in CartesianIndices(P)
        if P[I] > rand()  # QUESTION: is this the most efficient?
            G[I] = true
        end
    end
    return G
end
