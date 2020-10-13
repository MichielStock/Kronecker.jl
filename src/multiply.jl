@inline _alloc_temp_array(size_1::Int, x::AbstractVector{T}) where T = zeros(T, size_1)
@inline _alloc_temp_array(size_1::Int, x::AbstractMatrix{T}) where T = zeros(T, size_1, size(x, 2))

const MatrixOrFactorization{T} = Union{AbstractMatrix{T}, Factorization{T}}

for N in (1, 2)
    for op in [:mul, :ldiv]
        square_kernel! = Symbol("_kron_", op, "_kernel_square!")

        op! = Symbol(op, "!")
        if N == 1
            op_expr_square = :(@views q[slc] = $op!(temp[1:n], m, q[slc]))
        else
            op_expr_square = :(@views q[slc, :] = $op!(temp[1:n, :], m, q[slc, :]))
        end

        @eval function $square_kernel!(temp::AbstractArray{T, $N}, q::AbstractArray{T, $N}, n::Int, i_left::Int, m::MatrixOrFactorization{T}, i_right::Int) where T
            # apply kron(I(l), m, I(r)) where m is square to the given vector x, overwriting x in the process

            if m isa Factorization || m isa Adjoint{T, <:Factorization} || m isa Transpose{T, <:Factorization} || m != I
                # if matrix is the identity, skip matmul/div, unless it's some sort factorization, then proceed anyway
                irc = i_right * n

                base_i, top_i = 0, (irc - i_right)
                @inbounds for i_l in 1:i_left
                    for i_r in 1:i_right
                        slc = base_i + i_r : i_right : top_i + i_r

                        $op_expr_square
                    end

                    base_i += irc
                    top_i += irc
                end
            end
            return q
        end

        square_func! = Symbol("kron_", op, "_fast_square!")
        @eval function $square_func!(out::AbstractArray{T, $N}, x::AbstractArray{T, $N}, matrices::NTuple{M, MatrixOrFactorization{T}}) where {T, M}
            ns::Vector{Int} = [size(m, 1) for m in matrices]
            i_left::Int = 1
            i_right::Int = prod(ns)

            out = copy!(out, x)
            temp = _alloc_temp_array(maximum(ns), x)

            for s in 1:length(ns)
                n = ns[s]
                i_right ÷= n
                $square_kernel!(temp, out, n, i_left, matrices[s], i_right)
                i_left *= n
            end
            return out
        end

        # NOTE: ldiv! seems to give erroneous results if m is not square,
        #       using allocating ldiv for now
        if N == 1
            if op == :mul
                op_expr_rect = :(@views q′[slc_out] = mul!(temp[1:r_h], m, q[slc_in]))
            else
                op_expr_rect = :(@views q′[slc_out] = m \ q[slc_in])
            end
        else
            if op == :mul
                op_expr_rect = :(@views q′[slc_out, :] = mul!(temp[1:r_h, :], m, q[slc_in, :]))
            else
                op_expr_rect = :(@views q′[slc_out, :] = m \ q[slc_in, :])
            end
        end

        rect_kernel! = Symbol("_kron_", op, "_kernel_rect!")
        @eval function $rect_kernel!(temp::AbstractArray{T, $N}, q::AbstractArray{T, $N}, r_h::Int, c_h::Int, i_left::Int, m::MatrixOrFactorization{T}, i_right::Int) where T
            # apply kron(I(i_left), m, I(i_right)) to the given vector q

            # don't bother checking for identity, since we know the matrix
            #  is rectangular here
            irc = i_right * c_h
            irr = i_right * r_h

            size_ = i_left * irr
            q′ = _alloc_temp_array(size_, q)

            base_i, base_j = 0, 0
            top_i, top_j = (irc - i_right), (irr - i_right)
            @inbounds for i_l in 1:i_left
                for i_r in 1:i_right
                    slc_in  = base_i + i_r : i_right : i_r + top_i
                    slc_out = base_j + i_r : i_right : i_r + top_j

                    $op_expr_rect
                end

                base_i += irc
                top_i += irc

                base_j += irr
                top_j += irr
            end

            return q′
        end

        rect_func! = Symbol("kron_", op, "_fast_rect!")
        ri = (op == :mul) ? 1 : 2
        ci = (op == :mul) ? 2 : 1
        @eval function $rect_func!(out::AbstractArray{T, $N}, x::AbstractArray{T, $N}, matrices::NTuple{M, MatrixOrFactorization{T}}) where {T, M}
            r::Vector{Int} = [size(m, $ri) for m in matrices]
            c::Vector{Int} = [size(m, $ci) for m in matrices]
            i_left::Int = 1
            i_right::Int = prod(c)

            out_ = copy(x)
            temp = _alloc_temp_array(maximum(r), x)

            for h in 1:length(matrices)
                r_h, c_h = r[h], c[h]
                i_right ÷= c_h
                if r_h == c_h
                    $square_kernel!(temp, out_, r_h, i_left, matrices[h], i_right)
                else
                    out_ = $rect_kernel!(temp, out_, r_h, c_h, i_left, matrices[h], i_right)
                end
                i_left *= r_h
            end

            out = copy!(out, out_)
            return out
        end
    end

    @eval function kronsum_mul_fast!(out::AbstractArray{T, $N}, x::AbstractArray{T, $N}, matrices::NTuple{M, AbstractMatrix{T}}) where {T, M}
        ns::Vector{Int} = [size(m, 1) for m in matrices]
        i_left::Int = 1
        i_right::Int = prod(ns)

        out = fill!(out, zero(T))
        temp = copy(x)
        # should use similar instead, but there seems to be a bug when using copy! with empty SparseArrays
        small_temp = _alloc_temp_array(maximum(ns), x)

        # this loop is technically parallelizable,
        #  though that'd end up using more memory
        for s in 1:length(ns)
            n = ns[s]
            i_right ÷= n
            copy!(temp, x)
            out += _kron_mul_kernel_square!(small_temp, temp, n, i_left, matrices[s], i_right)
            i_left *= n
        end
        return out
    end

    @eval function mul!(out::AbstractArray{T, $N}, K::AbstractKroneckerSum, x::AbstractArray{T, $N}) where T
        matrices = getallsummands(K)
        return kronsum_mul_fast!(out, x, matrices)
    end
end
