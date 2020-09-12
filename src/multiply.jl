@inline _alloc_temp_array(size_1::Int, x::AbstractVector{T}) where T = zeros(T, size_1)
@inline _alloc_temp_array(size_1::Int, x::AbstractMatrix{T}) where T = zeros(T, size_1, size(x, 2))

function _kron_mv_kernel_square!(temp::AbstractArray{T, 1}, q::AbstractArray{T, 1}, n::Int, i_left::Int, m::AbstractMatrix{T}, i_right::Int) where T<:Number
    # apply kron(I(l), m, I(r)) where m is square to the given vector x, overwriting x in the process

    if m != I # if matrix is the identity, skip matmul/div
        irc = i_right * n

        base_i, top_i = 0, (irc - i_right)
        @inbounds for i_l in 1:i_left
            for i_r in 1:i_right
                slc = base_i + i_r : i_right : top_i + i_r
                @views q[slc] = mul!(temp[1:n], m, q[slc])
            end

            base_i += irc
            top_i += irc
        end
    end
    return q
end


function _kron_mv_kernel_square!(temp::AbstractArray{T, 2}, q::AbstractArray{T, 2}, n::Int, i_left::Int, m::AbstractMatrix{T}, i_right::Int) where T<:Number
    # apply kron(I(l), m, I(r)) where m is square to the given vector x, overwriting x in the process

    if m != I # if matrix is the identity, skip matmul/div
        irc = i_right * n

        base_i, top_i = 0, (irc - i_right)
        @inbounds for i_l in 1:i_left
            for i_r in 1:i_right
                slc = base_i + i_r : i_right : top_i + i_r
                @views q[slc, :] = mul!(temp[1:n, :], m, q[slc, :])
            end

            base_i += irc
            top_i += irc
        end
    end
    return q
end


function _kron_mv_kernel_rect(q::AbstractArray{T, 1}, r_h::Int, c_h::Int, i_left::Int, m::AbstractMatrix{T}, i_right::Int) where T<:Number
    # apply kron(I(i_left), m, I(i_right)) to the given vector q

    # don't bother checking for identity, since we know the matrix
    #  is rectangular here
    irc = i_right * c_h
    irr = i_right * r_h

    size_ = i_left * irr
    q′ = zeros(T, size_)

    base_i, base_j = 0, 0
    top_i, top_j = (irc - i_right), (irr - i_right)
    @inbounds for i_l in 1:i_left
        for i_r in 1:i_right
            slc_in  = base_i + i_r : i_right : i_r + top_i
            slc_out = base_j + i_r : i_right : i_r + top_j
            @views q′[slc_out] = m * q[slc_in]
        end

        base_i += irc
        top_i += irc

        base_j += irr
        top_j += irr
    end

    return q′
end

function _kron_mv_kernel_rect(q::AbstractArray{T, 2}, r_h::Int, c_h::Int, i_left::Int, m::AbstractMatrix{T}, i_right::Int) where T<:Number
    # apply kron(I(i_left), m, I(i_right)) to the given vector q

    # don't bother checking for identity, since we know the matrix
    #  is rectangular here
    irc = i_right * c_h
    irr = i_right * r_h

    size_ = i_left * irr
    q′ = zeros(T, size_, size(q, 2))

    base_i, base_j = 0, 0
    top_i, top_j = (irc - i_right), (irr - i_right)
    @inbounds for i_l in 1:i_left
        for i_r in 1:i_right
            slc_in  = base_i + i_r : i_right : i_r + top_i
            slc_out = base_j + i_r : i_right : i_r + top_j
            @views q′[slc_out, :] = m * q[slc_in, :]
        end

        base_i += irc
        top_i += irc

        base_j += irr
        top_j += irr
    end

    return q′
end


for N in (1, 2)
    @eval function kron_mv_fast_square!(out::AbstractArray{T, $N}, x::AbstractArray{T, $N}, matrices::AbstractMatrix{T}...) where T<:Number
        ns::Vector{Int} = [size(m, 1) for m in matrices]
        i_left::Int = 1
        i_right::Int = prod(ns)

        out = copy!(out, x)
        temp = _alloc_temp_array(maximum(ns), x)

        for s in 1:length(ns)
            n = ns[s]
            i_right ÷= n
            _kron_mv_kernel_square!(temp, out, n, i_left, matrices[s], i_right)
            i_left *= n
        end
        return out
    end

    @eval function kronsum_mv_fast!(out::AbstractArray{T, $N}, x::AbstractArray{T, $N}, matrices::AbstractMatrix{T}...) where T<:Number
        ns::Vector{Int} = [size(m, 1) for m in matrices]
        i_left::Int = 1
        i_right::Int = prod(ns)

        out = fill!(out, zero(T))
        temp = similar(x)
        stemp = _alloc_temp_array(maximum(ns), x)

        # this loop is technically parallelizable,
        #  though that'd end up using more memory
        for s in 1:length(ns)
            n = ns[s]
            i_right ÷= n
            copy!(temp, x)
            out += _kron_mv_kernel_square!(stemp, temp, n, i_left, matrices[s], i_right)
            i_left *= n
        end
        return out
    end

    @eval function kron_mv_fast_rect!(out::AbstractArray{T, $N}, x::AbstractArray{T, $N}, matrices::AbstractMatrix{T}...) where T<:Number
        r::Vector{Int} = [size(m, 1) for m in matrices]
        c::Vector{Int} = [size(m, 2) for m in matrices]
        out = copy!(out, x)

        i_left::Int = 1
        i_right::Int = prod(c)
        stemp = _alloc_temp_array(maximum(r), x)

        for h in 1:length(matrices)
            r_h, c_h = r[h], c[h]
            i_right ÷= c_h
            if r_h == c_h
                _kron_mv_kernel_square!(stemp, out, r_h, i_left, matrices[h], i_right)
            else
                out = _kron_mv_kernel_rect(out, r_h, c_h, i_left, matrices[h], i_right)
            end
            i_left *= r_h
        end

        return out
    end
end


function mul!(out::AbstractVecOrMat, K::AbstractKroneckerSum, x::AbstractVecOrMat)
    matrices = getallsummands(K)
    return kronsum_mv_fast!(out, x, matrices...)
end
