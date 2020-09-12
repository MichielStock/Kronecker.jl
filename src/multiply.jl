for N in (1,2)
    @eval function _kron_mv_kernel_square!(q::AbstractArray{T, $N}, n::Int, i_left::Int, m::AbstractMatrix{T}, i_right::Int) where T<:Number
        # apply kron(I(l), m, I(r)) where m is square to the given vector x, overwriting x in the process

        if m != I # if matrix is the identity, skip matmul/div
            @inbounds for i_l in 1:i_left, i_r in 1:i_right
                slc = (((i_l-1) * n)*i_right + i_r) : i_right : (((i_l * n) - 1)*i_right + i_r)
                if $N == 1
                    @views q[slc] = m * q[slc]
                else
                    @views q[slc, :] = m * q[slc, :]
                end
            end
        end
        return q
    end

    @eval function _kron_mv_kernel_rect(q::AbstractArray{T, $N}, r_h::Int, c_h::Int, i_left::Int, m::AbstractMatrix{T}, i_right::Int) where T<:Number
        # apply kron(I(i_left), m, I(i_right)) to the given vector q

        # don't bother checking for identity, since we know the matrix
        #  is rectangular here
        irc = i_right * c_h
        irr = i_right * r_h

        size_ = i_left * irr
        if $N == 1
            q′ = zeros(T, size_)
        else
            q′ = zeros(T, size_, size(q, 2))
        end

        base_i, base_j = 0, 0
        @inbounds for i_l in 1:i_left
            for i_r in 1:i_right
                slc_in  = base_i + i_r : i_right : base_i + i_r + (irc-i_right)
                slc_out = base_j + i_r : i_right : base_j + i_r + (irr-i_right)

                if $N == 1
                    @views q′[slc_out] = m * q[slc_in]
                else
                    @views q′[slc_out, :] = m * q[slc_in, :]
                end
                # change this to mul!
            end

            base_i += irc
            base_j += irr
        end

        return q′
    end

end


function kron_mv_fast_square!(out::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, matrices::AbstractMatrix{T}...) where T<:Number
    ns::Vector{Int} = [size(m, 1) for m in matrices]
    i_left::Int = 1
    i_right::Int = prod(ns)

    out = copy!(out, x)
    # stemp = zeros(T, maximum(ns))

    for s in 1:length(ns)
        n = ns[s]
        i_right ÷= n
        _kron_mv_kernel_square!(out, n, i_left, matrices[s], i_right)
        i_left *= n
    end
    return out
end



function kronsum_mv_fast!(out::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, matrices::AbstractMatrix{T}...) where T<:Number
    ns::Vector{Int} = [size(m, 1) for m in matrices]
    i_left::Int = 1
    i_right::Int = prod(ns)

    out = fill!(out, zero(T))
    temp = similar(x)
    # stemp = zeros(T, maximum(ns))

    # this loop is technically parallelizable,
    #  though that'd end up using more memory
    for s in 1:length(ns)
        n = ns[s]
        i_right ÷= n
        copy!(temp, x)
        out += _kron_mv_kernel_square!(temp, n, i_left, matrices[s], i_right)
        i_left *= n
    end
    return out
end

function kron_mv_fast_rect!(out::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, matrices::AbstractMatrix{T}...) where T<:Number
    r::Vector{Int} = [size(m, 1) for m in matrices]
    c::Vector{Int} = [size(m, 2) for m in matrices]
    out = copy!(out, x)

    i_left::Int = 1
    i_right::Int = prod(c)

    # stemp = zeros(T, maximum(r))

    for h in 1:length(matrices)
        r_h, c_h = r[h], c[h]
        i_right ÷= c_h
        if r_h == c_h
            _kron_mv_kernel_square!(out, r_h, i_left, matrices[h], i_right)
        else
            out = _kron_mv_kernel_rect(out, r_h, c_h, i_left, matrices[h], i_right)
        end
        i_left *= r_h
    end

    return out
end

function mul!(out::AbstractVecOrMat, K::AbstractKroneckerSum, x::AbstractVecOrMat)
    matrices = getallsummands(K)
    return kronsum_mv_fast!(out, x, matrices...)
end
