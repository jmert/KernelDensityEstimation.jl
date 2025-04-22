"""
    struct HistogramBinning <: AbstractBinningKDE end

Base case which generates a density estimate by histogramming the data.

See also [`LinearBinning`](@ref)
"""
struct HistogramBinning <: AbstractBinningKDE end

"""
    struct LinearBinning <: AbstractBinningKDE end

Base case which generates a density estimate by linear binning of the data.

See also [`HistogramBinning`](@ref)
"""
struct LinearBinning <: AbstractBinningKDE end

function _kdebin(::B, data, weights, lo, hi, nbins) where B <: Union{HistogramBinning, LinearBinning}
    T = eltype(data)  # unitful
    R = _invunit(T)
    U = _unitless(T)

    wd = hi - lo
    Δx = lo == hi ? zero(T) : wd / U(nbins)
    Δs = lo == hi ? oneunit(R) : U(nbins) / wd

    I = isnothing(weights) ? eachindex(data) : eachindex(data, weights)
    wsum = zero(U)
    f = zeros(R, nbins)
    for ii in I
        x = @inbounds data[ii]
        w = isnothing(weights) ? one(U) : @inbounds weights[ii]

        lo ≤ x ≤ hi || continue  # skip out-of-bounds elements
        if x == hi
            # handle the closed-right bound of the last bin specially
            zz = T(nbins)
            ii = nbins
        else
            # calculate fractional (0-indexed) bin position
            zz = Δs * (x - lo)
            # truncate to integer (1-indexed) bin index
            ii = unsafe_trunc(Int, zz) + 1

            # values exactly equal to the ideal bin edges can still cause problems;
            # compare the value against the next bin's edge value, and if x is not less than
            # the edge, adjust the index up by one more.
            right = lo + Δx * ii
            ii += x >= right
        end

        wsum += w
        if B === HistogramBinning
            f[ii] += w * oneunit(R)
        elseif B === LinearBinning
            # calculate fraction as relative distance from containing bin center
            ff = (zz - ii + 1) - one(U) / 2
            off = ifelse(signbit(ff), -1, 1)  # adjascent bin direction
            jj = clamp(ii + off, 1, nbins)  # adj. bin, limited to in-bounds where outer half-bins do not share

            ff = abs(ff)  # weights are positive
            f[ii] += w * oneunit(R) * (one(U) - ff)
            f[jj] += w * oneunit(R) * ff
        end
    end
    norm = (oneunit(T) * Δs) / wsum
    for ii in eachindex(f)
        f[ii] *= norm
    end
    return wsum, f
end

module Histogramming

import .._invunit, .._unitless

abstract type AbstractBinning end

struct HistogramBinning <: AbstractBinning end
struct LinearBinning <: AbstractBinning end

struct HistEdge{T,I}
    lo::T
    hi::T
    nbin::Int
    step::T
    invstep::I

    function HistEdge(lo::T, hi::T, nbin::Int) where {T}
        I = _invunit(T)
        U = _unitless(T)

        wd = hi - lo
        Δx = lo == hi ? zero(T) : wd / U(nbin)
        Δs = lo == hi ? oneunit(I) : U(nbin) / wd
        return new{T,I}(lo, hi, nbin, Δx, Δs)
    end
end
HistEdge(edge::AbstractRange) = HistEdge(first(edge), last(edge), length(edge) - 1)


function lookup(edge::HistEdge{T,I}, x::T) where {T,I}
    U = _unitless(T)
    # the right-most bin is closed right
    if x >= edge.hi
        return (edge.nbin, one(U))
    end

    # calculate fractional (0-indexed) bin position
    fracidx = edge.invstep * (x - edge.lo)
    # truncate to integer (1-indexed) bin index
    idx = unsafe_trunc(Int, fracidx) + 1

    # values exactly equal to the ideal bin edges can still cause problems;
    # compare the value against the next bin's edge value, and if x is not less than
    # the edge, adjust the index up by one more.
    right = edge.lo + edge.step * idx
    idx += x >= right

    # return (array index, fraction of bin beyond left edge)
    return (idx, fracidx - (idx - 1))
end

function _hist_inner!(::HistogramBinning,
                      dest::AbstractArray{R,N},
                      edges::NTuple{N,HistEdge{T,I}},
                      coord::NTuple{N,T},
                      weight::U
                     ) where {R, N, T, I, U}
    idx = map(ii -> lookup(edges[ii], coord[ii])[1], ntuple(identity, Val(N)))
    dest[idx...] += oneunit(R) * weight
    return nothing
end

@inline is_bit_set(x, i) = !iszero((x >>> (i - 1)) & 0x01)

function _hist_inner!(::LinearBinning,
                      dest::AbstractArray{R,N},
                      edges::NTuple{N,HistEdge{T,I}},
                      coord::NTuple{N,T},
                      weight::U
                     ) where {R, N, T, I, U}
    Z = ntuple(identity, Val(N))
    len = map(ii -> edges[ii].nbin, Z)
    idx = map(ii -> lookup(edges[ii], coord[ii])[1], Z)
    # subtract off half of bin step to convert from fraction from left edge to fraction
    # away from center
    del = map(ii -> lookup(edges[ii], coord[ii])[2] - one(U) / 2, Z)

    # iterate through all corners of a bounding hypercube by counting counting through
    # the permutations of {0,1} for each "axis" mapped to a bit in an integer

    # for bb in 0:(1 << N) - 1
    map(ntuple(i -> i - 1, Val(1 << N))) do bb
        # 0 = lower left (typically), 1 = upper right
        bits = map(i -> is_bit_set(bb, i), Z)
        # adjust from the 000... coordinate to the appropriate neighbor (±1 based on sign
        # of delta) by adding an offset if the bit is 1
        idxs = map(i -> clamp(idx[i] + copysign(bits[i], del[i]), 1, len[i]), Z)
        # the fraction of the weight to assign to a particular corner of the hypercube is
        # the volume of the intersection between the bin and a similarly-shaped cube around
        # the original coordinate
        frac = mapreduce(i -> bits[i] ? abs(del[i]) : one(U) - abs(del[i]), *, Z)

        dest[idxs...] += oneunit(R) * weight * frac
    end
    return nothing
end

end  # module Histogramming
