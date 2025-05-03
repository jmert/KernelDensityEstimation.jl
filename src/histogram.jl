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

Base.eltype(::HistEdge{T}) where {T} = T


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
                      edges::Tuple{Vararg{HistEdge,N}},
                      coord::Tuple{Vararg{Any,N}},
                      weight::W
                     ) where {R, N, W}
    idx = map(ii -> lookup(edges[ii], coord[ii])[1], ntuple(identity, Val(N)))
    dest[idx...] += oneunit(R) * weight
    return nothing
end

@inline is_bit_set(x, i) = !iszero((x >>> (i - 1)) & 0x01)

function _hist_inner!(::LinearBinning,
                      dest::AbstractArray{R,N},
                      edges::Tuple{Vararg{HistEdge,N}},
                      coord::Tuple{Vararg{Any,N}},
                      weight::W
                     ) where {R, N, W}
    Z = ntuple(identity, Val(N))
    len = map(ii -> edges[ii].nbin, Z)
    idx = map(ii -> lookup(edges[ii], coord[ii])[1], Z)
    # subtract off half of bin step to convert from fraction from left edge to fraction
    # away from center
    del = map(ii -> lookup(edges[ii], coord[ii])[2] - one(W) / 2, Z)

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
        frac = mapreduce(i -> bits[i] ? abs(del[i]) : one(del[i]) - abs(del[i]), *, Z)

        dest[idxs...] += oneunit(R) * weight * frac
    end
    return nothing
end

function _histogram!(binning::B,
                     dest::AbstractArray{R,N},
                     edges::Tuple{Vararg{HistEdge,N}},
                     data::AbstractVector{<:Tuple{Vararg{Any,N}}},
                     weights::Union{Nothing,<:AbstractVector},
                    ) where {B<:AbstractBinning, R, N}
    Z = ntuple(identity, Val(N))

    # run through data vector and bin entries if they are within bounds
    wsum = isnothing(weights) ? zero(_unitless(R)) : zero(eltype(weights))
    for ii in eachindex(data)
        coord = @inbounds data[ii]
        if !mapreduce(i -> edges[i].lo ≤ coord[i] ≤ edges[i].hi, &, Z)
            continue
        end
        w = isnothing(weights) ? one(_unitless(R)) : weights[ii]
        _hist_inner!(binning, dest, edges, coord, w)
        wsum += w
    end

    # no need to renormalize if everything was out-of-bounds
    iszero(wsum) && return wsum

    # apply renormalization
    #   N.B. treat zero-width bins as having unity scale factor to avoid dividing by zero
    invvol = mapreduce(i -> (s = edges[i].invstep; s / oneunit(s)), *, Z)
    norm = invvol / wsum
    for ii in eachindex(dest)
        @inbounds dest[ii] *= norm
    end

    return wsum
end

__hist_eltype(eltypes) = _invunit(typeof(mapreduce(oneunit, *, eltypes)))
_hist_eltype(edges::Tuple{Vararg{AbstractRange}}) = __hist_eltype(map(eltype, edges))
_hist_eltype(edges::Tuple{Vararg{HistEdge}}) = __hist_eltype(map(eltype, edges))

_hist_size(edges::Tuple{Vararg{AbstractRange}}) = map(e -> length(e) - 1, edges)
_hist_size(edges::Tuple{Vararg{HistEdge}}) = map(e -> e.nbin, edges)

function _histogram(binning::AbstractBinning,
                    data::AbstractVector{<:Tuple{Vararg{Any,N}}},
                    edges::Tuple{Vararg{HistEdge,N}};
                    weights::Union{Nothing,<:AbstractVector} = nothing
                   ) where {N}
    hist = zeros(_hist_eltype(edges), _hist_size(edges)...)
    _histogram!(binning, hist, edges, data, weights)
    return hist
end
function _histogram(binning::AbstractBinning,
                    data::AbstractVector{<:Tuple{Vararg{Any,N}}},
                    edges::Tuple{Vararg{AbstractRange,N}};
                    weights::Union{Nothing,<:AbstractVector} = nothing
                   ) where {N}
    edges′ = map(HistEdge, edges)
    return _histogram(binning, data, edges′; weights)
end
function _histogram(binning::AbstractBinning,
                    data::AbstractVector{<:Tuple{Vararg{Any,N}}},
                    edges::Union{<:HistEdge,<:AbstractRange}...;
                    weights::Union{Nothing,<:AbstractVector} = nothing
                   ) where {N}
    return _histogram(binning, data, edges; weights)
end

end  # module Histogramming
