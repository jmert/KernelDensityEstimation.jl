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
