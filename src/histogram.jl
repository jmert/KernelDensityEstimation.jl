"""
    histogram(v, nbins::Integer)
    histogram(v, edges::AbstractRange)

Bin the iterable collection `v` into a histogram the uniformly-spaced bins with  edges
`edges` (or `nbins` many bins between the data minimum and maximum).
The bins are each left half-open intervals except the last, which is a closed interval.
"""
function histogram end

function histogram(v, nbins::Integer)
    l, h = extrema(v)
    return histogram(v, range(l, h, length = nbins + 1))
end

function histogram(v, edges::AbstractRange{<:Real})
    off = first(edges)
    upl = last(edges)
    stp = step(edges)
    N = length(edges) - 1
    counts = zeros(Int, N)
    for x in v
        # N.B. ii is a 0-index offset
        ii = floor(Int, (x - off) / stp)
        if ii == N && x == upl
            ii = N - 1
        end
        (ii < 0 || ii >= N) && continue
        counts[ii + 1] += 1
    end
    return counts
end
