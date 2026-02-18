# Highest Posterior Density interval containing a target level for a 1D density
function hpd(x::AbstractVector, v::AbstractVector, level::Real = 0.995)
    p = v ./ sum(v)
    _, m = findmax(p)
    s = sort(p, rev = true)
    c = cumsum(s)
    t = s[searchsortedlast(c, level)]
    li = p[begin] > t ? firstindex(p) : findlast(<(t), @views(p[begin:m]))
    hi = p[end] > t ? lastindex(p) : findfirst(<(t), @views(p[m:end]))
    l = @view(x[begin:m])[max(begin, li)]
    h = @view(x[m:end])[min(hi, end)]
    return (l, h)
end

# Highest Posterior Density contour levels containing the desired area for a 2D density
function hpd(v::AbstractArray, levels::AbstractVector{<:Real} = [0.68, 0.95])
    s = sort(vec(v), rev = true)
    c = cumsum(s) ./ sum(s)
    return [s[searchsortedfirst(c, l)] for l in levels]
end
