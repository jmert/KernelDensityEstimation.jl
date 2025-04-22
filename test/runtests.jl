using Test
using KernelDensityEstimation
const KDE = KernelDensityEstimation

struct IsEqualMatch{T} <: AbstractString
    msg::T
end
Base.isequal(m::IsEqualMatch, o) = occursin(m.msg, o)
Base.:(==)(m::IsEqualMatch, o::AbstractString) = occursin(m.msg, o)
Base.:(==)(o::AbstractString, m::IsEqualMatch) = occursin(m.msg, o)
function Base.show(io::IO, m::IsEqualMatch)
    print(io, "match")
    if m isa IsEqualMatch{Regex}
        Base.print_quoted(io, m.msg.pattern)
        print(io, "r")
    else
        Base.print_quoted(io, m.msg)
    end
end
macro match_str(msg, flags="")
    if flags == "r"
        return IsEqualMatch(Regex(msg))
    elseif flags == ""
        return IsEqualMatch(msg)
    else
        error("Unrecognized flag(s) `$flags`.")
    end
end


@testset verbose=true "KernelDensityEstimation tests" begin
    @testset "Convolutions" begin; include("conv.jl"); end
    @testset "Histograms" begin; include("histogram.jl"); end
    @testset verbose=true "Kernel Density Estimation" begin; include("kde.jl"); end
    @testset verbose=true "Extensions" begin
        if isdefined(Base, :get_extension)
            include("extensions.jl")
        else
            @test_skip "Skipped"
        end
    end
end
