module KernelDensityEstimation

export kde

include("conv.jl")
include("kde.jl")


using PrecompileTools: @setup_workload, @compile_workload
@setup_workload let
    v32 = collect(exp2.(range(-10f0, 0f0, length = 16)))
    v64 = collect(exp2.(range(-10e0, 0e0, length = 16)))
    @compile_workload begin
        kde(v32)
        kde(v64)
    end
end

end
