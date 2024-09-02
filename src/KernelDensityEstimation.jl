module KernelDensityEstimation

export kde

include("conv.jl")
include("kde.jl")


using PrecompileTools: @setup_workload, @compile_workload
@setup_workload begin
    v32 = collect(range(1f0, 2f0, length = 4))
    v64 = collect(range(1e0, 2e0, length = 4))
    @compile_workload begin
        kde(v32)
        kde(v64)
    end
end

end
