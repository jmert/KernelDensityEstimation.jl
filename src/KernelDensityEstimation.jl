module KernelDensityEstimation

export kde

include("histogram.jl")
include("conv.jl")
include("kde.jl")


using PrecompileTools: @setup_workload, @compile_workload
@setup_workload begin
    v32 = Float32[(1:0.1:1)...]
    v64 = Float64[(1:0.1:1)...]
    @compile_workload begin
        kde(v32)
        kde(v64)
    end
end

end
