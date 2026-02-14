using .KDE: estimate

using LinearAlgebra: Cholesky, Diagonal, Symmetric, cholesky
using Statistics: std
using Random: Random, randn
using Unitful

_bounds(; lo = nothing, hi = nothing, bc = :open) = (lo, hi, bc)

edges2centers(r) = KDE._edgerange_to_centers((r,))[1]
centers2edges(r) = (Δ = step(r) / 2; range(first(r) - Δ, last(r) + Δ, length = length(r) + 1))

rv_norm_σ = 2.1
rv_norm_x = rv_norm_σ .* randn(Random.seed!(Random.default_rng(), 1234), 100)

@testset "Interface" begin
    # eltype and ndims inherited from AbstractKDE definition
    @test eltype(KDE.UnivariateKDE{Float32}) == Float32
    @test eltype(KDE.BivariateKDE{Int}) == Int
    @test eltype(KDE.MultivariateKDE{BigFloat,3}) == BigFloat
    @test eltype(KDE.AbstractKDE{Float64,6}) == Float64
    @test ndims(KDE.UnivariateKDE{Float32}) == 1
    @test ndims(KDE.BivariateKDE{Int}) == 2
    @test ndims(KDE.MultivariateKDE{BigFloat,3}) == 3
    @test ndims(KDE.AbstractKDE{Float64,6}) == 6

    # instances are constructible - all with tuples of axes and density array
    K1t = KDE.UnivariateKDE((0.0:1.0,), zeros(1))
    K2t = KDE.BivariateKDE((0.0:1.0, 0.0:1.0,), zeros(1, 1))
    K3t = KDE.MultivariateKDE((0.0:1.0, 0.0:1.0, 0.0:1.0), zeros(1, 1, 1))
    # and uni-/bivariate with second construct that does not require the tuples
    K1d = KDE.UnivariateKDE(0.0:1.0, zeros(1))
    K2d = KDE.BivariateKDE(0.0:1.0, 0.0:1.0, zeros(1, 1))


    # base property list is just the field list
    pp = Set([:axes, :density])
    @test Set(propertynames(K3t)) == pp
    # and uni-/bivariate add extra convenience shortcuts
    @test Set(propertynames(K1t)) == pp ∪ Set((:x, :f))
    @test Set(propertynames(K2t)) == pp ∪ Set((:x, :y, :f))

    # getproperty then works inferrably for all of these options (when used syntactically
    # as `a.b`, not `getproperty(a, :b)` where getting can be dynamic).
    # @inferred requires a function call, so wrap property access in a wrapper function.
    @inline getaxes(K) = K.axes
    @test @inferred(getaxes(K1t)) isa NTuple{1, StepRangeLen}
    @test @inferred(getaxes(K2t)) isa NTuple{2, StepRangeLen}
    @test @inferred(getaxes(K3t)) isa NTuple{3, StepRangeLen}
    @inline getdensity(K) = K.density
    @test @inferred(getdensity(K1t)) isa Array{Float64, 1}
    @test @inferred(getdensity(K2t)) isa Array{Float64, 2}
    @test @inferred(getdensity(K3t)) isa Array{Float64, 3}
    @inline getf(K) = K.f
    @test @inferred(getf(K1t)) isa Array{Float64, 1}
    @test @inferred(getf(K2t)) isa Array{Float64, 2}
    @inline getx(K) = K.x
    @test @inferred(getx(K1t)) isa StepRangeLen
    @test @inferred(getx(K2t)) isa StepRangeLen
    @inline gety(K) = K.y
    @test @inferred(gety(K2t)) isa StepRangeLen

    # uni-/bivariate structures can be iterates to unpack their axes and density (to
    # facilitate unpacking before Julia v1.8 where property destructuring was added)
    x, f, rest... = K1t
    @test iterate(rest) === nothing
    x, y, f, rest... = K2t
    @test iterate(rest) === nothing

    SIMPLE_TYPES = [
        (Float32,), (Float64,),
        (Float32, Float32,), (Float32, Float64,), (Float64, Float64,),
        (Float32, Float32, Float32), (Float32, Float64, Float64), (Float64, Float64, Float64),
    ]

    @testset "MultivariateKDE[Info] of eltypes $eltypes" for eltypes in SIMPLE_TYPES
        E = promote_type(eltypes...)
        N = length(eltypes)

        # N.B. check especially for concrete types!

        # multivariate KDE objects
        T = KDE.multivariate_type_from_axis_eltypes(eltypes...)
        @test T <: KDE.MultivariateKDE{E,N}
        @test Base.isconcretetype(T)
        @test fieldtype(T, :axes) <: Tuple{map(e -> AbstractRange{e}, eltypes)...}
        @test fieldtype(T, :density) <: Array{E,N}

        # corresponding info structures
        T = KDE.multivariateinfo_type_from_axis_eltypes(eltypes...)
        @test T <: KDE.MultivariateKDEInfo{E,N}
        @test Base.isconcretetype(T)
        @test fieldtype(T, :bandwidth) <: Union{Nothing, (N == 1 ? E : Cholesky{E})}
        @test fieldtype(T, :kernel) <: Union{Nothing, KDE.MultivariateKDE{E,N}}
    end
end

@testset "Options Handling" begin
    import .KDE: init

    # Make sure the option processing is fully inferrable
    WT = @NamedTuple{bounds::NTuple{3,Any}, nbins::Any, bandwidth::Any, bwratio::Any}
    #  don't cover too many variations, since that just adds to the test time
    variations = [WT((bounds, nbins, bandwidth, bwratio)) for
        bounds in ((nothing, nothing, :open), (-1, nothing, KDE.Closed)),
        nbins in (nothing, 10),
        bandwidth in (KDE.SilvermanBandwidth(), 1.0),
        bwratio in (1,)
    ]
    method = KDE.LinearBinning()

    RT = Tuple{Tuple{Vector{Float64}}, Nothing, KDE.UnivariateKDEInfo{Float64}}
    @testset "options = $kws" for kws in variations
        @test @inferred(init(method, [1.0]; kws...)) isa RT
    end
    # also test that default values work
    @test @inferred(init(method, [1.0], bandwidth = 1.0)) isa RT

    # Unused options that make it down to the option processing step log a warning message
    @test_logs((:warn, "Unused keyword argument(s)"),
               KDE.init(method, [1.0], bandwidth = 1.0, unusedarg=true))
end

@testset "Bounds Handling" begin
    # conversion from symbols
    @test convert(KDE.Boundary.T, :open)        == KDE.Open
    @test convert(KDE.Boundary.T, :closed)      == KDE.Closed
    @test convert(KDE.Boundary.T, :closedleft)  == KDE.ClosedLeft
    @test convert(KDE.Boundary.T, :closedright) == KDE.ClosedRight
    @test_throws(ArgumentError(match"Unknown boundary condition: .*?"r),
                 convert(KDE.Boundary.T, :something))

    # inferring boundary specifications
    @test KDE.bounds([1.0, 2.0], (-Inf, Inf, nothing)) === (1.0, 2.0, KDE.Open)
    @test KDE.bounds([1.0, 2.0], (0.0, Inf, nothing))  === (0.0, 2.0, KDE.ClosedLeft)
    @test KDE.bounds([1.0, 2.0], (-Inf, 0, nothing))   === (1.0, 0.0, KDE.ClosedRight)
    @test KDE.bounds([1.0, 2.0], (0, 1, nothing))      === (0.0, 1.0, KDE.Closed)
    for bnds in [(NaN, 1.0), (Inf, 1.0), (0.0, NaN), (0.0, -Inf)]
        @test_throws(ArgumentError(match"Invalid [^\s]+ bound: `(hi|lo) = .*?`"r),
                     KDE.bounds(Float64[], (bnds..., nothing)))
    end
    @test_throws(ArgumentError(match"Cannot infer boundary conditions with unspecified limits.*"r),
                 KDE.bounds(Float64[], (nothing, nothing, nothing)))


    v = [1.0, 2.0]
    kws = (; lo = 0.0, hi = 5.0, nbins = 5, bandwidth = 1.0, bwratio = 1)
    kws_bc(bc) = (; bounds = (kws.lo, kws.hi, bc), kws.nbins, kws.bandwidth, kws.bwratio)

    # closed boundaries -- histogram cells will span exactly lo/hi input
    x = range(kws.lo, kws.hi, length = kws.nbins + 1)
    k, info = estimate(KDE.HistogramBinning(), v; kws_bc(:closed)...)
    @test info.domain[1][3] === KDE.Closed
    @test length(k.x) == kws.nbins
    @test step(k.x) == step(x)
    @test centers2edges(k.x) == x

    # open boundaries --- histogram extends to 8x further left/right than lo/hi
    x = range(kws.lo - 8kws.bandwidth, kws.hi + 8kws.bandwidth, length = kws.nbins + 1)
    k, info = estimate(KDE.HistogramBinning(), v; kws_bc(:open)...)
    @test info.domain[1][3] === KDE.Open
    @test length(k.x) == kws.nbins
    @test step(k.x) == step(x)
    @test centers2edges(k.x) == x

    # half-open boundaries --- either left or right side is extended, while the other is
    # exact
    x = range(kws.lo, kws.hi + 8kws.bandwidth, length = kws.nbins + 1)
    k, info = estimate(KDE.HistogramBinning(), v; kws_bc(:closedleft)...)
    @test info.domain[1][3] === KDE.ClosedLeft
    @test length(k.x) == kws.nbins
    @test step(k.x) == step(x)
    @test centers2edges(k.x) == x

    x = range(kws.lo - 8kws.bandwidth, kws.hi, length = kws.nbins + 1)
    k, info = estimate(KDE.HistogramBinning(), v; kws_bc(:closedright)...)
    @test info.domain[1][3] === KDE.ClosedRight
    @test length(k.x) == kws.nbins
    @test step(k.x) == step(x)
    @test centers2edges(k.x) == x

    # test that the half-open aliases work
    k, info = estimate(KDE.HistogramBinning(), v; kws_bc(:openleft)...)
    @test info.domain[1][3] === KDE.ClosedRight
    k, info = estimate(KDE.HistogramBinning(), v; kws_bc(:openright)...)
    @test info.domain[1][3] === KDE.ClosedLeft

    # bounds=(lo, hi) is interpreted as bounds=(lo, hi, :open)
    k1, info = estimate(KDE.HistogramBinning(), v; kws_bc(:open)...)
    kws′ = merge(kws_bc(:open), (; bounds = (kws.lo, kws.hi)))
    k2, info = estimate(KDE.HistogramBinning(), v; kws′...)
    @test k1.x == k2.x && k1.f == k2.f

    # bounds= argument overrides lo=,hi=,boundary= and warns
    M = KDE.LinearBinning()
    @test_logs((:warn, "Keyword `bounds` is overriding non-nothing `lo`, `hi`, and/or `boundary`."),
               kde([1.0]; method = M, bounds = (0.0, 1.0, :open), lo = -1.0, bandwidth = 1.0))
    @test_logs((:warn, "Keyword `bounds` is overriding non-nothing `lo`, `hi`, and/or `boundary`."),
               kde([1.0]; method = M, bounds = (0.0, 1.0, :open), hi = 2.0, bandwidth = 1.0))
    @test_logs((:warn, "Keyword `bounds` is overriding non-nothing `lo`, `hi`, and/or `boundary`."),
               kde([1.0]; method = M, bounds = (0.0, 1.0, :open), boundary = :closed, bandwidth = 1.0))


    # argument error for unknown specifications rather than a method error
    struct UnknownBoundsSpec end
    @test_throws ArgumentError KDE.bounds(v, UnknownBoundsSpec())
    @test_throws ArgumentError KDE.bounds(UnknownBoundsSpec(), (0.0, 1.0, :closed))
end

@testset "Direct estimates ($(N)D)" for N in 1:3
    # By "direct", we mean that we bypass the init() step and call the appropriate
    # estimate(method, data, weights, info) method.

    @testset "$msg" for (msg, unit) in [("Unitless", 1.0), ("Unitful", 1.0u"m")]
        # raw data
        coords = unit .* [0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 4.5]
        samples = ntuple(_ -> coords, Val(N))
        # units in density space
        Tinvunit = typeof(inv(unit) ^ N)

        # construct an info object
        info_type = KDE.multivariateinfo_type_from_axis_eltypes(map(eltype, samples)...)
        info = info_type(KDE.HistogramBinning())
        # manually set properties that will normally be set by init()
        info.bwratio = ntuple(_ -> 1, Val(N))
        info.bandwidth = N == 1 ? 1.0 : cholesky(Matrix(Diagonal(ones(N))))

        # use implicit uniform weight case
        weights = nothing

        @testset "Simple binning" begin
            info.method = method = KDE.HistogramBinning()

            ## closed domain
            info.kernel = nothing
            info.domain = ntuple(_ -> (0unit, 5unit, KDE.Closed), Val(N))
            K, info = @inferred estimate(method, samples, weights, info)
            # binned density has correct form
            # N.B. no need to test the binning itself — already done in the histogram tests
            @test K isa KDE.MultivariateKDE{Tinvunit,N}
            @test all(step(a) == 1unit for a in K.axes)  # bwratio and bandwidth given, so step is fixed
            @test all(extrema(a) == (0.5unit, 4.5unit) for a in K.axes)

            # kronecker-delta kernel has been populated
            @test info.kernel isa KDE.MultivariateKDE{Float64,N}
            @test sum(info.kernel.density) == 1.0
            @test prod(length, info.kernel.axes) == 1
            @test length(info.kernel.density) == 1


            ## (mixed) half-open interval(s)
            info.kernel = nothing
            info.domain = ntuple(i -> (0unit, 5unit, i == 1 ? KDE.OpenRight : KDE.Closed), Val(N))
            K, info = @inferred estimate(method, samples, weights, info)
            @test all(step(a) == 1unit for a in K.axes)  # bwratio and bandwidth given, so step is fixed
            @test extrema(K.axes[1]) == (0.5unit, 4.5unit + 8.0unit #= 2 × bandwidth=#)
            if N > 1
                @test all(extrema(a) == (0.5unit, 4.5unit) for a in K.axes[2:end])
            end
        end

        @testset "Basic density" begin
            info.method = method = KDE.BasicKDE()

            ## open domain
            info.kernel = nothing
            info.domain = ntuple(_ -> (0unit, 5unit, KDE.Closed), Val(N))
            K, info = @inferred estimate(method, samples, weights, info)
            # binned density has correct form
            @test K isa KDE.MultivariateKDE{Tinvunit,N}
            @test all(step(a) == 1unit for a in K.axes)  # bwratio and bandwidth given, so step is fixed
            #@test all(extrema(a) == (0.5unit, 4.5unit) for a in K.axes)
            # gaussian density has been populated
            @test info.kernel isa KDE.MultivariateKDE{Float64,N}
            @test sum(info.kernel.density) ≈ 1.0
            @test size(info.kernel.density) == ntuple(_ -> 2#=sided=# * 4#=sigma=# + 1, Val(N))
        end
    end
end

@testset "Simple Binning" begin
    kws = (; bandwidth = KDE.SilvermanBandwidth(), bwratio = 1)

    # raw data
    v₁ = Float64[0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 4.5]
    # non-zero elements of the probability **mass** function
    #   N.B. KDE returns probability **density** function
    h₁ = [1, 2, 3, 2, 1] ./ length(v₁)
    @test sum(h₁) ≈ 1.0
    # bandwidth estimate
    ν, σ̂ = length(v₁), std(v₁, corrected = false)
    b₀ = σ̂ * (4//3 / ν)^(1/5)  # Silverman's rule

    # prepared using known limits and number of bins
    k₁, info = estimate(KDE.HistogramBinning(), v₁;
                        bounds = _bounds(lo=0, hi=5, bc=:closed), nbins = 5, kws...)
    Δx = step(k₁.x)
    @test k₁.x == edges2centers(0.0:1.0:5.0)
    @test isapprox(k₁.f .* Δx, h₁)
    @test sum(k₁.f) * Δx ≈ 1.0
    @test info.bandwidth[1] ≈ b₀

    # prepared with known number of bins, but automatically selected bounds
    k₂, info = estimate(KDE.HistogramBinning(), v₁;
                        bounds = _bounds(bc = :closed), nbins = 5, kws...)
    Δx = step(k₂.x)
    @test k₂.x == edges2centers(range(0.5, 4.5, length = 5 + 1))
    @test isapprox(k₂.f .* Δx, h₁)
    @test sum(k₂.f) * Δx ≈ 1.0
    @test info.bandwidth[1] ≈ b₀

    # prepared with unknown number of bins, but known bounds
    k₃, info = estimate(KDE.HistogramBinning(), v₁;
                        bounds = _bounds(lo=0, hi=5, bc=:closed), kws...)
    Δx = step(k₃.x)
    @test k₃.x == edges2centers(range(0.0, 5.0, length = round(Int, 5 / b₀) + 1))
    @test isapprox(filter(!iszero, k₃.f) .* Δx, h₁)
    @test sum(k₃.f) * Δx ≈ 1.0
    @test info.bandwidth[1] ≈ b₀

    # prepared with unknown limits and number of bins
    k₄, info = estimate(KDE.HistogramBinning(), v₁;
                        bounds = _bounds(bc=:closed), kws...)
    Δx = step(k₄.x)
    @test isapprox(filter(!iszero, k₄.f) .* Δx, h₁)
    @test sum(k₄.f) * Δx ≈ 1.0
    @test info.bandwidth[1] ≈ b₀

    # prepared using an alternative sampling density (compared to previous)
    k₅, info = estimate(KDE.HistogramBinning(), v₁;
                        bounds = _bounds(bc=:closed), kws..., bwratio = 16)
    Δx = step(k₅.x)
    @test step(k₅.x) < step(k₄.x)
    @test isapprox(filter(!iszero, k₅.f) .* Δx, h₁)
    @test sum(k₅.f) * Δx ≈ 1.0
    @test info.bandwidth[1] ≈ b₀

    # make sure errors do not occur when uniform data is provided
    let (k, info) = estimate(KDE.HistogramBinning(), ones(100);
                             bounds = _bounds(bc=:closed), kws...)
        @test collect(k.x) == [1.0]
        @test isfinite(info.bandwidth[1]) && !iszero(info.bandwidth[1])
        @test step(k.x) == 0.0  # zero-width bin
        @test sum(k.f) == 1.0  # like a Kronecker delta
    end

    # make sure the bandwidth argument is converted to the appropriate common type
    _, info = estimate(KDE.HistogramBinning(), v₁; kws...)
    @test eltype(info.bandwidth) === Float64
    _, info = estimate(KDE.HistogramBinning(), v₁; bandwidth = 1)  # explicit but not same type
    @test eltype(info.bandwidth) === Float64
    _, info = estimate(KDE.HistogramBinning(), Float32.(v₁); kws...)
    @test eltype(info.bandwidth) === Float32
    _, info = estimate(KDE.HistogramBinning(), Float32.(v₁); bandwidth = 1)
    @test eltype(info.bandwidth) === Float32


    expT = Tuple{KDE.UnivariateKDE{Float64,<:AbstractRange{Float64},<:AbstractVector{Float64}},
                 KDE.UnivariateKDEInfo{Float64}}
    @test @inferred(estimate(KDE.HistogramBinning(), [1.0, 2.0]; nbins = 2, kws...)) isa expT
end

@testset "Weighting" begin
    rv = rv_norm_x
    N = length(rv)

    Nlen = Float64(N)
    Npos = Float64(count(>=(0), rv))
    weight1 = ones(length(rv))
    weight2 = 2 .* weight1
    wpositive = [1.0 + (x ≥ 0) for x in rv]

    # shortcut to avoid repeating ourselves...
    _initinfo(v, args...; kwargs...) = KDE.init(KDE.BasicKDE(), v, args...; kwargs...)[3]

    # using weight values
    @test _initinfo(rv).neffective === Nlen
    @test _initinfo(rv, nothing).neffective === Nlen
    @test _initinfo(rv, weight1).neffective === Nlen
    @test _initinfo(rv, weight2).neffective === Nlen

    # weight values with bounds that exclude data
    @test _initinfo(rv, nothing; bounds = (0.0, nothing)).neffective === Npos
    @test _initinfo(rv, weight1; bounds = (0.0, nothing)).neffective === Npos
    @test _initinfo(rv, weight2; bounds = (0.0, nothing)).neffective === Npos

    # weight type differing from data type
    @test _initinfo(rv, Int.(weight1)).neffective === Nlen
    @test _initinfo(rv, Float32.(weight1)).neffective === Nlen

    # having non-uniform weights leads to a smaller effective sample size (for the same
    # number of samples)
    info1 = _initinfo(rv, weight1)
    info2 = _initinfo(rv, wpositive)
    @test info1.neffective > info2.neffective

    # Bandwidth estimation accounts for weights
    @testset "$(nameof(typeof(bandwidth)))" for bandwidth in (KDE.SilvermanBandwidth(), KDE.ISJBandwidth())
        bounds = (-5rv_norm_σ, 5rv_norm_σ, KDE.Open)

        # uniform weights reduce to the same effective sample size
        bw0 = KDE.bandwidth(bandwidth, rv, bounds...)
        bw1 = KDE.bandwidth(bandwidth, rv, bounds...; weights = weight1)
        bw2 = KDE.bandwidth(bandwidth, rv, bounds...; weights = weight2)
        @test bw0 == bw1
        @test bw0 == bw2
    end

    # send weights through the high-level interface
    K0 = kde(rv)
    K1 = kde(rv, weights = weight1)
    @test K1 == K0

    # the mean of distribution should shift to the right when positive values are
    # weighted twice as much
    Kp = kde(rv, weights = wpositive)
    @test sum(K1.x .* K1.f) < sum(Kp.x .* Kp.f)
end

@testset "Gaussian Kernel" begin
    x_odd = range(-6.0, 6.0, length = 501)
    x_even = range(-6.0, 6.0, length = 500)

    σ = sqrt(2.0π)
    C1 = cholesky(Symmetric(π .* fill(2.0, 1, 1), :L))
    C2 = cholesky(Symmetric(π .* [2.0 0.5; 0.5 1.0], :L))
    ϵ = eps(σ)^(3/4)

    # verify that specialized implementations agree with the general case
    GenericT = Tuple{NTuple{N, StepRangeLen}, Cholesky} where {N}
    # 1D specalization - s = scalar, 1 = generic 1, o = odd
    gso = KDE.kernel_gaussian((x_odd,), σ)
    g1o = invoke(KDE.kernel_gaussian, GenericT, (x_odd,), C1)
    @test gso ≈ g1o rtol=ϵ
    # 1D specialization - s = scalar, 1 = generic, e = even
    gse = KDE.kernel_gaussian((x_even,), σ)
    g1e = invoke(KDE.kernel_gaussian, GenericT, (x_even,), C1)
    @test gse ≈ g1e rtol=ϵ
    # 2D specalization - d = double, 2 = generic, o = odd
    gdo = KDE.kernel_gaussian((x_odd, x_odd), C2)
    g2e = invoke(KDE.kernel_gaussian, GenericT, (x_odd, x_odd), C2)
    @test gdo ≈ g2e rtol=ϵ
    # 2D specalization - d = double, 2 = generic, e = even
    gde = KDE.kernel_gaussian((x_even, x_even), C2)
    g2e = invoke(KDE.kernel_gaussian, GenericT, (x_even, x_even), C2)
    @test gde ≈ g2e rtol=ϵ

    # everything should be mass normalized properly (implicitly true between specialized
    # and generic implementations already if elementwise checks pass)
    @test sum(gso) ≈ 1.0
    @test sum(gse) ≈ 1.0
    @test sum(gdo) ≈ 1.0
    @test sum(gde) ≈ 1.0


    # 2D specialization checks that the cholesky matrix is 2x2
    @test_throws DimensionMismatch KDE.kernel_gaussian((x_odd, x_odd), C1)
    # generic version also checks that the choleksy matrix is appropriately sized
    @test_throws DimensionMismatch KDE.kernel_gaussian((x_odd, x_odd, x_even), C2)
end

@testset "Basic Kernel Density Estimate" begin
    nbins = 11  # use odd value to have symmetry
    nsamp = 10nbins
    Δx = 1.0 / nbins

    v_uniform = range(-1, 1, length = nsamp)
    p_uniform = 1.0 / nbins

    # Tune the bandwidth to be much smaller than the bin sizes, which effectively means
    # we just get back the internal histogram (when performing only a basic KDE).
    (x, f), _ = estimate(KDE.BasicKDE(), v_uniform;
                         bounds = _bounds(bc=:closed), nbins = nbins, bandwidth = 0.1Δx)
    @test all(isapprox.(f .* step(x), p_uniform, atol = 1e-3))
    # Then increase the bandwidth to be the same size of the bins. The outermost bins will
    # impacted by the kernel convolving with implicit zeros beyond the edges of the
    # distribution, so we should expect a significant decrease in the first and last bin.
    (x, f), _ = estimate(KDE.BasicKDE(), v_uniform;
                         bounds = _bounds(bc=:closed), nbins = nbins, bandwidth = Δx)
    @test all(isapprox.(f[2:end-1] .* step(x), p_uniform, atol = 1e-3))  # approximate p_uniform
    @test all(<(-1e-3), f[[1,end]] .* step(x) .- p_uniform) # systematically less than p_uniform

    # If we put a Kronecker delta in the center of a span with a wide (lo, hi) spread,
    # we should effectively just get back the convolution kernel.
    v_kron = zeros(100)
    bw = 1.0

    (x, f1), _ = estimate(KDE.BasicKDE(), v_kron; bounds = (-6bw, 6bw, :closed), nbins = 251, bandwidth = bw)
    (x, f2), _ = estimate(KDE.BasicKDE(), v_kron; bounds = (-6bw, 6bw, :closed), nbins = 251, bandwidth = bw)
    g = exp.(.-(x ./ bw) .^2 ./ 2) ./ (bw * sqrt(2π)) .* step(x)
    @test all(isapprox.(f1 .* step(x), g, atol = 1e-5))
    @test all(isapprox.(f2 .* step(x), g, atol = 1e-5))

    kws = (; bandwidth = KDE.SilvermanBandwidth())

    # On an open interval, probability should be conserved.
    k, _ = estimate(KDE.BasicKDE(), v_uniform; bounds = _bounds(bc=:open), kws...)
    @test sum(k.f) * step(k.x) ≈ 1.0

    # But on (semi-)closed intervals, the norm will drop
    kl, _ = estimate(KDE.BasicKDE(), v_uniform; bounds = _bounds(bc=:closedleft), kws...)
    nl = sum(kl.f) * step(kl.x)
    @test nl < 0.96

    kr, _ = estimate(KDE.BasicKDE(), v_uniform; bounds = _bounds(bc=:closedright), kws...)
    nr = sum(kr.f) * step(kr.x)
    @test nr < 0.96
    @test nl ≈ nr  # should be nearly symmetric

    kc, _ = estimate(KDE.BasicKDE(), v_uniform; bounds = _bounds(bc=:closed), kws...)
    nc = sum(kc.f) * step(kc.x)
    @test nc < 0.91
end

@testset "Linear Boundary Correction" begin
    nbins = 11  # use odd value to have symmetry
    nsamp = 10nbins
    Δx = 1.0 / nbins

    v_uniform = range(-1, 1, length = nsamp)
    p_uniform = 1.0 / nbins

    # Enable the normalization option, which makes a correction for the implicit zeros
    # being included in the convolution
    (x, f), _ = estimate(KDE.LinearBoundaryKDE(), v_uniform;
                         bounds = _bounds(bc=:closed), nbins = nbins, bandwidth = Δx)
    @test all(isapprox.(f .* step(x), p_uniform, atol = 1e-3))  # approximately p_uniform
    # and that correction keeps working as the bandwidth causes multiple bins to be affected
    (x, f), _ = estimate(KDE.LinearBoundaryKDE(), v_uniform;
                         bounds = _bounds(bc=:closed), nbins = nbins, bandwidth = 6Δx)
    @test all(isapprox.(f .* step(x), p_uniform, atol = 1e-3))  # approximately p_uniform

    # The boundary correction only applies when a hard boundary is encountered, so the
    # BasicKDE and LinearBoundaryKDE should provide very similar distributions when
    # the boundary is Open.
    (x₁, f₁), _ = estimate(KDE.BasicKDE(), v_uniform,
                           bounds = _bounds(bc=:open), nbins = nbins, bandwidth = 2Δx)
    (x₂, f₂), _ = estimate(KDE.LinearBoundaryKDE(), v_uniform;
                           bounds = _bounds(bc=:open), nbins = nbins, bandwidth = 2Δx)
    @test all(isapprox.(f₁, f₂, atol = 1e-3))
end

@testset "Type Handling" begin
    kws = (; method = KDE.MultiplicativeBiasKDE(), bandwidth = KDE.ISJBandwidth())
    rv = rv_norm_x
    K1 = kde(rv; kws...)

    # Equality and hashing
    K2 = kde(rv; kws...)
    @test K1 !== K2
    @test K1 == K2
    @test hash(K1) == hash(K2)
    K3 = kde(rv[1:50]; kws...)
    @test K1 !== K3
    @test K1 != K3
    @test hash(K1) != hash(K3)

    K4 = kde(Float32.(rv); kws...)
    @test eltype(K1) == eltype(K1.x) == eltype(K1.f) == Float64
    @test eltype(K4) == eltype(K4.x) == eltype(K4.f) == Float32

    @testset "Unitful numbers" begin
        urv = Quantity.(rv, u"m")
        Ku = @inferred kde(urv; kws...)

        # The density is presumably not identical due to differences in how the compiler
        # was able to optimize the floating point operations, but we still expect pretty
        # stringent agreement
        @test K1.x == Ku.x ./ u"m"
        @test K1.f ≈ Ku.f .* u"m"  rtol=2eps(1.0)

        U = typeof(1.0u"m")
        R = typeof(1 / 1.0u"m")
        @test Ku isa KDE.UnivariateKDE{R, <:AbstractRange{U}, <:AbstractVector{R}}
    end
end

@testset "Show ($(N)D)" for N in 1:3
    typename = N == 1 ? "UnivariateKDE" :
               N == 2 ? "BivariateKDE" :
               "MultivariateKDE"

    axes = ((-10:0.01:10 for _ in 1:N)...,)
    dims = ((25 for _ in 1:N)...,)

    K = KDE.MultivariateKDE(axes, randn(dims...))
    infoT = KDE.multivariateinfo_type_from_axis_eltypes(map(eltype, axes)...)
    info = infoT(KDE.BasicKDE())
    info.bwratio = ((8.0 for _ in 1:N)...,)
    info.bandwidth_alg = KDE.SilvermanBandwidth()
    info.nbins = dims
    info.neffective = prod(dims)
    info.kernel = KDE.MultivariateKDE(((range(0.0, 0.0, length=1) for _ in 1:N)...,),
                                      fill(1.0, (1 for _ in 1:N)...))

    buf = IOBuffer()

    @testset "$(typename)" begin
        # Check that compact-mode printing is shorter than full printing
        show(IOContext(buf, :compact => true), K)
        shortmsg = String(take!(buf))
        show(IOContext(buf, :compact => false), K)
        longmsg = String(take!(buf))
        @test length(shortmsg) < length(longmsg)
        @test occursin("…", shortmsg) && !occursin("…", longmsg)

        # For the longer output, there's both a limited and unlimited variation. Again,
        # check that the lengths differ as expected.
        show(IOContext(buf, :limit => true), K)
        limitlongmsg = String(take!(buf))
        @test length(shortmsg) < length(limitlongmsg)
        @test length(limitlongmsg) < length(longmsg)

        # Verify the headers appear as expected --- the expected header is the alias name
        # only after the `public` feature was added
        longname = "MultivariateKDE{Float64, $N}"
        shortname = N >= 3 || !isdefined(Base, :ispublic) ? longname : "$(typename){Float64}"

        @test occursin(shortname, shortmsg)
        @test occursin(shortname, limitlongmsg)
        fulltype = sprint(show, typeof(K), context = IOContext(buf, :limit => false, :compact => false))
        @test occursin(fulltype, longmsg)  # full parametric typing
    end

    @testset "$(typename)Info" begin
        # Check that compact-mode printing is shorter than full printing
        show(IOContext(buf, :compact => true), info)
        shortmsg = String(take!(buf))
        show(IOContext(buf, :compact => false), info)
        longmsg = String(take!(buf))
        @test length(shortmsg) < length(longmsg)
        @test occursin("…", shortmsg) && !occursin("…", longmsg)

        # For the longer output, there's both a limited and unlimited variation. Again,
        # check that the lengths differ as expected.
        show(IOContext(buf, :limit => true), info)
        limitlongmsg = String(take!(buf))
        @test length(shortmsg) < length(limitlongmsg)
        @test length(limitlongmsg) < length(longmsg)

        # Verify the headers appear as expected --- the expected header is the alias name
        # only after the `public` feature was added
        longname = "MultivariateKDEInfo{Float64, $N}"
        shortname = N >= 3 || !isdefined(Base, :ispublic) ? longname : "$(typename)Info{Float64}"

        @test occursin(shortname, shortmsg)
        @test occursin(shortname, limitlongmsg)
        fulltype = sprint(show, typeof(info), context = IOContext(buf, :limit => false, :compact => false))
        @test occursin(fulltype, longmsg)  # full parametric typing

        # The three-arg version for text/plain output is more sophisticated
        ioc = IOContext(buf, :displaysize => (50, 120), :limit => true)
        show(ioc, MIME"text/plain"(), info)
        pretty = String(take!(buf))
        plines = split(strip(pretty), '\n')
        header = popfirst!(plines)
        # heading line contains type info
        @test occursin(shortname, header)
        # fields are then properly padded
        alignment = map(l -> findfirst(==(':'), l), plines)
        @test all(==(alignment[1]), alignment)
        # the sentinel width is properly kept up-to-date
        padding = map(l -> findfirst(!=(' '), l) - 1, plines)
        @test minimum(padding) == 2
    end
end

if isdefined(Test, :detect_closure_boxes)
    @testset "Codegen quality" begin
        @test length(Test.detect_closure_boxes(KDE)) == 0
    end
end
