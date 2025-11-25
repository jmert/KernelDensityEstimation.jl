using Documenter
using DocumenterCitations
using DocumenterInterLinks
using KernelDensityEstimation

doctest = "--fix"  in ARGS ? :fix :
          "--test" in ARGS ? true :
          get(ENV, "CI", "false") == "true"

DocMeta.setdocmeta!(KernelDensityEstimation, :DocTestSetup, quote
                        using KernelDensityEstimation
                        const KDE = KernelDensityEstimation
                    end; recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib");
                           style = :numeric)
links = InterLinks(
    "Julia" => "https://docs.julialang.org/en/v1/",
    "UnicodePlots" => "https://juliaplots.org/UnicodePlots.jl/stable/",
)

showcases = map(ff -> joinpath("showcase", ff),
                filter!(endswith(".md"), readdir(joinpath(@__DIR__, "src", "showcase"))))

makedocs(
    format = Documenter.HTML(
        mathengine = Documenter.MathJax3(),
        assets = String[
            "assets/citations.css",
            "assets/overrides.css",
        ],
    ),
    sitename = "Kernel Density Estimation",
    authors = "Justin Willmert",
    pages = [
        "index.md",
        "userguide.md",
        "extensions.md",
        "explain.md",
        hide("showcase.md", showcases),
        "api.md",
        "releasenotes.md",
        "references.md",
    ],
    modules = [KernelDensityEstimation],
    doctest = doctest,
    remotes = nothing,
    warnonly = true,
    plugins = [bib, links],
)


deploydocs(
    repo = "github.com/jmert/KernelDensityEstimation.jl",
    push_preview = false
)
