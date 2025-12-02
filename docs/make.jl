using Documenter
using DocumenterCitations
using DocumenterInterLinks
using KernelDensityEstimation

include("FigureExpander.jl")

function mdglob(root)
    return map(filter!(endswith(".md"), readdir(joinpath(@__DIR__, "src", root)))) do ff
        joinpath(root, ff)
    end
end


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

makedocs(
    format = [
        #Documenter.LaTeX(),
        Documenter.HTML(
            mathengine = Documenter.MathJax3(),
            assets = String[
                "assets/citations.css",
                "assets/overrides.css",
            ],
        ),
    ],
    sitename = "Kernel Density Estimation",
    authors = "Justin Willmert",
    pages = [
        "index.md",
        "userguide.md",
        "extensions.md",
        "explain.md",
        hide("showcase.md", mdglob("showcase")),
        "api.md",
        "releasenotes.md",
        hide("devdocs/index.md", mdglob("devdocs")),
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
