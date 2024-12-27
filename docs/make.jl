using Documenter
using DocumenterCitations
using KernelDensityEstimation

doctest = "--fix"  in ARGS ? :fix :
          "--test" in ARGS ? true : false

DocMeta.setdocmeta!(KernelDensityEstimation, :DocTestSetup, :(using KernelDensityEstimation); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib");
                           style = :numeric)

showcases = map(ff -> joinpath("showcase", ff),
                filter!(endswith(".md"), readdir(joinpath(@__DIR__, "src", "showcase"))))

makedocs(
    format = Documenter.HTML(
        mathengine = Documenter.MathJax3(),
        assets = String["assets/citations.css"],
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
        "references.md",
    ],
    modules = [KernelDensityEstimation],
    doctest = doctest,
    remotes = nothing,
    warnonly = true,
    plugins = [bib],
)


deploydocs(
    repo = "github.com/jmert/KernelDensityEstimation.jl",
    push_preview = false
)
