using Documenter
using DocumenterCitations
using KernelDensityEstimation

doctest = "--fix"  in ARGS ? :fix :
          "--test" in ARGS ? true : false

DocMeta.setdocmeta!(KernelDensityEstimation, :DocTestSetup, :(using KernelDensityEstimation); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    format = Documenter.HTML(
        mathengine = Documenter.MathJax3(),
        assets = String["assets/citations.css"],
    ),
    sitename = "Kernel Density Estimation",
    authors = "Justin Willmert",
    pages = [
        "index.md",
        "tutorials.md",
        "howto.md",
        "explain.md",
        "reference.md",
    ],
    modules = [KernelDensityEstimation],
    doctest = doctest,
    remotes = nothing,
    warnonly = true,
    plugins = [bib],
)
