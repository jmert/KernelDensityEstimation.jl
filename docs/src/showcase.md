````@setup

@eval Main begin
    using Markdown

    macro showcase_source(sourcefile = "index.jl")
        sourcecode = read(joinpath(dirname(@__FILE__), sourcefile), String)
        sourcecode = join(map(s -> isempty(s) ? s : " "^4 * s, split(sourcecode, '\n')), '\n')

        shield(t, c, l) = "https://img.shields.io/badge/$t-$c?style=for-the-badge&logo=$l&logoColor=white"
        codebutton     = """[![download code]($(shield("code","blue","julia")))]($sourcefile)"""
        projectbutton  = """[![download Project.toml]($(shield("Project.toml","green","toml")))](Project.toml)"""
        manifestbutton = """[![download Manifest.toml]($(shield("Manifest.toml","purple","toml")))](Manifest.toml)"""

        return Markdown.parse("""
        !!! details "Source Code"

            $codebutton
            $projectbutton
            $manifestbutton

            ```julia
        $sourcecode
            ```
        """)
    end
    nothing
end
````

# Showcase

## [Simple Distributions](showcase/simple_distributions/index.md)

![](showcase/simple_distributions/index.svg)

## [CosmoMC Weighted Chains (BK18 baseline likelihood analysis)](showcase/bk18_likelihood/index.md)

![](showcase/bk18_likelihood/index.svg)
