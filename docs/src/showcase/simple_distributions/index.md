# [Simple distributions](@id showcase_simple)

![](index.svg)

!!! details "Source Code"
    ````@eval
    import Markdown
    sourcecode = read(joinpath(dirname(@__FILE__), "index.jl"), String)
    Markdown.parse("""
    ```julia
    $sourcecode
    ```
    """)
    ````
