module FigureExpander

import Documenter: Documenter, Expanders, Selectors
import Markdown
import MarkdownAST


struct FigureNode <: MarkdownAST.AbstractBlock
    img::MarkdownAST.Node{Nothing}
    caption::MarkdownAST.Node{Nothing}
    params::Dict{Symbol,Any}
end
MarkdownAST.iscontainer(::FigureNode) = true
MarkdownAST.can_contain(::FigureNode, ::MarkdownAST.AbstractElement) = true

"""
Expands to a figure block, appropriate for both HTML and LaTeX output.

## Example

````md
```@figure; htmllabel = "Figure 1", latexlabel = "fig:key_name"
![alt text for accesibility](./source_file.ext)

Any _Markdown_ content can then be written as the rest of the "code" block,
which will be passed through to the `<figcaption>` / `\\caption` blocks in
HTML / ``\\LaTeX``, respectively.  The preceding Markdown image is parsed and
used to define the `<img>` / `\\includegraphics` elements, so that the image and
caption are included in the document as they should be when written natively.

Following the opening ` ```@figure`, optional parameters may be included
as a key-value list comma-separated values. The only two recognized parameters
at this time are:

- htmllabel

  A name which is prepended to the caption included in HTML output. Intended
  to give explicit text for use in referring to the figure.

- latexlabel

  The label name to emit in a `\\label` command after the caption, for use
  in references elsewhere in the LaTeX document.
```
````
"""
abstract type FigureBlock <: Expanders.NestedExpanderPipeline end

Selectors.order(::Type{FigureBlock}) = 50.0
Selectors.matcher(::Type{FigureBlock}, node, page, doc) = Documenter.iscode(node, r"^@figure")

param_regex = r"""
    (?<name>\w+) # word-like name
    \s*=\s*
    (?:
        # a quoted string expression (with escapes) --- https://stackoverflow.com/a/10786066
        "(?<strval>[^"\\]*(?:\\.[^"\\]*)*)"
    |
        # a simple literal expression -- no whitespace, not starting with "
        (?<litval>[^"][^\s]*)
    )
    \s*,?\s*  # eat comma separator and spaces, as necessary
    """x

function Selectors.runner(::Type{FigureBlock}, node, page, doc)
    @assert node.element isa MarkdownAST.CodeBlock
    x = node.element

    matched = match(r"^@figure\s*(?<params>;.*)?$", x.info)
    matched === nothing && error("invalid '@figure' syntax: $(x.info))")
    content = convert(MarkdownAST.Node, Markdown.parse(x.code))
    isempty(content.children) && error("invalid `@figure` block: block is empty")

    # expect to find image as the first element in the block
    imgpara = MarkdownAST.unlink!(first(content.children))
    img = only(imgpara.children)
    if !(img.element isa MarkdownAST.Image)
        error("invalid `@figure` block: expected first contents to be image, none found")
    end

    # everything that follows is the caption; if the node is now empty, add a placeholder
    # paragraph
    if isempty(content.children)
        push!(content.children, MarkdownAST.@ast MarkdownAST.Paragraph())
    end

    # parse parameters and store
    params = Dict{Symbol,String}()
    if !isnothing(matched[:params])
        paramstr = strip(lstrip(isequal(';'), matched[:params]))
        idx = firstindex(paramstr)
        while true
            m = match(param_regex, paramstr, idx)
            m === nothing && break
            params[Symbol(m[:name])] = something(m[:strval], m[:litval])
            idx = nextind(paramstr, idx, length(m.match))
        end
        if idx <= lastindex(paramstr)
            error("invalid `@figure` syntax: trailing string in `$paramstr`")
        end
    end

    node.element = FigureNode(img, content, params)
    return nothing
end

import Documenter.MDFlatten
function MDFlatten.mdflatten(io, ::MarkdownAST.Node, figure::FigureNode)
    MDFlatten.mdflatten(io, figure.img)
    if !isempty(figure.caption.children)
        print(io, '\n')
        MDFlatten.mdflatten(io, figure.caption)
    end
    return nothing
end

#########################################################################################

import Documenter.HTMLWriter: HTMLWriter, DCtx, get_url, pretty_url, relhref
import Documenter.DOM: @tags

function HTMLWriter.domify(dctx::DCtx, ::MarkdownAST.Node, node::FigureNode)
    ctx, navnode = dctx.ctx, dctx.navnode
    @tags figure figcaption

    navnode_dir = dirname(navnode.page)
    navnode_url = get_url(ctx, navnode)

    # we have to rewrite the image URL from relative to the markdown source to
    # relative to where Documenter is going to create the HTML file
    img = node.img.element
    imgsrc = joinpath(navnode_dir, img.destination)
    imgsrc = pretty_url(ctx, relhref(navnode_url, imgsrc))
    imgnode = MarkdownAST.@ast MarkdownAST.Image(imgsrc, img.title)

    inner = [HTMLWriter.domify(dctx, imgnode)]

    caption = node.caption
    if haskey(node.params, :htmllabel) && !isempty(node.params[:htmllabel])
        label = MarkdownAST.@ast MarkdownAST.Paragraph() do
            MarkdownAST.Strong() do
                node.params[:htmllabel]
            end
            ": "
        end
        prepend!(first(caption.children).children, label.children)
    end
    if !isempty(caption.children)
        push!(inner, figcaption(HTMLWriter.domify(dctx, caption)))
    end

    return figure(inner)
end

#########################################################################################

import Documenter.LaTeXWriter: LaTeXWriter, wrapinline, wrapblock

function LaTeXWriter.latex(io::LaTeXWriter.Context, node::MarkdownAST.Node, image::FigureNode)
    img = image.img.element
    caption = image.caption
    wrapblock(io, "figure") do
        LaTeXWriter._println(io, "\\centering")
        url = img.destination
        url = if Documenter.isabsurl(url)
            @warn "images with absolute URLs not supported in LaTeX output in $(Documenter.locrepr(io.filename))" url=url
            # We nevertheless output an \includegraphics with the URL. The LaTeX build will
            # then give an error, indicating to the user that something wrong. Only the
            # warning would be drowned by all the output from LaTeX.
            img.destination
        elseif startswith(url, '/')
            # URLs starting with a / are assumed to be relative to the document's root
            normpath(lstrip(url, '/'))
        else
            normpath(joinpath(dirname(io.filename), url))
        end
        url = replace(url, "\\" => "/") # use / on Windows too.
        #url, _ = splitext(url) # let LaTeX figure out the file extension
        wrapinline(io, "includegraphics[max width=\\linewidth]") do
            LaTeXWriter._print(io, url)
        end
        LaTeXWriter._println(io)
        wrapinline(io, "caption[$(img.title)]") do
            LaTeXWriter.latex(io, caption.children; toplevel = false)
        end
        if haskey(image.params, :latexlabel)
            wrapinline(io, "label") do
                LaTeXWriter._print(io, image.params[:latexlabel])
            end
        end
        LaTeXWriter._println(io)
    end
end

end # FigureExpander
