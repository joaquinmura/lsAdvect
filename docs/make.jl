using LevelSetAdvection
using Documenter

makedocs(;
    modules=[LevelSetAdvection],
    authors="Mohamed Tarek <mohamed82008@gmail.com> and contributors",
    repo="https://github.com/joaquinmura/LevelSetAdvection.jl/blob/{commit}{path}#L{line}",
    sitename="LevelSetAdvection.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://joaquinmura.github.io/LevelSetAdvection.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/joaquinmura/LevelSetAdvection.jl",
)
