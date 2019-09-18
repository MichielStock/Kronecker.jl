using Documenter, Kronecker, LinearAlgebra

makedocs(sitename="Kronecker.jl",
        authors = "Michiel Stock",
        modules = [Kronecker])

deploydocs(
        repo = "github.com/MichielStock/Kronecker.jl.git",
        )
