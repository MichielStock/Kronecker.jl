using Pkg

tmp_packages = ["Kronecker", "Documenter"]

push!(LOAD_PATH,"../src/")

Pkg.activate(".")

Pkg.add.(tmp_packages) # IMPORTANT

using Documenter, Kronecker, LinearAlgebra

makedocs(sitename="Kronecker.jl",
        authors = "Michiel Stock",
        format = :html,
        modules = [Kronecker])

deploydocs(
        repo = "github.com/MichielStock/Kronecker.jl.git",
        )
