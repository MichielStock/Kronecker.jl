#=
using Pkg

tmp_packages = ["Kronecker", "Documenter"]

push!(LOAD_PATH,"../src/")

Pkg.activate(".")

Pkg.add.(tmp_packages) # IMPORTANT
=#

using Documenter, Kronecker, LinearAlgebra

makedocs(
    sitename = "Kronecker.jl",
    authors = "Michiel Stock",
    format = Documenter.HTML(),
    modules = [Kronecker],
    pages = Any[
        "Basic use" => "man/basic.md",
        "Types" => "man/types.md",
        "Multiplication" => "man/multiplication.md",
        "Factorization methods" => "man/factorization.md",
        "Indexed Kronecker products" => "man/indexed.md",
        "Kronecker sums" => "man/kroneckersums.md",
        "Kronecker powers and graphs" => "man/kroneckerpowers.md"
    ]
)

deploydocs(
    repo = "github.com/MichielStock/Kronecker.jl.git",
)
