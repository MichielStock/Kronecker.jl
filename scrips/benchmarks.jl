#=
Created on Tuesday 2 July 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Benchmarking Kroncker.jl compated to native functions.
=#

using Kronecker, Plots, LinearAlgebra


sizes = [5, 10, 25, 50, 100, 250, 500, 1000, 5000]
Kwarmup = rand(10, 10) ⊗ rand(10, 10)

tmax = 10

# inverse


@elapsed inv(Kwarmup)

times_kron = []
times_naive = []

for s in sizes
    A = randn(s, s) / s^2
    B = rand(s, s) / s^2
    v = rand(s^2)
    K = A ⊗ B
    t = @elapsed inv(K)
    push!(times_kron, t)
    if s^2 < 5000
        K = collect(K)
        t = @elapsed inv(K)
        push!(times_naive, t)
        compute_naive = t < tmax
    end
end

plot(sizes.^2, times_kron, label="matrix inverse", color=:blue, legend=:bottomright, lw=2)
plot!(sizes[1:length(times_naive)].^2, times_naive, color=:blue, ls=:dash, label="", lw=2 )


# determinant

times_kron = []
times_naive = []

@elapsed det(Kwarmup)

for s in sizes
    A = randn(s, s)
    B = rand(s, s)
    v = rand(s^2)
    K = A ⊗ B
    t = @elapsed det(K)
    push!(times_kron, t)
    if s^2 < 5000
        K = collect(K)
        t = @elapsed det(K)
        push!(times_naive, t)
        compute_naive = t < tmax
    end
end

plot!(sizes.^2, times_kron, label="determinant", color=:green, lw=2)
plot!(sizes[1:length(times_naive)].^2, times_naive, color=:green, ls=:dash, label="", lw=2)

# squaring

times_kron = []
times_naive = []

v = rand(100)

@elapsed Kwarmup * Kwarmup

for s in sizes
    A = randn(s, s)
    B = rand(s, s)
    v = rand(s^2)
    K = A ⊗ B
    t = @elapsed K * K
    push!(times_kron, t)
    if s^2 < 5000
        K = collect(K)
        t = @elapsed K * K
        push!(times_naive, t)
        compute_naive = t < tmax
    end
end

plot!(sizes.^2, times_kron, label="squaring", color=:red, lw=2)
plot!(sizes[1:length(times_naive)].^2, times_naive, color=:red, ls=:dash, label="", lw=2)

# vector multiplication

times_kron = []
times_naive = []

v = rand(100)

@elapsed Kwarmup * v
@elapsed collect(Kwarmup) * v

for s in sizes
    A = randn(s, s)
    B = rand(s, s)
    v = rand(s^2)
    K = A ⊗ B
    t = @elapsed K * v
    push!(times_kron, t)
    if s^2 < 5000
        K = collect(K)
        t = @elapsed K * v
        push!(times_naive, t)
        compute_naive = t < tmax
    end
end

plot!(sizes.^2, times_kron, label="matrix-vector mult.", color=:orange, lw=2)
plot!(sizes[1:length(times_naive)].^2, times_naive, color=:orange, ls=:dash, label="", lw=2)


yaxis!(:log10)
ylabel!("CPU time (s)")
xaxis!(:log10)
xlabel!("Kronecker product size")
title!("Performance Kronecker.jl v.s. native code (--)")

savefig("benchmark.png")
