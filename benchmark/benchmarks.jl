using Kronecker
using BenchmarkTools

SUITE = BenchmarkGroup()

SUITE["mul_vec"] = BenchmarkGroup("vector", "multiplication")
SUITE["mul_mat"] = BenchmarkGroup("matrix", "multiplication")
SUITE["ldiv_vec"] = BenchmarkGroup("vector", "solve")
SUITE["ldiv_mat"] = BenchmarkGroup("matrix", "solve")

###############################################################################

for k in keys(SUITE)
    SUITE[k]["all_square"] = BenchmarkGroup()
end

KProd = kronecker([randn(2,2) for i in 1:10]...)
KPow = kronecker(randn(2,2), 10)
KSum = kroneckersum([randn(2,2) for i in 1:10]...)
v = randn(2^10)
V = randn(2^10, 7)

SUITE["mul_vec"]["all_square"]["KroneckerProduct"] = @benchmarkable ($KProd * $v)
SUITE["mul_vec"]["all_square"]["KroneckerPower"] = @benchmarkable ($KPow * $v)
SUITE["mul_vec"]["all_square"]["KroneckerSum"] = @benchmarkable ($KSum * $v)
SUITE["ldiv_vec"]["all_square"]["KroneckerProduct"] = @benchmarkable ($KProd \ $v)
SUITE["ldiv_vec"]["all_square"]["KroneckerPower"] = @benchmarkable ($KPow \ $v)
SUITE["ldiv_vec"]["all_square"]["KroneckerSum"] = @benchmarkable ($KSum \ $v)

###############################################################################

for k in keys(SUITE)
    SUITE[k]["all_rectangular"] = BenchmarkGroup()
end

KProd = kronecker([randn(3,2) for i in 1:10]...)
KPow = kronecker(randn(3,2), 10)

SUITE["mul_vec"]["all_rectangular"]["KroneckerProduct"] = @benchmarkable ($KProd * $v)
SUITE["mul_vec"]["all_rectangular"]["KroneckerPower"] = @benchmarkable ($KPow * $v)
SUITE["ldiv_vec"]["all_rectangular"]["KroneckerProduct"] = @benchmarkable ($KProd \ $v)
SUITE["ldiv_vec"]["all_rectangular"]["KroneckerPower"] = @benchmarkable ($KPow \ $v)

###############################################################################

for k in keys(SUITE)
    SUITE[k]["mixed_squareness"] = BenchmarkGroup()
end

KP = kronecker([randn(2,2) for i in 1:5]..., [randn(3,2) for i in 1:5]...)
v = randn(2^10)

SUITE["mul_vec"]["mixed_squareness"]["KroneckerProduct"] = @benchmarkable ($KP * $v)
SUITE["ldiv_vec"]["mixed_squareness"]["KroneckerProduct"] = @benchmarkable ($KP \ $v)
