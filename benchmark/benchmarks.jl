using Kronecker
using BenchmarkTools

SUITE = BenchmarkGroup()

SUITE["multiply"] = BenchmarkGroup()

###############################################################################

SUITE["multiply"]["all_square"] = BenchmarkGroup()

KProd = kronecker([randn(2,2) for i in 1:10]...)
KPow = kronecker(randn(2,2), 10)
KSum = kroneckersum([randn(2,2) for i in 1:10]...)
v = randn(2^10)

SUITE["multiply"]["all_square"]["KroneckerProduct"] = @benchmarkable ($KProd * $v)
SUITE["multiply"]["all_square"]["KroneckerPower"] = @benchmarkable ($KPow * $v)
SUITE["multiply"]["all_square"]["KroneckerSum"] = @benchmarkable ($KSum * $v)

###############################################################################

SUITE["multiply"]["all_rectangular"] = BenchmarkGroup()

KProd = kronecker([randn(3,2) for i in 1:10]...)
KPow = kronecker(randn(3,2), 10)
v = randn(2^10)

SUITE["multiply"]["all_rectangular"]["KroneckerProduct"] = @benchmarkable ($KProd * $v)
SUITE["multiply"]["all_rectangular"]["KroneckerPower"] = @benchmarkable ($KPow * $v)

###############################################################################

SUITE["multiply"]["mixed_squareness"] = BenchmarkGroup()

KP = kronecker([randn(2,2) for i in 1:5]..., [randn(3,2) for i in 1:5]...)
v = randn(2^10)

SUITE["multiply"]["mixed_squareness"]["KroneckerProduct"] = @benchmarkable ($KP * $v)
