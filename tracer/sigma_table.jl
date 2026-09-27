# Tabulated collision cross section sigma(E_cm), log-log interpolated.
# Used by ParticleTracing3D.jl's --sigma-model table (see tracer/data/bafplus_ne_q1.tsv
# and tools/cross_section/ for how the default BaF+-Ne table was computed).

const SIGMA_TABLE_COLUMNS = Dict("central" => 2, "low" => 3, "high" => 4)

"""
    load_sigma_table(path, variant) -> (logE, logQ)

Reads a whitespace-separated table (lines starting with # are comments) whose first
column is the centre-of-mass collision energy E/k_B in K and whose columns 2/3/4 are the
central/low/high cross sections in m^2. Returns natural logs for log-log interpolation.
"""
function load_sigma_table(path::AbstractString, variant::AbstractString)
    haskey(SIGMA_TABLE_COLUMNS, variant) || error("--sigma-table-variant must be central, low or high")
    col = SIGMA_TABLE_COLUMNS[variant]
    E = Float64[]; Q = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        f = split(s)
        push!(E, parse(Float64, f[1])); push!(Q, parse(Float64, f[col]))
    end
    length(E) >= 2 || error("sigma table $path has fewer than 2 rows")
    issorted(E) && allunique(E) || error("sigma table $path energies must be strictly increasing")
    all(>(0), E) && all(>(0), Q) || error("sigma table $path must contain positive values")
    return log.(E), log.(Q)
end

"""
    sigma_from_table(logE, logQ, E_K)

Log-log linear interpolation of the table at collision energy E_K (K); clamped to the
end values outside the tabulated range.
"""
@inline function sigma_from_table(logE, logQ, E_K)
    x = log(max(E_K, 1e-300))
    x <= logE[1] && return exp(logQ[1])
    x >= logE[end] && return exp(logQ[end])
    i = searchsortedlast(logE, x)
    t = (x - logE[i]) / (logE[i+1] - logE[i])
    return exp(logQ[i] + t * (logQ[i+1] - logQ[i]))
end
