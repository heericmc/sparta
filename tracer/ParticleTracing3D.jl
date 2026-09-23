# True-3D frozen-field particle tracer, for cases/fluor-cell-3d (a real,
# non-axisymmetric geometry -- an off-axis buffer gas inlet -- so the 2D
# axisymmetric ParticleTracing.jl cannot be used as-is).
#
# This is NOT a rewrite of the physics. It reuses ParticleTracing.jl's
# collision kinematics, free-path sampling, and trajectory/statistics
# machinery UNCHANGED (collide!, exact_partner, freePath, propagate,
# SimulateParticles are all already geometry-agnostic -- they just call
# whatever interpolate!/getCollision closures build_field() hands them).
# Only three things are genuinely different for a real 3D case, and this
# file replaces exactly those:
#   1. Geometry: reads cell_fluor3d.surf's 3D "Points"/"Triangles" format
#      (not 2D "Points"/"Lines"), and getCollision() does a real 3D
#      segment-vs-triangle-mesh test (Moller-Trumbore) instead of 2D line
#      intersection.
#   2. Field: reads a genuine 3D SPARTA grid dump (cell centres in x,y,z
#      with their own u,v,w,nrho,temp -- not a 2D r-z field with an
#      assumed azimuthal rotation, which is only valid for an axisymmetric
#      geometry and this one is not).
#   3. Spawn position/CLI args are x,y,z directly, not r,z,phi.
#
# See ../docs/hpc-fluor-cell-3d-molecules-handoff.md for how to run this
# and what parameters to use for the BaF+ case. Smoke-tested locally
# against a synthetic cube geometry + synthetic field (single- and
# multi-threaded, both a slow near-thermal case and a fast 500eV-speed
# case) -- NOT against the real cell_fluor3d.surf/field.grid, which
# weren't available on the machine that wrote this. Treat the first real
# run as the actual first end-to-end test, and see the handoff doc's
# validation checklist before trusting results from it.
#
# Derived from ParticleTracing.jl (see that file's own header for its
# further upstream provenance). GNU GPL version 3; see ../licenses/GPL-3.0.txt.

using Distributed
using Random
using LinearAlgebra
using NearestNeighbors
using CSV
using DataFrames
using Printf
using ArgParse
using OnlineStats
using SpecialFunctions
using Statistics

import Base: convert

# ---- Statistics machinery: identical to ParticleTracing.jl ----

struct TrajStats
    v::OnlineStat
    t::OnlineStat
    ncolls::OnlineStat
    lfree::OnlineStat
end

@inline TrajStats() = TrajStats(CovMatrix(2), Variance(), Variance(), Variance())

struct StatsArray
    stats::Matrix{TrajStats}
    minr::Float64
    maxr::Float64
    rbins::Int
    minz::Float64
    maxz::Float64
    zbins::Int
    rstep::Float64
    zstep::Float64
end

@inline function StatsArray(minr, maxr, rbins, minz, maxz, zbins)
    stats = Array{TrajStats}(undef, rbins, zbins)
    for i in 1:rbins
        for j in 1:zbins
            stats[i,j] = TrajStats()
        end
    end
    return StatsArray(stats, minr, maxr, rbins, minz, maxz, zbins, rbins/(maxr-minr), zbins/(maxz-minz))
end

# NOTE: this bins on (radial-ish, axial) exactly like the 2D file, using x
# as the "axial" coordinate and sqrt(y^2+z^2) as the "radial" coordinate,
# purely as a 2D SUMMARY projection for diagnostics (StatsArray was never
# the particle's actual physics, just a binned reporting aid). For a
# genuinely off-axis geometry this projection is a simplification -- it's
# fine for a quick x-position vs. "distance off the x-axis" profile, but
# don't read y/z asymmetry out of it. If per-region 3D binning is needed
# later, that's a real change to make here, not implied by anything in
# this file today.
@inline function updateStats!(s::StatsArray, x, v, t, ncolls, lfree)
    r = sqrt(x[2]^2+x[3]^2)
    ridx = min(s.rbins, 1+floor(Int, s.rstep*(r-s.minr)))
    zidx = min(s.zbins, 1+floor(Int, s.zstep*(x[1]-s.minz)))
    fit!(s.stats[ridx, zidx].v, [(-x[3]*v[2]+x[2]*v[3])/sqrt(x[2]^2+x[3]^2), v[1]])
    fit!(s.stats[ridx, zidx].t, t)
    fit!(s.stats[ridx, zidx].ncolls, ncolls)
    fit!(s.stats[ridx, zidx].lfree, lfree)
    return s.stats[ridx, zidx]
end

@inline function merge!(a::StatsArray, b::StatsArray)
    for i in 1:a.rbins
        for j in 1:a.zbins
            OnlineStats.merge!(a.stats[i,j].v, b.stats[i,j].v)
            OnlineStats.merge!(a.stats[i,j].t, b.stats[i,j].t)
            OnlineStats.merge!(a.stats[i,j].ncolls, b.stats[i,j].ncolls)
            OnlineStats.merge!(a.stats[i,j].lfree, b.stats[i,j].lfree)
        end
    end
    return a
end

@inline function convert(::Type{Matrix}, s::StatsArray)
    M = Array{Float64}(undef, s.rbins*s.zbins, 14)
    idx = 1
    for i in 1:s.rbins
        for j in 1:s.zbins
            stats = s.stats[i,j]
            M[idx,1] = s.minr+(i-0.5)/s.rstep
            M[idx,2] = s.minz+(j-0.5)/s.zstep
            M[idx,3] = stats.t.n
            M[idx,4] = stats.t.μ
            M[idx,5] = stats.t.σ2
            M[idx,6] = stats.v.b[1]
            M[idx,7] = stats.v.b[2]
            M[idx,8] = stats.v.A[1,1]
            M[idx,9] = stats.v.A[2,2]
            M[idx,10] = stats.v.A[1,2]
            M[idx,11] = stats.ncolls.μ
            M[idx,12] = stats.ncolls.σ2
            M[idx,13] = stats.lfree.μ
            M[idx,14] = stats.lfree.σ2
            idx += 1
        end
    end
    return M
end

@inline function convert(::Type{DataFrame}, s::StatsArray)
    m = convert(Matrix, s)
    df = DataFrame(m, [:r, :z, :n, :t, :tvar, :vr, :vz, :vrvar, :vzvar, :vrvzcov, :ncolls, :ncollsvar, :lfree, :lfreevar])
    return df
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "geom"
            help = "the 3D triangulated SPARTA surf file (Points/Triangles), e.g. cell_fluor3d.surf"
            required = true
        "flow"
            help = "the SPARTA 3D grid dump (id xc yc zc xlo ylo zlo xhi yhi zhi nrho u v w temp)"
            required = true
        "-n"
            help = "the number of particles to simulate."
            arg_type = Int
            default = 10000
        "-x"
            help = "x position at which to spawn particles (m)"
            arg_type = Float64
            default = 0.03048
        "-y"
            help = "y position at which to spawn particles (m)"
            arg_type = Float64
            default = 0.01905
        "-z"
            help = "z position at which to spawn particles (m)"
            arg_type = Float64
            default = 0.01905
        "--vx"
            help = "mean x velocity at which to spawn particles (m/s)"
            arg_type = Float64
            default = 24843.71
        "--vy"
            help = "mean y velocity at which to spawn particles (m/s)"
            arg_type = Float64
            default = 0.0
        "--vz"
            help = "mean z velocity at which to spawn particles (m/s)"
            arg_type = Float64
            default = 0.0
        "-T"
            help = "thermal spread to add on top of the mean spawn velocity (K); 0 = a perfectly monoenergetic, mono-directional beam. This is a modeling simplification (see handoff doc) -- set a nonzero value if the real source's energy/angular spread is known."
            arg_type = Float64
            default = 0.0
        "-m"
            help = "mass of buffer gas atoms (AMU)"
            arg_type = Float64
            default = 20.1797
        "-M"
            help = "mass of traced particles (AMU)"
            arg_type = Float64
            default = 156.325
        "--sigma"
            help = "collision cross section between buffer gas and particle (m^2)"
            arg_type = Float64
            default = 5.0e-19
        "--saveall"
            help = "saves all particles if nonzero or just those that leave the cell if zero"
            arg_type = Int
            default = 0
        "--stats"
            help = "filename to save average properties of trajectories"
            arg_type = String
            default = nothing
        "--exitstats"
            help = "filename to save average properties of trajectories where the particle hits the simulation boundary"
            arg_type = String
            default = nothing
        "--spawnout"
            help = "filename to save every particle's spawn position as idx,x,y,z"
            arg_type = String
            default = nothing
        "--seed"
            help = "RNG seed; 0 leaves the RNG unseeded. Threaded runs reproduce only at a fixed thread count."
            arg_type = Int
            default = 0
        "--spawn"
            help = "initial position mode: point, gaussball, or uniformball (all centred at -x/-y/-z)"
            arg_type = String
            default = "point"
        "--spawnsize"
            help = "sigma of the gaussball mode, or radius of the uniformball mode (m)"
            arg_type = Float64
            default = 0.0
        "--trajprint"
            help = "if nonzero N, print a row for EVERY leg of particles 1..N (negative idx in column 1). Cost is ~1 row per collision per traced particle, so keep N small (~12)."
            arg_type = Int
            default = 0
        "--sampler"
            help = "collision partner sampler: exact (direct flux-weighted Maxwellian rejection; default) or table (legacy lookup-table rejection)"
            arg_type = String
            default = "exact"
        "--spawnclip"
            help = "if nonzero (default), reject spawn positions separated from the spawn reference point by geometry; resamples until valid"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

args = parse_commandline()
const MASS_PARTICLE = args["M"]
const MASS_BUFFER_GAS = args["m"]
const MASS_REDUCED = MASS_PARTICLE * MASS_BUFFER_GAS / (MASS_PARTICLE + MASS_BUFFER_GAS)
const kB = 8314.46
const σ_BUFFER_GAS_PARTICLE = args["sigma"]
const TRAJPRINT = args["trajprint"]
TRAJPRINT > 100 && @printf(stderr,
    "WARNING: --trajprint %d records all collision legs and can produce large output.\n", TRAJPRINT)
args["sampler"] in ("table", "exact") || error("--sampler must be table or exact")
const EXACT_SAMPLER = args["sampler"] == "exact"
const SPAWNCLIP = args["spawnclip"]
const SPAWN_REF = [args["x"], args["y"], args["z"]]

# ---- Rejection-sampling machinery for the buffer-gas thermal velocity:
# identical to ParticleTracing.jl, species-agnostic (only uses field T/u/v/w) ----

struct SampleParams
    μ_vg::Float64
    σ_vg::Float64
    σ_θ::Float64
end

struct LookupTable
    Tmin::Float64
    Tstep::Float64
    Tmax::Float64
    nT::Int64
    Umin::Float64
    Ustep::Float64
    Umax::Float64
    nU::Int64
    table::Matrix{SampleParams}
end

@inline function g(x, μ, σ)
    exp(-0.5*((x-μ)/σ)^2)/(σ*sqrt(2*π))
end

function sample(u, T, μ_vg, σ_vg, σ_θ, M=2.0)
    if T < 1E-2
        return (u, 0.0)
    end
    v_g = 0.0
    θ = 0.0
    bessel = 0.0
    i = 0
    imax = 50*M
    while true
        y = abs(μ_vg + σ_vg * Random.randn())
        bessel = SpecialFunctions.besseli(0, min(MASS_BUFFER_GAS*u*y/(kB*T), 10))
        f_y = exp(-MASS_BUFFER_GAS*(u^2+y^2)/(2*kB*T)) * y * bessel * MASS_BUFFER_GAS / (kB*T)
        r = f_y/(M*g(y, μ_vg, σ_vg))
        if Random.rand() < r
            v_g = y
            break
        end
        if i > imax
            println(stderr, "Maximum iterations exceeded in sampling v_g for u $u, T $T.")
            v_g = μ_vg
            break
        end
        i += 1
    end
    i = 0
    while true
        y = abs(σ_θ * Random.randn())
        f_y = exp(MASS_BUFFER_GAS * u * v_g * cos(y) / (kB*T)) / (pi * bessel)
        r = f_y/(2*M*g(y, 0, σ_θ))
        if Random.rand() < r && y < π
            θ = y
            break
        end
        if i > imax
            println(stderr, "Maximum iterations exceeded in sampling θ for u $u, T $T.")
            v_g = μ_vg
            break
        end
        i += 1
    end
    return v_g, θ
end

function sample(u, T, table, M=2.0)
    i_T = max(1,min(round(Int64, (T - table.Tmin) / table.Tstep), table.nT))
    i_U = max(1,min(round(Int64, (u - table.Umin) / table.Ustep), table.nU))
    p = table.table[i_T, i_U]
    return sample(max(table.Umin,min(u, table.Umax)), max(table.Tmin,min(T, table.Tmax)), p.μ_vg, 1.5*p.σ_vg, 3*p.σ_θ, M)
end

function generate_lookup_table(Tmin, Tstep, Tmax, Umin, Ustep, Umax, nsamples=100, Msample=20)
    Ts = Tmin:Tstep:Tmax
    Us = Umin:Ustep:Umax
    table = LookupTable(Tmin, Tstep, Tmax, length(Ts), Umin, Ustep, Umax, length(Us), Matrix{SampleParams}(undef, length(Ts), length(Us)))
    for (i, T) in enumerate(Ts)
        for (j, U) in enumerate(Us)
            σ_vg = 1.5*sqrt(8*kB*(T+0.2)/(π*MASS_BUFFER_GAS))
            σ_θ = 1.5*pi*σ_vg/(σ_vg+U)
            μ_vg = U + σ_vg
            vg_samples = zeros(nsamples)
            θ_samples = zeros(nsamples)
            for i in 1:nsamples
                vg_samples[i], θ_samples[i] = sample(U, T, μ_vg, σ_vg, σ_θ, Msample)
            end
            table.table[i,j] = SampleParams(mean(vg_samples), std(vg_samples), std(θ_samples))
        end
    end
    return table
end

# ---- Collision kinematics and free-path sampling: identical to
# ParticleTracing.jl. This is a general elastic two-body collision (exact
# for ANY mass ratio and ANY relative velocity, not a near-thermal
# approximation), and an exact flux-weighted rejection sampler for the
# collision partner's velocity -- both already correct for a fast (500 eV)
# particle slowing through a thermal gas. See docs/hpc-fluor-cell-3d-
# molecules-handoff.md for why this does NOT need new physics here. ----

@inline function exact_partner(w0, T)
    s = sqrt(kB*T/MASS_BUFFER_GAS)
    gmax = LinearAlgebra.norm(w0) + 8.0*s
    gmax == 0.0 && return [0.0, 0.0, 0.0]
    while true
        u = s .* Random.randn(3)
        g = sqrt((w0[1]-u[1])^2+(w0[2]-u[2])^2+(w0[3]-u[3])^2)
        if Random.rand()*gmax < g
            return u
        end
    end
end

@inline function collide!(v, vg, T, table)
    if EXACT_SAMPLER
        vg = vg .+ exact_partner(v .- vg, T)
    else
        u = sqrt((v[1]-vg[1])^2+(v[2]-vg[2])^2+(v[3]-vg[3])^2)
        vgmag, θ = sample(u, T, table)
        if u < 1E-3
            vgdir = LinearAlgebra.normalize(Random.rand(3) .- 0.5)
        else
            vgdir = (vg - v) ./ u
        end
        vrand = LinearAlgebra.normalize(Random.rand(3) .- 0.5)
        vperp = LinearAlgebra.normalize(vrand .- dot(vrand, vgdir) .* vgdir)
        vg = v .+ vgmag .* (cos(θ) .* vgdir .+ sin(θ) .* vperp)
    end
    cosχ = 2*Random.rand() - 1
    sinχ = sqrt(1 - cosχ^2)
    θ = 2 * π * Random.rand()
    g = sqrt((v[1] - vg[1])^2 + (v[2] - vg[2])^2 + (v[3] - vg[3])^2)
    v[1] = MASS_PARTICLE * v[1] + MASS_BUFFER_GAS * (vg[1] + g * cosχ)
    v[2] = MASS_PARTICLE * v[2] + MASS_BUFFER_GAS * (vg[2] + g * sinχ * cos(θ))
    v[3] = MASS_PARTICLE * v[3] + MASS_BUFFER_GAS * (vg[3] + g * sinχ * sin(θ))
    v .= v ./ (MASS_PARTICLE + MASS_BUFFER_GAS)
end

@inline function freePropagate!(xnext, x, v, t)
    xnext[1] = x[1] + v[1]*t
    xnext[2] = x[2] + v[2]*t
    xnext[3] = x[3] + v[3]*t
end

@inline function freePath(v, vrel, T, ρ)
    λ = sqrt(v[1]^2 + v[2]^2 + v[3]^2)/(ρ*σ_BUFFER_GAS_PARTICLE*sqrt(8*kB*T/(MASS_BUFFER_GAS*pi) + vrel^2))
    return min(-log(Random.rand()) * λ, 1000.0)
end

# ---- 3D geometry: Moller-Trumbore segment-vs-triangle intersection ----

@inline function cross3(a, b)
    (a[2]*b[3]-a[3]*b[2], a[3]*b[1]-a[1]*b[3], a[1]*b[2]-a[2]*b[1])
end
@inline function sub3(a, b)
    (a[1]-b[1], a[2]-b[2], a[3]-b[3])
end
@inline function dot3(a, b)
    a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
end

"""
    segment_hits_triangle(x1, x2, v0, v1, v2)

True if the line SEGMENT from x1 to x2 crosses the triangle (v0,v1,v2).
Standard Moller-Trumbore ray-triangle test, with t (the intersection's
fractional position along x1->x2) restricted to [0,1] to make it a
segment test rather than an infinite-ray test.
"""
@inline function segment_hits_triangle(x1, x2, v0, v1, v2)
    EPS = 1e-12
    edge1 = sub3(v1, v0)
    edge2 = sub3(v2, v0)
    dir = sub3(x2, x1)
    h = cross3(dir, edge2)
    a = dot3(edge1, h)
    if abs(a) < EPS
        return false
    end
    f = 1.0 / a
    s = sub3(x1, v0)
    u = f * dot3(s, h)
    if u < 0.0 || u > 1.0
        return false
    end
    q = cross3(s, edge1)
    v = f * dot3(dir, q)
    if v < 0.0 || u + v > 1.0
        return false
    end
    t = f * dot3(edge2, q)
    return 0.0 <= t <= 1.0
end

# ---- Spatial acceleration for getCollision(): a real production geometry
# has thousands of triangles and a particle can take tens of thousands of
# (typically very short, sub-mm) free-flight steps to thermalize -- a
# naive per-step scan of every triangle does not scale. This buckets
# triangles into a uniform 3D grid once at startup; each free-flight
# segment then only tests triangles whose bucket it (plus a 1-cell margin)
# overlaps, not the whole mesh. ----

struct TriangleGrid
    xlo::Float64
    ylo::Float64
    zlo::Float64
    cellsize::Float64
    nx::Int
    ny::Int
    nz::Int
    cells::Dict{NTuple{3,Int},Vector{Int}}
end

function build_triangle_grid(points, triangles; target_per_cell=8)
    n = length(triangles)
    xlo=Inf; ylo=Inf; zlo=Inf; xhi=-Inf; yhi=-Inf; zhi=-Inf
    for p in points
        xlo = min(xlo, p[1]); xhi = max(xhi, p[1])
        ylo = min(ylo, p[2]); yhi = max(yhi, p[2])
        zlo = min(zlo, p[3]); zhi = max(zhi, p[3])
    end
    vol = max(xhi-xlo, 1e-9) * max(yhi-ylo, 1e-9) * max(zhi-zlo, 1e-9)
    cellsize = max(cbrt(vol * target_per_cell / max(n, 1)), 1e-6)
    nx = max(1, ceil(Int, (xhi-xlo)/cellsize))
    ny = max(1, ceil(Int, (yhi-ylo)/cellsize))
    nz = max(1, ceil(Int, (zhi-zlo)/cellsize))
    cells = Dict{NTuple{3,Int},Vector{Int}}()
    for (ti, tri) in enumerate(triangles)
        v0, v1, v2 = points[tri[1]], points[tri[2]], points[tri[3]]
        txlo=min(v0[1],v1[1],v2[1]); txhi=max(v0[1],v1[1],v2[1])
        tylo=min(v0[2],v1[2],v2[2]); tyhi=max(v0[2],v1[2],v2[2])
        tzlo=min(v0[3],v1[3],v2[3]); tzhi=max(v0[3],v1[3],v2[3])
        ixlo=clamp(floor(Int,(txlo-xlo)/cellsize),0,nx-1); ixhi=clamp(floor(Int,(txhi-xlo)/cellsize),0,nx-1)
        iylo=clamp(floor(Int,(tylo-ylo)/cellsize),0,ny-1); iyhi=clamp(floor(Int,(tyhi-ylo)/cellsize),0,ny-1)
        izlo=clamp(floor(Int,(tzlo-zlo)/cellsize),0,nz-1); izhi=clamp(floor(Int,(tzhi-zlo)/cellsize),0,nz-1)
        for ix in ixlo:ixhi, iy in iylo:iyhi, iz in izlo:izhi
            key = (ix,iy,iz)
            if haskey(cells, key)
                push!(cells[key], ti)
            else
                cells[key] = [ti]
            end
        end
    end
    return TriangleGrid(xlo, ylo, zlo, cellsize, nx, ny, nz, cells)
end

@inline function candidate_triangles(grid::TriangleGrid, x1, x2)
    out = Int[]
    sxlo=min(x1[1],x2[1]); sxhi=max(x1[1],x2[1])
    sylo=min(x1[2],x2[2]); syhi=max(x1[2],x2[2])
    szlo=min(x1[3],x2[3]); szhi=max(x1[3],x2[3])
    ixlo=clamp(floor(Int,(sxlo-grid.xlo)/grid.cellsize)-1,0,grid.nx-1)
    ixhi=clamp(floor(Int,(sxhi-grid.xlo)/grid.cellsize)+1,0,grid.nx-1)
    iylo=clamp(floor(Int,(sylo-grid.ylo)/grid.cellsize)-1,0,grid.ny-1)
    iyhi=clamp(floor(Int,(syhi-grid.ylo)/grid.cellsize)+1,0,grid.ny-1)
    izlo=clamp(floor(Int,(szlo-grid.zlo)/grid.cellsize)-1,0,grid.nz-1)
    izhi=clamp(floor(Int,(szhi-grid.zlo)/grid.cellsize)+1,0,grid.nz-1)
    for ix in ixlo:ixhi, iy in iylo:iyhi, iz in izlo:izhi
        key = (ix,iy,iz)
        if haskey(grid.cells, key)
            for ti in grid.cells[key]
                ti in out || push!(out, ti)
            end
        end
    end
    return out
end

"""
    read_surf3d(path)

Parses a SPARTA 3D triangulated surf file (Points/Triangles sections, as
written by cases/fluor-cell-3d/gen_fluor3d.py). Returns (points, triangles)
where points[i] is a (x,y,z) tuple (1-based, matching the file's point
indices) and triangles is a list of (p1,p2,p3) point-index tuples.
"""
function read_surf3d(path)
    lines = readlines(path)
    i = 1
    while !endswith(strip(lines[i]), "points")
        i += 1
    end
    n_points = parse(Int, split(lines[i])[1]); i += 1
    n_tris = parse(Int, split(lines[i])[1]); i += 1
    while strip(lines[i]) != "Points"
        i += 1
    end
    i += 1
    while strip(lines[i]) == ""
        i += 1
    end
    points = Vector{NTuple{3,Float64}}(undef, n_points)
    for _ in 1:n_points
        parts = split(lines[i])
        idx = parse(Int, parts[1])
        points[idx] = (parse(Float64, parts[2]), parse(Float64, parts[3]), parse(Float64, parts[4]))
        i += 1
    end
    while strip(lines[i]) != "Triangles"
        i += 1
    end
    i += 1
    while strip(lines[i]) == ""
        i += 1
    end
    triangles = Vector{NTuple{3,Int}}(undef, n_tris)
    for k in 1:n_tris
        parts = split(lines[i])
        triangles[k] = (parse(Int, parts[3]), parse(Int, parts[4]), parse(Int, parts[5]))
        i += 1
    end
    return points, triangles
end

interps = 0

function build_field(geomFile, gridFile)
    points, triangles = read_surf3d(geomFile)
    tri_grid = build_triangle_grid(points, triangles)

    # 3D SPARTA grid dump: id xc yc zc xlo ylo zlo xhi yhi zhi nrho u v w temp
    # (matches `dump gd grid all ... id xc yc zc xlo ylo zlo xhi yhi zhi f_ag[*]`
    # in in.ne_fluor3d_scoping, where f_ag[*] = [nrho, u, v, w, temp]).
    griddf = CSV.read(gridFile, DataFrame;
        header=[:id,:xc,:yc,:zc,:xlo,:ylo,:zlo,:xhi,:yhi,:zhi,:nrho,:u,:v,:w,:T],
        skipto=10, ignorerepeated=true, delim=' ')
    griddf = griddf[griddf.T .> 0, :]

    xlo = minimum(griddf.xlo); xhi = maximum(griddf.xhi)
    ylo = minimum(griddf.ylo); yhi = maximum(griddf.yhi)
    zlo = minimum(griddf.zlo); zhi = maximum(griddf.zhi)
    max_x_geom = xhi  # kept for output-array-shape parity with the 2D file; unused downstream here

    DataFrames.select!(griddf, [:xc, :yc, :zc, :u, :v, :w, :T, :nrho])
    grids = Matrix(griddf)

    kdtree = KDTree(transpose(Matrix(grids[:,1:3])); leafsize=10)

    Tmin = minimum(griddf.T)
    Tmax = maximum(griddf.T)
    Umin = 0.0
    Umax = 1.5*maximum(sqrt.(griddf.u.^2 .+ griddf.v.^2 .+ griddf.w.^2))
    # A field region with near-uniform temperature or bulk velocity (very
    # plausible in parts of a mostly-thermalized cell) gives a zero-width
    # range here, and Tmin:0:Tmax / Umin:0:Umax throws ArgumentError:
    # range step cannot be zero. Found via a smoke test with intentionally
    # uniform synthetic data; the same crash is reachable with real data.
    # A tiny positive floor collapses the lookup table to (correctly) a
    # single entry for that near-uniform region instead of crashing.
    Tstep = max((Tmax - Tmin) / 20, 1e-6)
    Ustep = max((Umax - Umin) / 20, 1e-6)
    table = generate_lookup_table(Tmin, Tstep, Tmax, Umin, Ustep, Umax)

    """
        interpolate!(props, x)

    Nearest-cell-centre lookup in true 3D (x,y,z) -- no axisymmetric
    rotation assumption, unlike the 2D tracer, because this geometry
    (off-axis gas inlet) is genuinely not axisymmetric.
    """
    @inline function interpolate!(props, x)
        global interps
        interps += 1
        idx = knn(kdtree, [x[1], x[2], x[3]], 1)[1][1]
        interp = view(grids, idx, :)
        props[1] = interp[1]  # xc (unused downstream, kept for layout parity)
        props[2] = interp[2]  # yc
        props[3] = interp[4]  # vgx = u
        props[4] = interp[5]  # vgy = v
        props[5] = interp[6]  # vgz = w
        props[6] = interp[7]  # T
        props[7] = interp[8]  # rho (number density)
        props[8] = 0.0
    end

    """
        getCollision(x1, x2)

    True 3D segment-vs-triangle-mesh test, using the spatial grid to test
    only nearby triangles rather than the whole mesh. Returns 0 (no
    collision), 1 (hit the wall mesh), or 2 (left the simulation box
    bounds -- read from the grid dump's own cell extents, not hardcoded,
    so this always matches whatever create_box the SPARTA deck actually
    used). Allocates a small fresh candidate list per call rather than a
    thread-indexed scratch buffer -- Threads.threadid() is not guaranteed
    to stay within 1:Threads.nthreads() on every Julia version (newer
    versions have a separate "interactive" thread pool that threadid()
    can return IDs from, while nthreads() by default counts only the
    default pool -- hit exactly this as a BoundsError in local testing on
    1.13). Correctness across whatever Julia version actually runs this
    matters more than avoiding one small per-call allocation.
    """
    @inline function getCollision(x1, x2)
        for ti in candidate_triangles(tri_grid, x1, x2)
            p1, p2, p3 = triangles[ti]
            if segment_hits_triangle(x1, x2, points[p1], points[p2], points[p3])
                return 1
            end
        end
        if x2[1] < xlo || x2[1] > xhi || x2[2] < ylo || x2[2] > yhi || x2[3] < zlo || x2[3] > zhi
            return 2
        end
        return 0
    end

    return (; interpolate!, getCollision, table, max_x_geom)
end

"""
    propagate(xinit, vin, interp!, getCollision, table, stats=nothing, idx=0)

Identical in structure to ParticleTracing.jl's propagate(), minus the
harmonic-trap (omega) and spin-flip (pflip) parameters -- there is no
ion-guide field modeled here (see the handoff doc), so there is nothing
for a spin flip to be relative to either. Free flight is a straight line;
propagation uses freePropagate! (plain x += v*t), not the trap-aware
version.
"""
@inline function propagate(xinit, vin, interp!, getCollision, table, stats=nothing, idx=0)
    x = deepcopy(xinit)
    props = zeros(8)
    interp!(props, x)
    xnext = deepcopy(x)
    v = deepcopy(vin)
    time = 0.0
    collides = 0

    if LinearAlgebra.norm(v) < 1E-6
        collide!(v, view(props, 3:5), props[6], table)
        collides += 1
    end

    while true
        interp!(props, x)
        vrel = sqrt((v[1] - props[3])^2 + (v[2] - props[4])^2 + (v[3] - props[5])^2)
        dist = freePath(v, vrel, props[6], props[7])
        vmag = LinearAlgebra.norm(v)
        if vmag < 1E-6
            xnext .= x
        else
            freePropagate!(xnext, x, v, dist/vmag)
        end
        if getCollision(x, xnext) != 0
            return (x[1], x[2], x[3], xnext[1], xnext[2], xnext[3], v[1], v[2], v[3], collides, time)
        else
            time += dist / max(vmag, 1e-30)
            collides += 1
            if TRAJPRINT != 0 && idx <= TRAJPRINT
                print(@sprintf("%d %e %e %e %e %e %e %e %e %e %d %e\n", -idx,
                x[1], x[2], x[3], xnext[1], xnext[2], xnext[3], v[1], v[2], v[3], collides, time))
            end
        end
        if !isnothing(stats)
            updateStats!(stats, x, v, time, collides, dist)
        end
        x .= xnext
        collide!(v, view(props, 3:5), props[6], table)
    end
end

function SimulateParticles(
    geomFile,
    gridFile,
    nParticles,
    generateParticle,
    print_stuff=true,
    saveall=0,
    savestats=nothing,
    saveexitstats=nothing,
    rbins=100,
    zbins=100;
    savespawns=nothing,
    make_stats=nothing
    )

    (; interpolate!, getCollision, table, max_x_geom) = build_field(geomFile, gridFile)

    # Stats binning bounds: x (axial-ish) from the field's own extent;
    # "radial" (sqrt(y^2+z^2)) bounds derived generously from the field's
    # y/z spread. See the StatsArray comment above re: this being a
    # summary projection, not the actual 3D physics.
    griddf_bounds = CSV.read(gridFile, DataFrame;
        header=[:id,:xc,:yc,:zc,:xlo,:ylo,:zlo,:xhi,:yhi,:zhi,:nrho,:u,:v,:w,:T],
        skipto=10, ignorerepeated=true, delim=' ')
    griddf_bounds = griddf_bounds[griddf_bounds.T .> 0, :]
    minz = minimum(griddf_bounds.xlo); maxz = maximum(griddf_bounds.xhi)
    maxr = maximum(sqrt.((griddf_bounds.yc .- mean(griddf_bounds.yc)).^2 .+
                          (griddf_bounds.zc .- mean(griddf_bounds.zc)).^2)) * 1.5 + 1e-6

    new_stats = isnothing(make_stats) ?
        (() -> StatsArray(0.0, maxr, rbins, minz, maxz, zbins)) : make_stats

    output_dim = length(propagate(zeros(3), zeros(3), interpolate!, (x,y)->true, table))
    outputs = zeros(nParticles, output_dim)

    if !isnothing(savestats)
        allstats = new_stats()
    end
    if !isnothing(saveexitstats)
        boundstats = new_stats()
    end

    chunks = collect(Iterators.partition(1:nParticles, cld(nParticles, Threads.nthreads())))
    part_all = isnothing(savestats) ? nothing : [new_stats() for _ in chunks]
    part_bound = isnothing(saveexitstats) ? nothing : [new_stats() for _ in chunks]

    spawn_rejects = Threads.Atomic{Int}(0)
    spawns = isnothing(savespawns) ? nothing : zeros(nParticles, 3)
    @sync for (c, chunk) in enumerate(chunks)
        Threads.@spawn for i in chunk
            stats = nothing
            if !isnothing(saveexitstats) || !isnothing(savestats)
                stats = new_stats()
            end
            xpart, vpart = generateParticle()
            if SPAWNCLIP != 0
                tries = 0
                while getCollision(SPAWN_REF, xpart) != 0
                    tries += 1
                    tries > 1000 && error("--spawnclip: 1000 consecutive spawn candidates were cut off from the spawn reference $(SPAWN_REF) by geometry; check -x/-y/-z and the .surf file")
                    xpart, vpart = generateParticle()
                end
                tries > 0 && Threads.atomic_add!(spawn_rejects, tries)
            end
            isnothing(spawns) || (spawns[i,:] .= xpart)
            outputs[i,:] .= propagate(xpart, vpart, interpolate!, getCollision, table, stats, i)
            colltype = getCollision(outputs[i,[1,2,3]], outputs[i,[4,5,6]])
            if !isnothing(savestats)
                merge!(part_all[c], stats)
            end
            if colltype == 2 && !isnothing(saveexitstats)
                merge!(part_bound[c], stats)
            end
            if print_stuff && (saveall != 0 || colltype == 2)
                print(@sprintf("%d %e %e %e %e %e %e %e %e %e %d %e\n", i,
                outputs[i,1], outputs[i,2], outputs[i,3], outputs[i,4], outputs[i,5], outputs[i,6], outputs[i,7], outputs[i,8], outputs[i,9], outputs[i,10], outputs[i,11]))
            end
        end
    end

    for c in eachindex(chunks)
        isnothing(part_all) || merge!(allstats, part_all[c])
        isnothing(part_bound) || merge!(boundstats, part_bound[c])
    end
    if !isnothing(savespawns)
        CSV.write(savespawns, DataFrame(idx = 1:nParticles, x = spawns[:,1],
                                        y = spawns[:,2], z = spawns[:,3]))
    end
    if SPAWNCLIP != 0 && spawn_rejects[] > 0
        @printf(stderr, "Spawn candidates rejected by --spawnclip: %d (%.2f%% of %d draws)\n",
            spawn_rejects[], 100*spawn_rejects[]/(spawn_rejects[] + nParticles), spawn_rejects[] + nParticles)
    end

    return outputs, boundstats, allstats
end

function main(args)
    nParticles = args["n"]

    if args["seed"] != 0
        Random.seed!(args["seed"])
    end

    println("idx x y z xnext ynext znext vx vy vz collides time")

    thermal_spread = sqrt(kB*args["T"]/MASS_PARTICLE)
    spawnmode = args["spawn"]
    spawnmode in ("point", "gaussball", "uniformball") ||
        error("unknown --spawn mode: $spawnmode")
    spawnPosition() =
        if spawnmode == "point"
            [args["x"], args["y"], args["z"]]
        elseif spawnmode == "gaussball"
            s = args["spawnsize"]
            [args["x"] + s*Random.randn(), args["y"] + s*Random.randn(), args["z"] + s*Random.randn()]
        else # uniformball
            u = Random.randn(3)
            u .*= args["spawnsize"] * cbrt(Random.rand()) / LinearAlgebra.norm(u)
            [args["x"] + u[1], args["y"] + u[2], args["z"] + u[3]]
        end
    generateParticle() = (
        spawnPosition(),
        [args["vx"] + Random.randn() * thermal_spread,
         args["vy"] + Random.randn() * thermal_spread,
         args["vz"] + Random.randn() * thermal_spread])

    nthreads = Threads.nthreads()
    @printf(stderr, "Threads: %d\n", nthreads)
    start = time()
    outputs, boundstats, allstats = SimulateParticles(
        args["geom"],
        args["flow"],
        nParticles,
        generateParticle,
        true,
        args["saveall"],
        !isnothing(args["stats"]),
        !isnothing(args["exitstats"]);
        savespawns = args["spawnout"])
    runtime = time() - start
    if !isnothing(args["stats"])
        CSV.write(args["stats"], convert(DataFrame, allstats))
    end
    if !isnothing(args["exitstats"])
        CSV.write(args["exitstats"], convert(DataFrame, boundstats))
    end

    @printf(stderr, "Time: %.3e\n",runtime)
    @printf(stderr, "Time per particle: %.3e\n", runtime/nParticles)
    @printf(stderr, "Time per collision: %.3e\n", runtime/sum(outputs[:,10]))
    @printf(stderr, "Interpolates: %.3e\n",interps)
    @printf(stderr, "Collides: %.3e\n",sum(outputs[:,10]))

    return allstats
end

allstats = main(args)
