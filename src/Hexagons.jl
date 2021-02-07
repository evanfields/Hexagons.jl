module Hexagons

import Base: convert, length, collect, iterate

export CoordAxial, CoordCubic,
       HexFlatTop, HexPointyTop,
       to_coord_form,
       center, vertices,
       hex_containing


# Various ways to index hexagons in a grid
# ----------------------------------------

## 
# Hex coordinates:
# Each hexagon has coordinates in the hexagonal grid system.
# Conversions between hex coordinates do not depend on the
# mapping between hexagon space and Cartesian space.
##

"""Abstract hexagon coordinates. A subtype of `AbstractCoord` locates a hexagon
within a grid of hexagons, but does not contain enough information to map a hexagon
to non-hexagonal (e.g. Cartesian) space. For example, `CoordAxial(0, 0)` can be thought
of as the center of a grid of hexagons. But without knowing hexagon orientation (see
`Hexagon`), the Cartesian `(x,y)` coordinates of the vertices of the hexagon at
`CoordAxial(0, 0)` are not defined.

See https://www.redblobgames.com/grids/hexagons/#coordinates for some details."""
abstract type AbstractCoord end

struct CoordAxial{T <: Real} <: AbstractCoord
    q::T
    r::T
end
# semantic based equality and hash
Base.:(==)(c1::CoordAxial, c2::CoordAxial) = c1.q == c2.q && c1.r == c2.r
Base.hash(c::CoordAxial, h::UInt64) = hash(c.q, hash(c.r, hash(:CoordAxial, h)))

"""Cubic hexagon coordinates. Components must sum to zero.

See [`AbstractCoord`](@ref) for more information on hexagon coordinates."""
struct CoordCubic{T <: Real} <: AbstractCoord
    x::T
    y::T
    z::T
    
    function CoordCubic(x, y, z)
        T = promote_type(typeof(x), typeof(y), typeof(z))
        maxabs = T <: Integer ? 0 : sqrt(eps(T))
        if abs(x + y + z) > maxabs
            throw(DomainError(
                (x,y,z),
                "Components of cubic hexagon coordinates must sum to 0."
            ))
        end
        return new{T}(x, y, z)
    end
end
# use axial form for == and hash for non axial types
Base.:(==)(c1::AbstractCoord, c2::AbstractCoord) = CoordAxial(c1) == CoordAxial(c2)
Base.hash(c::AbstractCoord, h::UInt64) = hash(CoordAxial(c), h)

const _COORD_TYPES = (CoordAxial, CoordCubic)

# Convert between hexagon coordinate systems

function convert(::Type{CoordAxial}, coord::CoordCubic)
    CoordAxial(coord.x, coord.z)
end
CoordAxial(c::AbstractCoord) = convert(CoordAxial, c)

function convert(::Type{CoordCubic}, coord::CoordAxial)
    CoordCubic(coord.q, -coord.q - coord.r, coord.r)
end
CoordCubic(c::AbstractCoord) = convert(CoordCubic, c)

##
# Hexagons:
# A hexagon has an orientation (either flat or pointy top, encoded by the type)
# and coordinates in hex space. 
# Orientation plus hex coordinates (plus an implicit unit-edge-length assumption) allows
# for mapping between hexagon space and Cartesian space.
# to-do: support variable size?
##

abstract type Hexagon end

"""
    HexPointyTop{T <: AbstractCoord} <: Hexagon
A hexagon oriented so that in the Cartesian plane, the hexagon's top and bottom
are points, and left/right sides are parallel to the y-axis.
"""
struct HexPointyTop{T <: AbstractCoord} <: Hexagon
    coords::T
end
# Hexagon equality doesn't depend on internal coordinate representation,
# just the actual hexagon represented.
function Base.:(==)(hex1::HexPointyTop, hex2::HexPointyTop)
    return CoordAxial(hex1.coords) == CoordAxial(hex2.coords)
end
Base.hash(hex::HexPointyTop, h::UInt64) = hash(hex.coords, hash(:HexPointyTop, h))

"""
    HexFlatTop{T <: AbstractCoord} <: Hexagon
A hexagon oriented so that in the Cartesian plane, the hexagon's top and bottom
are parallel to the x-axis.
"""
struct HexFlatTop{T <: AbstractCoord} <: Hexagon
    coords::T
end
function Base.:(==)(hex1::HexFlatTop, hex2::HexFlatTop)
    return CoordAxial(hex1.coords) == CoordAxial(hex2.coords)
end
Base.hash(hex::HexFlatTop, h::UInt64) = hash(hex.coords, hash(:HexFlatTop, h))

# Convenience constructors without manual instantiation of coord objects
"""
    HexFlatTop(q,r)
A flat top hexagon with axial hexagon coordinates `(q,r)`.
"""
HexFlatTop(q, r) = HexFlatTop(CoordAxial(q, r))
"""
    HexFlatTop(x,y,z)
A flat top hexagon with cubic coordinates `(x,y,z)`.
"""
HexFlatTop(x, y, z) = HexFlatTop(CoordCubic(x, y, z))
"""
    HexPointyTop(q,r)
A pointy top hexagon with axial hexagon coordinates `(q,r)`.
"""
HexPointyTop(q, r) = HexPointyTop(CoordAxial(q, r))
"""
    HexPointyTop(x,y,z)
A pointy top hexagon with cubic coordinates `(x,y,z)`.
"""
HexPointyTop(x, y, z) = HexPointyTop(CoordCubic(x, y, z))

##
# Hexagon coordinate Conversions
##

"""
    to_coord_form(coord_type, hex)

Return a hexagon equal to `hex` with internal coordinate representation
of type `coord_type`.
"""
function to_coord_form end
for hextype in (HexPointyTop, HexFlatTop), coordtype in _COORD_TYPES
    @eval begin 
        function to_coord_form(::Type{$coordtype}, h::$hextype)
            return ($hextype)(convert($coordtype, h.coords))
        end
    end
end

##
# Hexagon to Cartesian mappings
##

"""
    center(hex::Hexagon)

Return the Cartesian coordiantes of a hexagon's center as a tuple (x,y).
"""
function center(hex::HexPointyTop)
    c_ax = CoordAxial(hex.coords)
    return (
        sqrt(3) * (c_ax.q + c_ax.r / 2),
        3/2 * c_ax.r
    )
end
function center(hex::HexFlatTop)
    c_ax = CoordAxial(hex.coords)
    return (
        3/2 * c_ax.q,
        sqrt(3) * (c_ax.q / 2 + c_ax.r)
    )
end

"""
    vertices(hex::Hexagon)

Return the Cartesian coordinates of the vertices of `hex` as a list of `(x,y)` tuples.
The returned vertices are ordered counter-clockwise from the hexagon center, starting
from the positive x-axis.
"""
function vertices(hex::HexFlatTop)
    thetas = (i * pi / 6 for i in 0:2:10)
    center_x, center_y = center(hex)
    return [(center_x + cos(theta), center_y + sin(theta)) for theta in thetas]
end
function vertices(hex::HexPointyTop)
    thetas = (i * pi / 6 for i in 1:2:11)
    center_x, center_y = center(hex)
    return [(center_x + cos(theta), center_y + sin(theta)) for theta in thetas]
end

##
# Cartesian to hexagon
##

"""
    _nearest_integer_coord(c::AbstractCoord)

Find the integer hex coordinates closest to `c`.
"""
_nearest_integer_coord(c::CoordCubic{T} where {T <: Integer}) = c
function _nearest_integer_coord(c::CoordCubic{T}) where {T}
    rx, ry, rz = round(Int, c.x), round(Int, c.y), round(Int, c.z)
    x_diff, y_diff, z_diff = abs(rx - c.x), abs(ry - c.y), abs(rz - c.z)

    if x_diff > y_diff && x_diff > z_diff
        rx = -ry - rz
    elseif y_diff > z_diff
        ry = -rx - rz
    else
        rz = -rx - ry
    end

    CoordCubic(rx, ry, rz)
end
_nearest_integer_coord(coord::CoordAxial) = coord |> CoordCubic |> _nearest_integer_coord |> CoordAxial


"""
    hex_containing(HexFlatTop, x, y)

Return the HexFlatTop containing Cartesian point `(x,y)`.
"""
function hex_containing(::Type{HexFlatTop}, x, y)
    q = 2 / 3 * x
    r = -x / 3 + sqrt(3) / 3 * y
    return HexFlatTop(_nearest_integer_coord(CoordAxial(q,r)))
end

"""
    hex_containing(HexPointyTop, x, y)

Return the HexPointyTop containing Cartesian point `(x,y)`.
"""
function hex_containing(::Type{HexPointyTop}, x, y)
    q = sqrt(3) / 3 * x - y / 3
    r = 2 / 3 * y
    return HexPointyTop(_nearest_integer_coord(CoordAxial(q,r)))
end


#=
# Neighbor hexagon iterator
# -------------------------

struct HexagonNeighborIterator
    hex::HexagonCubic
end

const CUBIC_HEX_NEIGHBOR_OFFSETS = [
     1 -1  0;
     1  0 -1;
     0  1 -1;
    -1  1  0;
    -1  0  1;
     0 -1  1;
]

neighbors(hex::Hexagon) = HexagonNeighborIterator(convert(HexagonCubic, hex))

length(::HexagonNeighborIterator) = 6

function iterate(it::HexagonNeighborIterator, state=1)
    state > 6 && return nothing
    dx = CUBIC_HEX_NEIGHBOR_OFFSETS[state, 1]
    dy = CUBIC_HEX_NEIGHBOR_OFFSETS[state, 2]
    dz = CUBIC_HEX_NEIGHBOR_OFFSETS[state, 3]
    neighbor = HexagonCubic(it.hex.x + dx, it.hex.y + dy, it.hex.z + dz)
    return (neighbor, state + 1)
end


# Diagonal hexagon iterator
# -------------------------

struct HexagonDiagonalIterator
    hex::HexagonCubic
end

const CUBIC_HEX_DIAGONAL_OFFSETS = [
    +2 -1 -1;
    +1 +1 -2;
    -1 +2 -1;
    -2 +1 +1;
    -1 -1 +2;
    +1 -2 +1;
]

diagonals(hex::Hexagon) = HexagonDiagonalIterator(convert(HexagonCubic, hex))

length(::HexagonDiagonalIterator) = 6

function iterate(it::HexagonDiagonalIterator, state=1)
    state > 6 && return nothing
    dx = CUBIC_HEX_DIAGONAL_OFFSETS[state, 1]
    dy = CUBIC_HEX_DIAGONAL_OFFSETS[state, 2]
    dz = CUBIC_HEX_DIAGONAL_OFFSETS[state, 3]
    diagonal = HexagonCubic(it.hex.x + dx, it.hex.y + dy, it.hex.z + dz)
    return (diagonal, state + 1)
end


# Iterator over the vertices of a hexagon
# ---------------------------------------

struct HexagonVertexIterator
    x_center::Float64
    y_center::Float64
    xsize::Float64
    ysize::Float64

    function HexagonVertexIterator(x, y, xsize=1.0, ysize=1.0)
        new((Float64(x)), (Float64(y)),
            (Float64(xsize)), (Float64(ysize)))
    end

    function HexagonVertexIterator(hex::Hexagon,
                                   xsize=1.0, ysize=1.0, xoff=0.0, yoff=0.0)
        c = center(hex, xsize, ysize, xoff, yoff)
        new((Float64(c[1])), (Float64(c[2])),
            (Float64(xsize)), (Float64(ysize)))
    end
end

function vertices(hex::Hexagon, xsize=1.0, ysize=1.0, xoff=0.0, yoff=0.0)
    c = center(hex, xsize, ysize, xoff, yoff)
    HexagonVertexIterator(c[1], c[2], xsize, ysize)
end

# TODO: remove this function?
function hexpoints(x, y, xsize=1.0, ysize=1.0)
    collect(Tuple{Float64, Float64},
            HexagonVertexIterator(Float64(x), Float64(y),
                                  Float64(xsize), Float64(ysize)))
end

length(::HexagonVertexIterator) = 6

function iterate(it::HexagonVertexIterator, state=1)
    state > 6 && return nothing
    theta = 2*pi/6 * (state-1+0.5)
    x_i = it.x_center + it.xsize * cos(theta)
    y_i = it.y_center + it.ysize * sin(theta)
    return ((x_i, y_i), state + 1)
end

struct HexagonDistanceIterator
    hex::HexagonCubic
    n::Int
end

function hexagons_within(n::Int, hex::Hexagon = HexagonAxial(0, 0))
    cubic_hex = convert(HexagonCubic, hex)
    HexagonDistanceIterator(hex, n)
end
hexagons_within(hex::Hexagon, n::Int) = hexagons_within(n, hex)

length(it::HexagonDistanceIterator) = it.n * (it.n + 1) * 3 + 1

function iterate(it::HexagonDistanceIterator, state=(-it.n, 0))
    x, y = state
    x > it.n && return nothing
    z = -x-y
    hex = HexagonCubic(x, y, z)
    y += 1
    if y > min(it.n, it.n-x)
        x += 1
        y = max(-it.n, -it.n - x)
    end
    hex, (x, y)
end


collect(it::HexagonDistanceIterator) = collect(HexagonCubic, it)

# Iterator over a ring of hexagons
# ---------------------------------------

struct HexagonRingIterator
    hex::HexagonCubic
    n::Int
end

function ring(n::Int, hex::Hexagon = HexagonAxial(0, 0))
    # println("New hexring with center $hex and n $n")
    cubic_hex = convert(HexagonCubic, hex)
    HexagonRingIterator(cubic_hex, n)
end
ring(hex::Hexagon, n::Int) = ring(n, hex)

length(it::HexagonRingIterator) = it.n * 6

function iterate(it::HexagonRingIterator,
                 state::(Tuple{Int, HexagonCubic})=(1, neighbor(it.hex, 5, it.n)))
    hex_i, cur_hex = state
    hex_i > length(it) && return nothing
    # println("HexagonRingIterator: at position $hex_i ($cur_hex)")
    ring_part = div(hex_i - 1, it.n) + 1
    next_hex = neighbor(cur_hex, ring_part)
    cur_hex, (hex_i + 1, next_hex)
end

collect(it::HexagonRingIterator) = collect(HexagonCubic, it)

# Iterator over all hexes within a certain distance
# -------------------------------------------------

struct HexagonSpiralIterator
    hex::HexagonCubic
    n::Int
end

struct HexagonSpiralIteratorState
    hexring_i::Int
    hexring_it::HexagonRingIterator
    hexring_it_i::Int
    hexring_it_hex::HexagonCubic
end

function spiral(n::Int, hex::Hexagon = HexagonAxial(0, 0))
    cubic_hex = convert(HexagonCubic, hex)
    HexagonSpiralIterator(cubic_hex, n)
end
spiral(hex::Hexagon, n::Int) = spiral(n, hex)

length(it::HexagonSpiralIterator) = it.n * (it.n + 1) * 3

# The state of a HexagonSpiralIterator consists of
# 1. an Int, the index of the current ring
# 2. a HexagonRingIterator and its state to keep track of the current position
#    in the ring.

function iterate(it::HexagonSpiralIterator)
    first_ring = ring(it.hex, 1)
    iterate(it, HexagonSpiralIteratorState(1, first_ring, start(first_ring)...))
end

function iterate(it::HexagonSpiralIterator, state::HexagonSpiralIteratorState)
    state.hexring_i > it.n && return nothing
    # Get current state
    hexring_i, hexring_it, hexring_it_i, hexring_it_hex =
        state.hexring_i, state.hexring_it, state.hexring_it_i, state.hexring_it_hex
    # Update state of inner iterator
    hexring_it_hex, (hexring_it_i, hexring_it_hex_next) =
                next(hexring_it, (hexring_it_i, hexring_it_hex))
    # Check if inner iterator is done, and update if necessary
    if done(hexring_it, (hexring_it_i, hexring_it_hex_next))
        hexring_i += 1
        hexring_it = ring(it.hex, hexring_i)
        hexring_it_i, hexring_it_hex_next = start(hexring_it)
        # println("In new ring $hexring_it")
    end

    # println("Currently at $hexring_it_hex, hexring is $hexring_it, state is $((hexring_i, (hexring_it_i, hexring_it_hex)))")
    hexring_it_hex, HexagonSpiralIteratorState(hexring_i, hexring_it,
                                               hexring_it_i, hexring_it_hex_next)
end

collect(it::HexagonSpiralIterator) = collect(HexagonCubic, it)

# Utilities
# ---------

function distance(a::Hexagon, b::Hexagon)
    hexa = convert(HexagonCubic, a)
    hexb = convert(HexagonCubic, b)
    max(abs(hexa.x - hexb.x),
        abs(hexa.y - hexb.y),
        abs(hexa.z - hexb.z))
end



# TODO: Split up in two functions for performance (distance)?
function neighbor(hex::HexagonCubic, direction::Int, distance::Int = 1)
    dx = CUBIC_HEX_NEIGHBOR_OFFSETS[direction, 1] * distance
    dy = CUBIC_HEX_NEIGHBOR_OFFSETS[direction, 2] * distance
    dz = CUBIC_HEX_NEIGHBOR_OFFSETS[direction, 3] * distance
    HexagonCubic(hex.x + dx, hex.y + dy, hex.z + dz)
end

function cube_linedraw(a::Hexagon, b::Hexagon)
    hexa = convert(HexagonCubic, a)
    hexb = convert(HexagonCubic, b)
    N = distance(hexa, hexb)
    dx, dy, dz = hexb.x - hexa.x, hexb.y - hexa.y, hexb.z - hexa.z
    ax, ay, az = hexa.x + 1e-6, hexa.y + 1e-6, hexa.z - 2e-6
    map(i -> nearest_cubic_hexagon(ax + i*dx, ay + i*dy, az + i*dz), 0:(1/N):1)
end

# Find the nearest hexagon in cubic coordinates.
function nearest_cubic_hexagon(x::Real, y::Real, z::Real)
    rx, ry, rz = round(Integer, x), round(Integer, y), round(Integer, z)
    x_diff, y_diff, z_diff = abs(rx - x), abs(ry - y), abs(rz - z)

    if x_diff > y_diff && x_diff > z_diff
        rx = -ry - rz
    elseif y_diff > z_diff
        ry = -rx - rz
    else
        rz = -rx - ry
    end

    HexagonCubic(rx, ry, rz)
end

# Return the index (in cubic coordinates) of the hexagon containing the
# point x, y
function cube_round(x, y, xsize=1.0, ysize=1.0)
    x /= xsize
    y /= ysize
    q = sqrt(3)/3 * x - y/3
    r = 2 * y / 3
    h = nearest_cubic_hexagon(q, -q - r, r)
    return h
end
=#

end # module Hexagons
