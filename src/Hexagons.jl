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

##
# Utilities in hexagon space
##

"""
    hex_distance(hexA::T, hexB::T) where {T <: Hexagon}

The distance in hexagon-space between two hexagons. Distances are only defined between
hexagons of the same type
"""
hex_distance(hexA::HexPointyTop, hexB::HexPointyTop) = hex_distance(hexA.coords, hexB.coords)

"""
    hex_distance(c1::AbstractCoord, c2::AbstractCoord)

The distance in hexagon-space between two hexagon coordinates.
"""
function hex_distance(c1::AbstractCoord, c2::AbstractCoord)
    return hex_distance(CoordCubic(c1), CoordCubic(c2))
end

# see RedBlobGames reference for derivation
function hex_distance(c1::CoordCubic, c2::CoordCubic)
    return (abs(c1.x - c2.x) + abs(c1.y - c2.y) + abs(c1.z - c2.z)) / 2
end

end # module Hexagons
