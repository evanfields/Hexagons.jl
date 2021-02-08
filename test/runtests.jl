using Hexagons
using Test
using Random: seed!
using Statistics: mean

# Test a few identities for the hexagon containing the point x, y
function run_point_test(x, y, hextype)
    hex = hex_containing(hextype, x, y)
    verts = vertices(hex)
    # x,y should be in the bounding box of the hexagon vertices
    @test minimum(v[1] for v in verts) <= x <= maximum(v[1] for v in verts)
    @test minimum(v[2] for v in verts) <= y <= maximum(v[2] for v in verts)
    # the center of the hexagon should be near its vertex mean
    mean_vert_x = mean(v[1] for v in verts)
    mean_vert_y = mean(v[2] for v in verts)
    hex_center = center(hex)
    @test isapprox(hex_center[1], mean_vert_x; atol = 1e-6)
    @test isapprox(hex_center[2], mean_vert_y; atol = 1e-6)
    # a string of type conversions should recover the original hex
    hex_axial = to_coord_form(CoordAxial, hex)
    hex_cubic = to_coord_form(CoordCubic, hex_axial)
    @test hex == hex_cubic
end
    
test_points = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
    (0, 47),
    (-4.7, 0),
    (1.234, 5.678),
    (1e6, 2e6),
]
# bunch of random test points
seed!(1234)
for _ in 1:1000
    push!(test_points, (rand() * 100, rand() * 100))
end
for point in test_points
    run_point_test(point..., HexPointyTop)
    run_point_test(point..., HexFlatTop)
end

# Test some hash and equality identities
ax0_int = CoordAxial(0, 0)
ax0_float = CoordAxial(0.0, 0.0)
ax1_int = CoordAxial(1, 1)
@test ax0_float == ax0_int
@test hash(ax0_float) == hash(ax0_int)
@test ax1_int != ax0_int
@test hash(ax1_int) != hash(ax0_int)
for T in (HexFlatTop, HexPointyTop)
    @test T(ax0_float) == T(ax0_int)
    @test hash(T(ax0_float)) == hash(T(ax0_int))
end
@test HexFlatTop(ax0_int) != HexPointyTop(ax0_int)
