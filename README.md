
# Hexagons

[![Build
Status](https://travis-ci.org/GiovineItalia/Hexagons.jl.svg?branch=master)](https://travis-ci.org/GiovineItalia/Hexagons.jl)

This package provides some basic utilities for working with hexagonal grids.
I forked [GiovineItalia's Hexagons.jl](https://github.com/GiovineItalia/Hexagons.jl) for several reasons:
* The previous Hexagons.jl repository doesn't seem fully maintained, at least as of this writing;
* The previous Hexagons.jl package only supports pointy-top hexagons, while I usually find flat-topped more visually appealing;
* The previous package supports offset coordinates, which I do not want to support!
Both this fork and the original Hexagons.jl are largely derived from Amit Patel's [terrific
refererence](http://www.redblobgames.com/grids/hexagons/).

## Status

The fork is in progress. As of 2021-01-23, if you need hexagon functionality,
I recommend the [`ef_zero_center` branch of this repo](https://github.com/evanfields/Hexagons.jl/tree/ef_zero_center).
That branch is the original fork plus a bugfix, whereas `master` is subject to sudden breaking changes.
