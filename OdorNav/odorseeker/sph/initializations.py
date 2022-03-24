#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" ODORSEEKER: SPH simulation for odorant plume simulation
    Copyright (C) 2019  Batuhan Başerdem

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as n

def init_particle_box(num=5, size=.1, dim=2, pack=2):
    """ Generate particles equally spaced in dim dimensional box """
    # Generator for introducing new dimensions
    pgen = n.linspace(
        (size/2)*(pack+1)/(num*pack+1),
        (size/2)*((2*num-1)*pack+1)/(num*pack+1),
        num)
    # Generator for capping new dimensions with boundaries
    bgen = n.linspace(0, size, pack*num+2)
    # Points and boundary points
    pts = n.empty((1, 0))
    bnd = n.empty((0, 0))
    # Storing the caps to append to the boundary
    cgen = n.linspace(0, size, 2)
    cap = n.empty((1, 0))
    for _ in range(dim):
        # Multiply existing points along the new axis
        pts = n.column_stack((
            n.tile(pts, [num, 1]),
            n.repeat(pgen, pts.shape[0])
            ))
        # Multiply boundary accross the new dimension, including the edge
        bnd = n.column_stack((
            n.tile(bnd, [pack*num+2, 1]),
            n.repeat(bgen, bnd.shape[0])
            ))
        # Append the cap at the maxima of new dimension
        tip = n.column_stack((
            n.tile(cap, [2, 1]),
            n.repeat(cgen, cap.shape[0])
            ))
        bnd = n.vstack((bnd, tip))
        # Create the new cap
        cap = n.column_stack((
            n.tile(cap, [pack*num, 1]),
            n.repeat(bgen[1:-1], cap.shape[0])
            ))
    # Format output as dictionary
    output = {
        'particles': pts,
        'boundary': bnd
        }

    return output


if __name__ == "__main__":
    # Do a display of initialization if called explicitly
    def print_color(inp, col, bold=False):
        """ Color string for nice viewing """
        ansi = {
            'none': '',
            'red': '\u001b[31m',
            'orange': '\u001b[33m',
            'yellow': '\u001b[93m',
            'green': '\u001b[32m',
            'cyan': '\u001b[36m',
            'blue': '\u001b[34m',
            'violet': '\u001b[35m',
            'brown': '\u001b[95m',
            'bold': '\u001b[1m',
            'reset': '\u001b[0m'}
        pre = ansi[col]
        if bold:
            pre = ansi['bold'] + pre
        pos = ansi['reset']
        print(pre, end="")
        print(inp)
        print(pos, end="")

    print_color('Demo of box initialization function:', 'cyan', bold=True)

    OUT = init_particle_box(num=10, size=1, dim=1, pack=1)
    print_color('\nTen particles in 1D:', 'orange', bold=True)
    print_color('Particles:', 'blue')
    print_color(OUT['particles'], 'blue')
    print_color('Boundary particles:', 'violet')
    print_color(OUT['boundary'], 'violet')

    OUT = init_particle_box(num=2, size=1, dim=2, pack=1)
    print_color('\nFour particles in 2D:', 'orange', bold=True)
    print_color('Particles:', 'blue')
    print_color(OUT['particles'], 'blue')
    print_color('Boundary particles:', 'violet')
    print_color(OUT['boundary'], 'violet')

    OUT = init_particle_box(num=1, size=1, dim=3, pack=1)
    print_color('\nOne particle in 3D:', 'orange', bold=True)
    print_color('Particles:', 'blue')
    print_color(OUT['particles'], 'blue')
    print_color('Boundary particles:', 'violet')
    print_color(OUT['boundary'], 'violet')
