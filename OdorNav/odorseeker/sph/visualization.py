#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" ODORSEEKER : SPH simulation for odorant plume simulation
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

# Important modules
import numpy as n
import matplotlib.pyplot as p

# Figure to draw plots on
F = 10

def particles_with_boundary_2D(part, boun):
    # Initialize plot
    p.figure(1)
    # Draw the points
    p1 = p.plot(part[:,0], part[:,1], 'ko' )
    p.setp(p1, markersize=10)
    p.setp(p1, markerfacecolor='#585858')
    # Draw the boundarie
    p2 = p.plot(boun[:,0], part[:,1], 'ko' )
    p.setp(p1, markersize=7.5)
    p.setp(p1, markerfacecolor='#383838')

