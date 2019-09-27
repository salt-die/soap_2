#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soap 2: another useless experiment in pygame

Like soap, we have pokable voronoi cells, but here we sample colors from images.

Poke voronoi cells by left-clicking.
'v' to toggle between Voronoi cells and Delaunay Triangulation.
'r' to reset the cells.
"""
import numpy as np
from numpy import array, where
from numpy.linalg import norm
from numpy.random import random_sample
from PIL import Image
from scipy.spatial.qhull import QhullError, Voronoi, Delaunay
import pygame
from pygame.mouse import get_pos as mouse_xy
from pygame.draw import polygon

#Load image to sample
PATH = '/home/salt/Documents/Python/pygame/soap_2/mona_lisa.jpg'
#PATH = '/home/salt/Documents/Python/pygame/soap_2/starry_night.jpg'
#PATH = '/home/salt/Documents/Python/pygame/soap_2/american_gothic.jpeg'
with Image.open(PATH) as image:
    DIM = array(image.size)
    IMAGE = np.frombuffer(image.tobytes(), dtype=np.uint8)
    IMAGE = IMAGE.reshape((DIM[1], DIM[0], 3))

CELLS = 500  #Number of voronoi cells.
MAX_VEL = 15  #Max velocity of the cell centers.


class Center:
    """
    Cell center class.  Cell centers have methods that affect their movement.
    """
    FRICTION = .97

    def __init__(self):
        self.velocity = array([0.0, 0.0])
        self.loc = random_sample(2) * DIM

    def delta_velocity(self, delta):
        """
        This adds delta to the velocity and then limits the velocity to
        MAX_VEL.
        """
        self.velocity += delta
        magnitude = norm(self.velocity)
        if magnitude > MAX_VEL:
            self.velocity *= MAX_VEL / magnitude

    def move(self):
        """
        Apply velocity to location.
        """
        #Reverse the velocity if out-of-bounds
        self.velocity[(self.loc < MAX_VEL) | (self.loc > DIM - MAX_VEL)] *= -1
        self.loc += self.velocity
        self.loc %= DIM  #Wrap around borders if reversing didn't prevent OOB
        self.velocity *= self.FRICTION  #Reduce velocity from friction
        self.velocity[abs(self.velocity) < .01] = 0.0  #Prevent jitter

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Soap 2')
        self.window = pygame.display.set_mode(DIM)
        self.reset() #Randomly place cell centers
        self.running = True
        self.voronoi = True

    def reset(self):
        """
        Reset position of the cell centers.
        """
        self.centers = {Center() for _ in range(CELLS)}

    def draw_voronoi_cells(self):
        """
        This function will handle drawing voronoi cells.
        """
        points = [center.loc for center in self.centers]
        try:
            vor = Voronoi(points)
        except (QhullError, ValueError):
            #Either too few points or points are degenerate.
            return

        polygons = [(vor.points[where(vor.point_region == i)][0],
                    [vor.vertices[j] for j in reg if j != -1])
                    for i, reg in enumerate(vor.regions)
                    if len(reg) > 3 or (len(reg) == 3 and -1 not in reg)]

        for center, poly in polygons:
            polygon(self.window, IMAGE[tuple(center.astype(int))[::-1]], poly)

    def draw_delaunay_triangulation(self):
        """
        Draws the Delaunay triangulation of cell centers.
        """
        points = [center.loc for center in self.centers]
        try:
            dual = Delaunay(points)
        except (QhullError, ValueError):
            #Either too few points or points are degenerate.
            return

        simplices = [[dual.points[i] for i in simplex]
                     for simplex in dual.simplices]

        for simplex in simplices:
            centroid = tuple((sum(simplex) / 3).astype(int))[::-1]
            polygon(self.window, IMAGE[centroid], simplex)

    def poke(self, loc):
        """
        Calculates how much a poke affects every center's velocity.
        """
        for center in self.centers:
            difference = center.loc - loc
            distance = norm(difference)
            poke_power = 100000 * difference / distance**3 if distance else 0
            center.delta_velocity(poke_power)

    def user_input(self):
        """
        Read input.
        """
        for event in pygame.event.get():
            if event.type == 12:  #Quit
                self.running = False
            elif event.type == 2:  #key down
                if event.key == 114:  #'r'
                    self.reset()
                elif event.key == 118:  #'v'
                    self.voronoi = not self.voronoi
            elif event.type == 5:  #Mouse down
                if event.button == 1:  #left-Click
                    self.poke(array(mouse_xy()))

    def move_centers(self):
        """
        Move the centers.
        """
        #Movement for cell centers
        for center in self.centers:
            center.move()

    def start(self):
        while self.running:
            self.window.fill((63, 63, 63))
            if self.voronoi:
                self.draw_voronoi_cells()
            else:
                self.draw_delaunay_triangulation()
            pygame.display.update()
            self.user_input()
            self.move_centers()
        pygame.quit()

if __name__ == "__main__":
    Game().start()
