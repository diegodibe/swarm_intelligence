import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.use('TkAgg')

'''
Gradient descent implementation ASR assignment 1

Authors: Diego Di Benedetto, Julia Hindel
'''

class Gradient:
    learning_rate = 0.001

    def __init__(self, rosenbrock_function):
        self.position = [random.uniform(world.min_X, world.max_X - world.correction_max), random.uniform(world.min_Y, world.max_Y - world.correction_max)]
        self.rosenbrock_function = rosenbrock_function
        # past positions of algorithm
        self.history = []
        self.finished = False

    def derivative_rosenbrock(self, point):
        dx = 2*point[0] - 400*point[0]*(point[1] - (point[0]**2))
        dy = 200 * (point[1] - (point[0] ** 2))
        return dx, dy

    def derivative_rastrigin(self, point):
        dx = 20 * np.pi * np.sin(2 * np.pi * point[0]) + 2 * point[0]
        return dx, dx

    def timestep(self):
        # save indexes of old position of gradient descent
        old_gradient_x = np.round(
            ((self.position[0] - world.min_X) / (world.max_X - world.min_X)) * world.discretization, 0).astype(int)
        old_gradient_y = np.round(
            ((self.position[1] - world.min_Y) / (world.max_Y - world.min_Y)) * world.discretization, 0).astype(int)
        self.history.append(self.position)
        if self.rosenbrock_function:
            delta = np.array(self.derivative_rosenbrock(self.position))
        else:
            delta = np.array(self.derivative_rastrigin(self.position))

        # new position
        self.position = self.position - np.dot(self.learning_rate, delta)

        # check if threshold for convergence is reached, new position is close to previous one
        new_gradient_x = np.round(
            ((self.position[0] - world.min_X) / (world.max_X - world.min_X)) * world.discretization, 0).astype(
            int)
        new_gradient_y = np.round(
            ((self.position[1] - world.min_Y) / (world.max_Y - world.min_Y)) * world.discretization, 0).astype(
            int)
        # if abs(world.Z[new_gradient_x][new_gradient_y] - world.Z[old_gradient_x][old_gradient_y]) < 0.0001:
        if abs(new_gradient_x - old_gradient_x) < 3 and abs(new_gradient_y - old_gradient_y) < 3:
            self.finished = True


class World:
    # coordinates of the world
    # rosenbrock
    # min_X = -2.5
    # max_X = 2.5  # max is excluded in np.arange
    # min_Y = -1.5
    # max_Y = 3.5

    # rastrigin, symmetric
    min_X = -5.
    max_X = 5.5  # max is excluded in np.arange
    min_Y = -5.
    max_Y = 5.5
    discretization = 999

    # parameter in rosenbrock function
    b = 100
    # True: rosenbrock function, False: rastrigin function
    rosenbrock_val = False

    def __init__(self):
        self.correction_max = (self.max_X - self.min_X) / self.discretization + 1
        x = np.linspace(self.min_X, self.max_X, self.discretization + 1)
        y = np.linspace(self.min_Y, self.max_Y, self.discretization + 1)
        self.X, self.Y = np.meshgrid(x, y)
        if self.rosenbrock_val:
            self.Z = self.rosenbrock(self.X, self.Y)
        else:
            self.Z = self.rastrigin(self.X, self.Y)

    def rosenbrock(self, X, Y):
        a = 0
        return (a - X) ** 2 + self.b * ((Y - X * X) ** 2)

    def rastrigin(self, X, Y):
        n = 2
        return (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 10 * n

    def display_world(self, Z):
        fig, ax = plt.subplots()
        plt.ion()

        surf = ax.imshow(Z, cmap=cm.Spectral, interpolation='nearest')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        return fig, ax

    def render_world(self, ax, gradient, Z, count):
        ax.clear()
        ax.imshow(Z, cmap=cm.Spectral, interpolation='nearest')
        # plt.locator_params(axis='x', nbins=10)
        # plt.locator_params(axis='y', nbins=10)
        # ax.set_xticklabels(np.arange(self.min_X - 1, self.max_X))
        # ax.set_yticklabels(np.arange(self.min_Y - 1, self.max_Y))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for entry in gradient.history:
            history_x = np.round(
                ((entry[0] - self.min_X) / (self.max_X - self.min_X)) * self.discretization, 0).astype(
                int)
            history_y = np.round(
                ((entry[1] - self.min_Y) / (self.max_Y - self.min_Y)) * self.discretization, 0).astype(
                int)
            ax.scatter(history_x, history_y, c='gray', s=50, marker='+')
        particle_x = np.round(
            ((gradient.position[0] - self.min_X) / (self.max_X - self.min_X)) * self.discretization, 0).astype(int)
        particle_y = np.round(
            ((gradient.position[1] - self.min_Y) / (self.max_Y - self.min_Y)) * self.discretization, 0).astype(int)
        ax.scatter(particle_x, particle_y, c='black', s=300, marker='x', zorder=1)
        plt.title(f'Iteration {count}')
        plt.draw()


results = []
no_simulations = 5

for i in range(no_simulations):
    print(f'simulation : {i}')
    world = World()
    gradient = Gradient(world.rosenbrock_val)
    fig, ax = world.display_world(world.Z)
    count = 0
    while count < 100 and not gradient.finished:
        gradient.timestep()
        count += 1
        world.render_world(ax, gradient, world.Z, count)
        # plt.waitforbuttonpress()
        plt.pause(0.01)
    gradient_x = np.round(
        ((gradient.position[0] - world.min_X) / (world.max_X - world.min_X)) * world.discretization, 0).astype(
        int)
    gradient_y = np.round(
        ((gradient.position[1] - world.min_Y) / (world.max_Y - world.min_Y)) * world.discretization, 0).astype(
        int)
    print(gradient.position, world.Z[gradient_x][gradient_y])
    results.append(world.Z[gradient_x][gradient_y])

#np.save('gradient_rose.npy', np.array(results))


