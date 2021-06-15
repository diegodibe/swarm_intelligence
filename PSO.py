import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.use('TkAgg')

'''
PSO implementation ASR assignment 1

Authors: Diego Di Benedetto, Julia Hindel
'''

class Swarm:
    # number of particles in swarm
    num = 20

    class Particle:
        # parameters to calculate velocity of PSO particle
        a = 0.8
        a_min = 0.5
        b = 1.8
        c = 1.2

        def __init__(self, no_iterations):
            self.position = [random.uniform(world.min_X, world.max_X - world.correction_max), random.uniform(world.min_Y, world.max_Y - world.correction_max)]
            self.local_best = [random.uniform(world.min_X, world.max_X - world.correction_max), random.uniform(world.min_Y, world.max_Y - world.correction_max)]
            self.max_velocity = [(world.max_X - world.min_X) / 20, (world.max_X - world.min_X) / 20]
            # reduction interval of a
            self.interval = (self.a - self.a_min) / no_iterations
            self.velocity = 0
            # past positions of particle
            self.history = []

        def calculate_velocity(self):
            self.velocity = self.a * self.velocity + self.b * random.random() \
                            * (np.array(self.local_best) - np.array(self.position)) \
                            + self.c * random.random() * (np.array(swarm.global_best) - np.array(self.position))

            # check if new velocity is within set boundaries
            if self.velocity[0] > self.max_velocity[0]:
                self.velocity[0] = self.max_velocity[0]
            if self.velocity[1] > self.max_velocity[1]:
                self.velocity[1] = self.max_velocity[1]
            if self.velocity[0] < -self.max_velocity[0]:
                self.velocity[0] = -self.max_velocity[0]
            if self.velocity[1] < -self.max_velocity[1]:
                self.velocity[1] = -self.max_velocity[1]

        def calculate_fitness(self):
            # indexes of position and local best calculation
            particle_x = np.round(
                ((self.position[0] - world.min_X) / (world.max_X - world.min_X)) * world.discretization, 0).astype(int)
            particle_y = np.round(
                ((self.position[1] - world.min_Y) / (world.max_Y - world.min_Y)) * world.discretization, 0).astype(int)
            local_x = np.round(
                ((self.local_best[0] - world.min_X) / (world.max_X - world.min_X)) * world.discretization, 0).astype(int)
            local_y = np.round(
                ((self.local_best[1] - world.min_Y) / (world.max_Y - world.min_Y)) * world.discretization, 0).astype(int)

            # new local best
            if world.Z[particle_x][particle_y] < world.Z[local_x][local_y]:
                self.local_best = self.position.copy()
                # indexes of global best calculation
                global_x = np.round(
                    ((swarm.global_best[0] - world.min_X) / (world.max_X - world.min_X)) * world.discretization, 0).astype(
                    int)
                global_y = np.round(
                    ((swarm.global_best[1] - world.min_Y) / (world.max_Y - world.min_Y)) * world.discretization , 0).astype(
                    int)
                if world.Z[particle_x][particle_y] < world.Z[global_x][global_y]:
                    #if abs(world.Z[global_x][global_y] - world.Z[particle_x][particle_y]) < 0.001:

                    # threshold reached if new global position is close to the old one
                    if abs(global_x - particle_x) < 3 and abs(global_y - particle_y) < 3:
                        swarm.finished = True
                        print('small change')
                    swarm.global_best = self.position.copy()
                    swarm.global_changed = True

        def inertia(self):
            # lower inertia, corresponding to the 'a' value, over time
            self.a -= self.interval

        def move(self):
            self.inertia()
            self.history.append(self.position)
            self.calculate_velocity()
            self.position = self.position + self.velocity

            # check if new position is within the environment
            if self.position[0] > world.max_X:
                self.position[0] = world.max_X - world.correction_max
            elif self.position[0] < world.min_X:
                self.position[0] = world.min_X
            if self.position[1] > world.max_Y:
                self.position[1] = world.max_Y - world.correction_max
            elif self.position[1] < world.min_Y:
                self.position[1] = world.min_Y
            self.calculate_fitness()

    # init swarm
    def __init__(self, min_x, max_x, min_y, max_y, no_iterations):
        self.global_best = [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
        self.particles = []
        self.finished = False
        # count how many iterations there have been no change to the global optimum
        self.counter_threshold = 0
        self.global_changed = False
        for i in range(self.num):
            self.particles.append(Swarm.Particle(no_iterations))

    def timestep(self):
        # perform a move for each particle and check if the threshold has been reached
        for particle in self.particles:
            particle.move()
        self.check_threshold()

    def check_threshold(self):
        # threshold is reached when the the global best does not change in 20 iterations
        if not self.global_changed:  # no particle changed global threshold
            self.counter_threshold += 1
        else:
            self.counter_threshold = 0
            self.global_changed = False
        if self.counter_threshold == 20:
            self.finished = True


class World:
    # coordinates of the environment
    # rosenbrock
    min_X = -2.5
    max_X = 2.5 # max is excluded in np.arange
    min_Y = -1.5
    max_Y = 3.5

    # rastrigin, symmetric
    # min_X = -5.
    # max_X = 5.5  # max is excluded in np.arange
    # min_Y = -5.
    # max_Y = 5.5

    discretization = 999 # +1 is added

    # parameter in rosenbrock function
    b = 100
    # True: rosenbrock function, False: rastrigin function
    rosenbrock_val = True

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

    def render_world(self, fig, ax, particles, Z, count):
        ax.clear()
        surf = ax.imshow(Z, cmap=cm.Spectral, interpolation='nearest')
        # ax.tick_params(axis='x', which='minor', direction='out', bottom=True, length=5)
        # plt.locator_params(axis='x', nbins=10)
        # plt.locator_params(axis='y', nbins=10)
        # ax.set_xticklabels(np.arange(self.min_X-0.5, self.max_X, step=0.5))
        # ax.set_yticklabels(np.arange(self.min_Y-0.5, self.max_Y, step=0.5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for particle in particles:
            for entry in particle.history[-5:]:
                history_x = np.round(
                    ((entry[0] - self.min_X) / (self.max_X - self.min_X)) * self.discretization, 0).astype(
                    int)
                history_y = np.round(
                    ((entry[1] - self.min_Y) / (self.max_Y - self.min_Y)) * self.discretization, 0).astype(
                    int)
                ax.scatter(history_x, history_y, c='gray', s=50, marker='+')
            particle_x = np.round(
                ((particle.position[0] - self.min_X) / (self.max_X - self.min_X)) * self.discretization, 0).astype(int)
            particle_y = np.round(
                ((particle.position[1] - self.min_Y) / (self.max_Y - self.min_Y)) * self.discretization, 0).astype(int)
            ax.scatter(particle_x, particle_y, c='black', s=300, marker='x', zorder=1)
        plt.title(f'Iteration {count}')
        plt.draw()

# save achieved minimum
results = []
# number of iterations
max = 100

no_simulations = 5
for i in range(no_simulations):
    print(f'simulation : {i}')
    world = World()
    fig, ax = world.display_world(world.Z)
    swarm = Swarm(world.min_X, world.max_X, world.min_Y, world.max_Y, max)
    count = 0
    while count < max and not swarm.finished:
        swarm.timestep()
        count += 1
        # for p in swarm.particles:
        #     print(p.position)
        world.render_world(fig, ax, swarm.particles, world.Z, count)
        # pause before next iteration
        plt.pause(0.1)

    # calculate indexes of the global minimum and use them to store the value of the environment
    global_x = np.round(
        ((swarm.global_best[0] - world.min_X) / (world.max_X - world.min_X)) * world.discretization, 0).astype(
        int)
    global_y = np.round(
        ((swarm.global_best[1] - world.min_Y) / (world.max_Y - world.min_Y)) * world.discretization, 0).astype(
        int)
    print('Global best: ', swarm.global_best, ' value: ', world.Z[global_x][global_y])
    results.append(world.Z[global_x][global_y])

#np.save('swarm_rastri_wang.npy', np.array(results))





