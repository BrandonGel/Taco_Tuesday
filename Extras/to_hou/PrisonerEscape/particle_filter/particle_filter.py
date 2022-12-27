import numpy as np
from simulator.abstract_object import AbstractObject
from simulator.utils import distance
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


class Particle:
    def __init__(self, env, location, target=None, direction=None):
        self.env = env
        self.location = np.array(location)
        self.target = np.array(target.location) if target is not None else None
        self.direction = np.array(direction) if direction is not None else None
        self.weight = 1
        self.speed = np.random.uniform(5, 7)  # reduced it for prediction

    def propagate_direction(self):
        step = self.direction * self.speed
        new_location = np.round([self.location[0] + step[0], self.location[1] + step[1]]).astype(np.int)
        if self.env.terrain.violate_edge_constraints(new_location[0], new_location[1], 1, 1):
            return
        if self.env.terrain.location_in_mountain(new_location):
            self.weight = 0
            return
        self.location = new_location

    def propagate(self):
        path_vector = np.array([self.target[0] - self.location[0], self.target[1] - self.location[1]])
        distance_movement = np.sqrt(path_vector[0] ** 2 + path_vector[1] ** 2)
        if distance_movement == 0:
            return
        direction = path_vector / distance_movement
        step = direction * self.speed
        new_location = np.round([self.location[0] + step[0], self.location[1] + step[1]]).astype(np.int)
        if self.env.terrain.violate_edge_constraints(new_location[0], new_location[1], 1, 1):
            return
        if self.env.terrain.location_in_mountain(new_location):
            self.weight = 0
            return
        self.location = new_location

    def adjust_weight(self, detected_location):
        distance_from_ground_truth = distance(self.location, detected_location)
        distance_from_ground_truth_rescaled = distance_from_ground_truth / np.sqrt(
            self.env.terrain.dim_x ** 2 + self.env.terrain.dim_y ** 2)
        confidence = 1 - distance_from_ground_truth_rescaled
        self.weight *= confidence

    def adjust_weight_direction(self, detected_direction):
        cos = detected_direction[0] * self.direction[0] + detected_direction[1] * self.direction[1]
        cos += 1.01  # make it always positive
        assert cos >= 0
        self.weight *= cos


class ParticleFilter:
    def __init__(self, env, num_particles=10000, direction_mode=False, initialize_within_camera_grid=False):
        """
        Creates an instance of particle filter -- baseline for our filtering and prediction pipelines
        :param env: instance of the Prisoner env
        :param num_particles: Number of particles for the particle filter
        :param direction_mode:
                    If true, initializes orientation of particles in random directions
                    If false, initializes a target (hideout) for each particle, and propagates the particles to that hideout
        :param initialize_within_camera_grid:
                    If true, initializes location of all particles within the camera grid (top-right corner)
                    If false, picks a random location within the env as the initial location for the particle
        """
        self.env = env
        self.num_particles = num_particles
        self.particles = []
        for _ in range(num_particles):
            # Whether to initialize particle locations within camera grid or randomly in the environment
            if initialize_within_camera_grid:
                location = AbstractObject.generate_random_locations_with_range(
                    (self.env.terrain.dim_x - 400, self.env.terrain.dim_x),
                    (self.env.terrain.dim_y - 400, self.env.terrain.dim_y))
            else:
                location = AbstractObject.generate_random_locations(self.env.terrain.dim_x, self.env.terrain.dim_y)

            if direction_mode:
                random_angle = np.random.uniform(-np.pi, np.pi)
                self.particles.append(Particle(env,
                                               location=location,
                                               direction=[np.cos(random_angle), np.sin(random_angle)]))
            else:
                # self.particles.append(Particle(env,
                #                                location=location,
                #                                target=np.random.choice(env.hideout_list)))  # TODO: We can't use locations of unknown hideouts. Fix this!

                # Initialize direction of the particles away from the terrain map edges
                # (useful when we initialize particles within the camera grid)
                random_angle = np.random.uniform(-np.pi/2, -np.pi)
                self.particles.append(Particle(env,
                                               location=location,
                                               direction=[np.cos(random_angle), np.sin(random_angle)]))

    def propagate(self):
        for particle in self.particles:
            if particle.weight > 0:
                particle.propagate()

    def propagate_direction(self):
        for particle in self.particles:
            if particle.weight > 0:
                particle.propagate_direction()

    def update_weight_according_to_detection(self, ground_truth_location):
        for particle in self.particles:
            if particle.weight > 0:
                particle.adjust_weight(ground_truth_location)

    def update_weight_according_to_direction(self, direction):
        for particle in self.particles:
            if particle.weight > 0:
                particle.adjust_weight_direction(direction)

    def update_particle_speed_direction(self, direction, speed=None):
        """
        Update the speed and detection of particles based on the detected fugitive's speed detection
        Here assuming that we can detect the speed and location of the fugitive as well
        :param direction: heading of the fugitive (must be between -pi to pi) at the time of detection
        :param speed: speed of the fugitive at the time of detection
        :return:
        """
        for particle in self.particles:
            if particle.weight > 0:
                if speed is not None:
                    particle.speed = speed
                if direction is not None:
                    particle.direction = direction

    def update_weights_without_detection(self):
        """
        Update weights of particles (close to blue agents) when the fugitive is not detected
        TODO ...
        :return:
        """
        raise NotImplementedError

    def normalize_all_weights(self):
        weights_sum = 0
        for particle in self.particles:
            weights_sum += particle.weight
        for particle in self.particles:
            particle.weight /= weights_sum

    def plot(self, save_name, to_cv=False, ground_truth_location=None, top_particles=100, hideout_locations=None):
        plt.clf()
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
        ax.set_xlim(0, 2428)
        ax.set_ylim(0, 2428)
        x, y, w = [], [], []

        particle_with_weight = []
        for particle in self.particles:
            particle_with_weight.append((particle.weight, particle))
        particle_with_weight_sorted = sorted(particle_with_weight, key=lambda x: x[0], reverse=True)
        # for particle in self.particles[:100]:
        for weight, particle in particle_with_weight_sorted[:top_particles]:
            # if particle.weight > 0:
            x.append(particle.location[0])
            y.append(particle.location[1])
            w.append(particle.weight)
            # ax.scatter(particle.location[0], particle.location[1], color='b')
            # else:
            #     ax.scatter(particle.location[0], particle.location[1], color='grey')
        x.extend([0, 0, 2428, 2428])
        y.extend([0, 2428, 0, 2428])
        w.extend([1e-6, 1e-6, 1e-6, 1e-6])
        sns.kdeplot(x=x, y=y, weights=w, cmap=None, fill=True, thresh=0, levels=100, )
        # print(ground_truth_location)
        if ground_truth_location is not None:
            ax.scatter(ground_truth_location[0], ground_truth_location[1], color='r', s=50)

        if hideout_locations is not None:
            for hideout in hideout_locations:
                loc = hideout.location
                ax.scatter(loc[0], loc[1], marker='x', color='yellow', s=50)

        if to_cv:
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            plt.close()
            plt.clf()
            return img
        else:
            plt.savefig(save_name)
            plt.close()
            plt.clf()
            return None
