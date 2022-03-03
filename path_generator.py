import torch
from torch.utils.data import IterableDataset
from torch import nn
from torch.nn import functional as f
from math import sqrt
from math import pi
import numpy as np
import math
import random


class HoleDataset(IterableDataset):
    def __init__(self, batch_size, steps=50, dataset_size=None, seed=1,
                    bounds=[[-0.5, 0.5, -0.5, 0.5]], arena_sz=[-1, 1, -1, 1]):
        super().__init__()
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)
        self.batch_size = batch_size
        self.steps = steps
        self.dataset_size = dataset_size
        self.iteration = 0
        self.bounds = bounds
        self.arena_sz = arena_sz

    def __iter__(self):
        return self

    def __next__(self):
        arena_sz = self.arena_sz

        # Reset the random seed if the number of samples exceeds the dataset size
        if self.dataset_size is not None and \
            self.dataset_size <= (self.batch_size * self.iteration):
            self.rng = np.random.RandomState(seed=self.seed)
            self.iteration = 0

        # Initialize position and direction
        initial_pos = self.initialize_pos()
        self.pos = initial_pos
        self.direction = 2 * np.pi * self.rng.rand(self.batch_size)

        # Calculate the paths for a random walk
        egocentric = np.empty((self.batch_size, self.steps, 2))
        allocentric = np.empty((self.batch_size, self.steps, 2))
        for i in range(self.steps):
            batch = self.step(i)
            egocentric[:, i, :], allocentric[:, i, :] = batch

        self.iteration += 1

        return torch.from_numpy(egocentric).float(), torch.from_numpy(allocentric).float(), torch.from_numpy(initial_pos).float()

    def step(self, iteration):
        speed = self.rng.rand(self.batch_size) * 0.05

        ang_vel = self.rng.randn(self.batch_size) / 2
        direction = np.mod(ang_vel + self.direction, 2 * np.pi)

        vel = np.stack([speed * np.cos(direction),
                        speed * np.sin(direction)], axis=1)

        r_f = self.pos + vel
        self.pos = self.calculate_path(self.pos, vel)
        self.direction = direction

        return vel, self.pos

    def out_of_bounds(self, pos, eps=1e-6):
        out_of_bounds = np.zeros(len(pos)).astype(bool)
        for bound in self.bounds:
            out_of_bounds |= (pos[:, 0] >= bound[0] - eps) & (pos[:, 0] <= bound[1] + eps) & \
                (pos[:, 1] >= bound[2] - eps) & (pos[:, 1] <= bound[3] + eps)
        return out_of_bounds

    def calculate_path(self, r_i, vel, eps=1e-6):
        # Get arena boundaries
        arena_sz = self.arena_sz

        # Get initial and final positions and calculate the displacement
        x_i, y_i = r_i[:, 0], r_i[:, 1]
        dx = vel[:, 0]
        dy = vel[:, 1]

        # Calculate the intersection between the path and the arena boundary
        # as t for r_f = r_i + t * dr
        t = np.stack([(arena_sz[1] - x_i) / dx, (arena_sz[0] - x_i) / dx,
                      (arena_sz[3] - y_i) / dy, (arena_sz[2] - y_i) / dy], axis=1)
        t[t < 0] = 1

        # Calculate the intersection between the path and the hole boundaries
        # as t for r_f = r_i + t *dr
        holes = []
        for bound in self.bounds:
            hole_t = np.stack([(bound[2] - y_i) / dy, (bound[3] - y_i) / dy,
                               (bound[0] - x_i) / dx, (bound[1] - x_i) / dx], axis=1)

            hole_t[hole_t < 0] = 1

            x_t = x_i[:, None] + hole_t*dx[:, None]
            y_t = y_i[:, None] + hole_t*dy[:, None]

            # If the path does not intersect the hole, then correct the intersection calculation
            hole = hole_t.copy()
            hole[((x_t <= bound[1]) & (x_t >= bound[0])) & ((y_t > bound[3]) | (y_t < bound[2]))] = 1
            hole[((x_t > bound[1]) | (x_t < bound[0])) & ((y_t <= bound[3]) & (y_t >= bound[2]))] = 1
            idx = (x_t <= bound[1] + eps) & (x_t >= bound[0] - eps) & (y_t <= bound[3] + eps) & (y_t >= bound[2] - eps)
            hole[idx] = hole_t[idx]
            holes.append(hole.copy())

        t = np.concatenate([t, *holes], axis=1)

        # Use the path that intersects the first boundary
        t = t.min(axis=1)

        # If an intersection occurs, shift the path back by epsilon (eps)
        # to avoid pathological behavior on boundary
        t[(t < 1) & (t > 0)] -= eps

        return r_i + t[:, None]*vel

    def initialize_pos(self, eps=1e-6):
        arena_sz = self.arena_sz
        
        initial_pos = np.stack([
            self.rng.uniform(low=arena_sz[0] + eps, high=arena_sz[1] - eps, size=self.batch_size),
            self.rng.uniform(low=arena_sz[2] + eps, high=arena_sz[3] - eps, size=self.batch_size),
        ], axis=1)
        oob = self.out_of_bounds(initial_pos)
        while oob.any():
            initial_pos[oob] = np.stack([
                self.rng.uniform(low=arena_sz[0] + eps, high=arena_sz[1] - eps, size=np.sum(oob)),
                self.rng.uniform(low=arena_sz[2] + eps, high=arena_sz[3] - eps, size=np.sum(oob)),
            ], axis=1)
            oob = self.out_of_bounds(initial_pos)

        return initial_pos


class MazeDataset(IterableDataset):
    def __init__(self, batch_size, adj_list, steps=50, dataset_size=None, seed=1):
        super().__init__()
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)
        self.batch_size = batch_size
        self.steps = steps
        self.dataset_size = dataset_size
        self.iteration = 0
        self.adj_list = adj_list
        self.edges = self._adj2edges(adj_list)
        self.bounds = [[x[0], x[2], x[1], x[3]] for x in self.edges]
        self.arena_sz = [-1, 1, -1, 1]

    def _adj2edges(self, adj_list):
        N = int(math.sqrt(len(adj_list)))
        edges = []
        for idx, lst in enumerate(adj_list):
            diff = set([ij2idx(i, j, N) for i, j in get_neighbors(N, idx)]) - set(lst)
            diff = [idx2ij(i, N) for i in list(diff)]
            for l in diff:
                i0, j0 = idx2ij(idx, N)
                i1, j1 = l
                di = i1-i0
                dj = j1-j0

                if di != 0:
                    edges.append([2/N*(i0+0.5*di+0.5) - 1, 2/N*(j0+1) - 1, 2/N*(i0 + 0.5*di+0.5) - 1, 2/N*j0 - 1])
                elif dj != 0:
                    edges.append([2/N*(i0+1) - 1, 2/N*(j0 + 0.5*dj+0.5) - 1, 2/N*i0 - 1, 2/N*(j0 + 0.5*dj+0.5) - 1])

        return edges

    def __iter__(self):
        return self

    def __next__(self):
        arena_sz = self.arena_sz

        # Reset the random seed if the number of samples exceeds the dataset size
        if self.dataset_size is not None and \
            self.dataset_size <= (self.batch_size * self.iteration):
            self.rng = np.random.RandomState(seed=self.seed)
            self.iteration = 0

        # Initialize position and direction
        initial_pos = self.initialize_pos()
        self.pos = initial_pos
        self.direction = 2 * np.pi * self.rng.rand(self.batch_size)

        # Calculate the paths for a random walk
        egocentric = np.empty((self.batch_size, self.steps, 2))
        allocentric = np.empty((self.batch_size, self.steps, 2))
        for i in range(self.steps):
            batch = self.step(i)
            egocentric[:, i, :], allocentric[:, i, :] = batch

        self.iteration += 1

        return torch.from_numpy(egocentric).float(), torch.from_numpy(allocentric).float(), torch.from_numpy(initial_pos).float()

    def step(self, iteration):
        speed = self.rng.rand(self.batch_size) * 0.1

        ang_vel = self.rng.randn(self.batch_size) / 3
        direction = np.mod(ang_vel + self.direction, 2 * np.pi)

        vel = np.stack([speed * np.cos(direction),
                        speed * np.sin(direction)], axis=1)

        r_f = self.pos + vel
        self.pos = self.calculate_path(self.pos, vel)
        self.direction = direction

        return vel, self.pos

    def out_of_bounds(self, pos, eps=1e-6):
        out_of_bounds = np.zeros(len(pos)).astype(bool)
        for edge in self.edges:
            out_of_bounds |= (pos[:, 0] >= edge[0]) & (pos[:, 0] <= edge[2]) & \
                (pos[:, 1] >= edge[1]) & (pos[:, 1] <= edge[3])
        return out_of_bounds

    def calculate_path(self, r_i, vel, eps=1e-6):
        # Get arena boundaries
        arena_sz = self.arena_sz

        # Get initial and final positions and calculate the displacement
        x_i, y_i = r_i[:, 0], r_i[:, 1]
        dx = vel[:, 0]
        dy = vel[:, 1]
        x_f, y_f = x_i + dx, y_i + dy

        # Calculate the intersection between the path and the arena boundary
        # as t for r_f = r_i + t * dr
        t = np.stack([(arena_sz[1] - x_i) / dx, (arena_sz[0] - x_i) / dx,
                      (arena_sz[3] - y_i) / dy, (arena_sz[2] - y_i) / dy], axis=1)
        t[t < 0] = 1

        # Calculate the intersection between the path and the hole boundaries
        # as t for r_f = r_i + t *dr
        lines = []
        for edge in self.edges:
            u_i, v_i, u_f, v_f = edge
            du, dv = u_f - u_i, v_f - v_i
            Z = 1/(dx*dv - dy*du)
            line_t = (u_i-x_i)*dv - (v_i-y_i)*du
            line_t *= Z
            line_s = (u_i-x_i)*dy - (v_i-y_i)*dx
            line_s *= Z

            line = line_t.copy()
            line[(line_t < 0) | (line_s > 1) | (line_s < 0)] = 1
            lines.append(line.copy())

        t = np.concatenate([t, np.stack(lines, axis=1)], axis=1)

        # Use the path that intersects the first boundary
        t = t.min(axis=1)

        # If an intersection occurs, shift the path back by epsilon (eps)
        # to avoid pathological behavior on boundary
        t[(t < 1) & (t > 0)] -= eps

        return r_i + t[:, None]*vel

    def initialize_pos(self, eps=1e-6):
        arena_sz = self.arena_sz
        
        initial_pos = np.stack([
            self.rng.uniform(low=arena_sz[0] + eps, high=arena_sz[1] - eps, size=self.batch_size),
            self.rng.uniform(low=arena_sz[2] + eps, high=arena_sz[3] - eps, size=self.batch_size),
        ], axis=1)
        oob = self.out_of_bounds(initial_pos)
        while oob.any():
            initial_pos[oob] = np.stack([
                self.rng.uniform(low=arena_sz[0] + eps, high=arena_sz[1] - eps, size=np.sum(oob)),
                self.rng.uniform(low=arena_sz[2] + eps, high=arena_sz[3] - eps, size=np.sum(oob)),
            ], axis=1)
            oob = self.out_of_bounds(initial_pos)

        return initial_pos


def idx2ij(idx, N):
    i = idx % N
    j = idx // N
    
    return i, j

def ij2idx(i, j, N):
    idx = N * j + i
    
    return idx

def get_neighbors(N, idx):
    neighbors = []
    i, j = idx2ij(idx, N)
    if i - 1 >= 0:
        neighbors.append((i-1, j))
    if i + 1 < N:
        neighbors.append((i+1, j))
    if j - 1 >= 0:
        neighbors.append((i, j-1))
    if j + 1 < N:
        neighbors.append((i, j+1))

    return neighbors


def generate_maze(N, seed=1):
    random.seed(seed)
    adj_list = [list() for i in range(N * N)]
    cell_stack = [0]
    visited = np.zeros(N * N, dtype=np.bool)
    while len(cell_stack) > 0:
        # Get top of the stack
        cur_cell = cell_stack[-1]
        
        # Mark current cell as visited
        visited[cur_cell] = True
        
        # Get unvisited, shuffled neighbors
        neighbors = map(lambda x: ij2idx(x[0], x[1], N), get_neighbors(N, cur_cell))
        neighbors = list(filter(lambda idx: not visited[idx], neighbors))
        neighbors = random.sample(neighbors, k=len(neighbors))
        
        if len(neighbors) == 0:
            cell_stack.pop()
            continue
        
        # Get a random neighbor
        cur_neighbor = neighbors.pop()
        visited[cur_neighbor] = True
        
        # Create an edge between the cell and neighbor
        adj_list[cur_cell].append(cur_neighbor)
        adj_list[cur_neighbor].append(cur_cell)
        
        # Append neighbor to the stack
        cell_stack.append(cur_neighbor)
        
    return adj_list
