import numpy as np
from random import randrange


def vec_to_onehot(vec, max_val=9):
    v = vec.astype(np.int).squeeze()
    one_hot = np.zeros((v.size, max_val + 1))
    one_hot[np.arange(v.size), v] = 1

    return one_hot


class ArenaDataset():
    class Arena:
        def __init__(self, batch_size, steps=500):
            self.rng = np.random.RandomState(seed=2)
            self.batch_size = batch_size
            self.steps = steps
            self.pos = np.zeros((batch_size, 2))
            self.direction = 2 * np.pi * np.random.rand(batch_size)

        def _in_bounds(self, pos):
            in_bounds = np.prod((pos > -1) & (pos < 1), axis=1) > 0
            return in_bounds

        def __iter__(self):
            self.__init__()
            return self

        def __next__(self):
            self.pos = np.zeros((self.batch_size, 2))
            self.direction = 2 * np.pi * self.rng.rand(self.batch_size)
            egocentric = np.empty((self.batch_size, self.steps, 2))
            allocentric = np.empty((self.batch_size, self.steps, 2))
            for i in range(self.steps):
                batch = self.step()
                egocentric[:, i, :], allocentric[:, i, :] = batch

            return egocentric, allocentric

        def step(self):
            speed = self.rng.rand(self.batch_size) * 0.2
            speed[self.rng.choice(self.batch_size, size=int(self.batch_size * 0.9), replace=False)] = 0

            ang_vel = self.rng.randn(self.batch_size) / 3
            direction = np.mod(ang_vel + self.direction, 2 * np.pi)

            # Testing when direction doesn't change when still
            # direction[speed == 0] = self.direction[speed == 0]
            
            pos = self.pos + np.stack([speed * np.cos(direction),
                                       speed * np.sin(direction)], axis=1)
            oob = ~self._in_bounds(pos)

            repeat = False
            while oob.any():
                if repeat:
                    speed[oob] = self.rng.rand(np.sum(oob)) * 0.2
                    ang_vel[oob] = self.rng.randn(np.sum(oob)) / 2
                else:
                    ang_vel[oob] = self.rng.randn(np.sum(oob)) / 3
                    
                direction[oob] = np.mod(ang_vel[oob] + self.direction[oob], 2 * np.pi)

                pos[oob, :] = self.pos[oob, :] + np.stack([speed[oob] * np.cos(direction[oob]),
                                                           speed[oob] * np.sin(direction[oob])],
                                                          axis=1)
                oob = ~self._in_bounds(pos)
                repeat = True

            self.pos = pos
            self.direction = direction
            self.speed = speed

            return np.stack([direction / (2 * np.pi), speed], axis=1), self.pos

    def __iter__(self):
        self.arena = self.Arena(self.batch_size, self.steps)
        return self.arena

    def __init__(self, batch_size=500, steps=500):
        super().__init__()
        self.batch_size = batch_size
        self.steps = steps


class SequenceDataset():
    class Arena:
        def __init__(self, batch_size, steps=100):
            self.batch_size = batch_size
            self.steps = steps
            self.pos = np.zeros((batch_size, 1))
            self.direction = np.random.choice([-1, 1], (batch_size, 1))

        def _in_bounds(self, pos):
            in_bounds = (pos <= 9) & (pos >= 0)
            return in_bounds

        def __iter__(self):
            self.__init__()
            return self

        def __next__(self):
            self.pos = np.zeros((self.batch_size, 1))
            self.direction = np.random.choice([-1, 1], (self.batch_size, 1))
            egocentric = np.empty((self.batch_size, self.steps, 2))
            allocentric = np.empty((self.batch_size, self.steps, 10))
            for i in range(self.steps):
                batch = self.step()
                egocentric[:, i, :], allocentric[:, i, :] = batch

            return egocentric, allocentric

        def step(self):
            speed = np.ones((self.batch_size, 1))
            speed[np.random.rand(self.batch_size, 1) > 0.1] = 0

            direction = np.random.choice(
                [-1, 1], size=(self.batch_size, 1), p=[1/3, 2/3]
            )

            pos = self.pos + speed * direction
            oob = ~self._in_bounds(pos)

            while oob.any():
                direction[oob] = 0

                pos[oob] = self.pos[oob] + speed[oob] * direction[oob]
                oob = ~self._in_bounds(pos)

            self.pos = pos
            self.direction = direction
            self.speed = speed

            return (np.concatenate([direction, speed], axis=1),
                    vec_to_onehot(self.pos))

    def __iter__(self):
        self.arena = self.Arena(self.batch_size, self.steps)
        return self.arena

    def __init__(self, batch_size=500, steps=100):
        super().__init__()
        self.batch_size = batch_size
        self.steps = steps


class LineDataset():
    class Arena:
        def __init__(self, batch_size, steps=100):
            self.batch_size = batch_size
            self.steps = steps
            self.pos = np.zeros((batch_size, 1))
            self.direction = np.random.choice([-1, 1], (batch_size, 1))

        def _in_bounds(self, pos):
            in_bounds = (pos < 1) & (pos > -1)
            return in_bounds

        def __iter__(self):
            self.__init__()
            return self

        def __next__(self):
            self.pos = np.zeros((self.batch_size, 1))
            self.direction = np.random.choice([-1, 1], (self.batch_size, 1))
            egocentric = np.empty((self.batch_size, self.steps, 2))
            allocentric = np.empty((self.batch_size, self.steps, 1))
            for i in range(self.steps):
                batch = self.step()
                egocentric[:, i, :], allocentric[:, i, :] = batch

            return egocentric, allocentric

        def step(self):
            speed = np.random.rand(self.batch_size, 1) * 0.2
            speed[np.random.rand(self.batch_size, 1) > 0.1] = 0

            direction = np.random.choice(
                [-1, 1], size=(self.batch_size, 1), p=[0.5, 0.5]
            )

            pos = self.pos + speed * direction
            oob = ~self._in_bounds(pos)

            while oob.any():
                direction[oob] = np.random.choice(
                    [-1, 1], size=np.sum(oob), p=[0.5, 0.5]
                )

                pos[oob] = self.pos[oob] + speed[oob] * direction[oob]
                oob = ~self._in_bounds(pos)

            self.pos = pos
            self.direction = direction
            self.speed = speed

            return (np.concatenate([direction, speed], axis=1), self.pos)

    def __iter__(self):
        self.arena = self.Arena(self.batch_size, self.steps)
        return self.arena

    def __init__(self, batch_size=500, steps=100):
        super().__init__()
        self.batch_size = batch_size
        self.steps = steps


class SphereDataset(ArenaDataset):
    class Arena:
        def __init__(self, batch_size, steps=500):
            self.batch_size = batch_size
            self.steps = steps
            self.pos = np.pi * np.ones((batch_size, 2))
            self.direction = 2 * np.pi * np.random.rand(batch_size)
            self.radius = 1

        def _in_bounds(self, pos):
            while ((pos > 2*np.pi) | (pos < 0)).any():
                pos[pos > 2*np.pi] += np.mod(pos, 2 * np.pi)
                pos[pos < 0] += np.mod(pos, 2 * np.pi)

            return pos

        def __next__(self):
            self.pos = np.pi * np.ones((self.batch_size, 2))
            self.direction = 2 * np.pi * np.random.rand(self.batch_size)
            self.speed = 0.1 * np.ones(self.batch_size)
            inertial_f = np.empty((self.batch_size, self.steps, 3))
            lab_f = np.empty((self.batch_size, self.steps, 2))
            for i in range(self.steps):
                batch = self.step()
                inertial_f[:, i, :], lab_f[:, i, :] = batch

            return inertial_f, lab_f

        def step(self):
            speed = np.random.rand(self.batch_size) * 0.2
            speed[np.random.rand(self.batch_size) > 0.1] = 0

            ang_vel = np.random.randn(self.batch_size) / 3
            direction = np.mod(ang_vel + self.direction, 2 * np.pi)

            r = np.stack([speed * np.cos(direction),
                          speed * np.sin(direction)], axis=1)

            pos = self.pos
            pos[r != 0] += r[r != 0] / self.radius

            pos = self._in_bounds(pos)
            
            self.pos = pos
            self.direction = direction
            self.speed = speed

            return np.stack([direction / (2 * np.pi), speed], axis=1), self.pos


class MNISTDataset():
    class MNIST:
        def __init__(self, batch_size, steps=10, train=True):
            self.batch_size = batch_size
            self.steps = steps + 1
            self.pos = np.random.randint(0, 4, (batch_size,)) * 3
            if train:
                self.mnist = [np.load('./mnist/digit-{}.npy'.format(d))
                              for d in range(10)]
            else:
                self.mnist = [np.load('./mnist/val-digit-{}.npy'.format(d))
                              for d in range(10)]
            
        def _in_bounds(self, pos):
            in_bounds = (pos >= 0) & (pos <= 9)
            return in_bounds
        
        def __iter__(self):
            self.__init__()
            return self

        def __next__(self):
            self.pos = np.random.randint(0, 10, (self.batch_size,))
            allocentric = np.empty((self.batch_size, self.steps))
            egocentric = [np.empty((self.batch_size, self.steps - 1)),
                          np.empty((self.batch_size, self.steps - 1, 1, 28, 28))]
            
            allocentric[:, 0] = self.pos
            for i in range(1, self.steps):
                batch = self.step()
                allocentric[:, i] = batch[1]
                egocentric[0][:, i - 1] = batch[0]
                egocentric[1][:, i - 1] = np.stack([self.mnist[d][randrange((len(self.mnist[d])))]
                                                    for d in batch[1]])
                
            return egocentric, allocentric

        def step(self):
            direction = np.random.randint(-1, 2, (self.batch_size,))
            p = np.random.rand(self.batch_size)
            direction[p < 0.25] = -1
            direction[(p >= 0.25) & (p < 0.5)] = 0
            direction[p > 0.5] = 1
            
            pos = self.pos + direction
            oob = ~self._in_bounds(pos)
            
            while oob.any():
                p = np.random.rand(self.batch_size)
                direction[(p < 0.25) & oob] = -1
                direction[(p >= 0.25) & (p < 0.5) & oob] = 0
                direction[(p > 0.5) & oob] = 1

                direction[oob] = np.random.randint(-1, 2, (np.sum(oob),))
                
                pos[oob] = self.pos[oob] + direction[oob]
                oob = ~self._in_bounds(pos)

            self.pos = pos

            return direction, self.pos

    def __iter__(self):
        self.mnist = self.MNIST(self.batch_size, self.steps)
        return self.mnist

    def __init__(self, batch_size=128, steps=5):
        super().__init__()
        self.batch_size = batch_size
        self.steps = steps
