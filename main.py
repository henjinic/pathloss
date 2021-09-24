import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

np.set_printoptions(precision=2)

class PathlossCalc:

    def __init__(self, cell_size=30, freq=900, dem=None):
        self._landcover_maps = []
        self._pathloss_functions = []
        self._cell_size = cell_size
        self._freq = freq

        self._routes = {
            (0, 1): {
                (0, 0): self._cell_size / 2,
                (0, 1): self._cell_size / 2
            },
            (1, 1): {
                (0, 0): self._cell_size / np.sqrt(2),
                (1, 1): self._cell_size / np.sqrt(2)
            },
            (1, 2): {
                (0, 0): self._cell_size * np.sqrt(5) / 4,
                (0, 1): self._cell_size * np.sqrt(5) / 4,
                (1, 1): self._cell_size * np.sqrt(5) / 4,
                (1, 2): self._cell_size * np.sqrt(5) / 4
            },
            (1, 3): {
                (0, 0): self._cell_size * np.sqrt(10) / 6,
                (0, 1): self._cell_size * np.sqrt(10) / 3,
                (1, 2): self._cell_size * np.sqrt(10) / 3,
                (1, 3): self._cell_size * np.sqrt(10) / 6
            },
            (1, 4): {
                (0, 0): self._cell_size * np.sqrt(17) / 8,
                (0, 1): self._cell_size * np.sqrt(17) / 4,
                (0, 2): self._cell_size * np.sqrt(17) / 8,
                (1, 2): self._cell_size * np.sqrt(17) / 8,
                (1, 3): self._cell_size * np.sqrt(17) / 4,
                (1, 4): self._cell_size * np.sqrt(17) / 8
            }
        }

        # negate
        temp_routes = deepcopy(self._routes)

        for (r, c), routes in temp_routes.items():
            for sign_r, sign_c in [(-1, 1), (1, -1), (-1, -1)]:
                if (r * sign_r, c * sign_c) in self._routes:
                    continue

                self._routes[r * sign_r, c * sign_c] = {(r * sign_r, c * sign_c): distance for (r, c), distance in routes.items()}

        # transpose
        temp_routes = deepcopy(self._routes)
        for (r, c), routes in temp_routes.items():
            if (c, r) in self._routes:
                continue

            self._routes[c, r] = {(c, r): distance for (r, c), distance in routes.items()}

    @property
    def shape(self):
        return self._landcover_maps[0].shape

    def add_landcover(self, landcover_map, pathloss_function):
        self._landcover_maps.append(landcover_map)
        self._pathloss_functions.append(pathloss_function)

    def run(self, antena_map, threshold=None):
        result = np.full(self.shape, 9999)

        for r, c in np.argwhere(antena_map):
            pathloss_map = self._fill(r, c, threshold)
            result = np.minimum(result, pathloss_map)

        result[result == 9999] = -1

        return result

    def _fill(self, r, c, threshold):
        result = np.full(self.shape, 9999)
        result[r, c] = 0

        for vector, route in self._routes.items():
            self._fill_recur(result, r, c, 0, vector, route, threshold)

        return result

    def _fill_recur(self, pathloss_map, r, c, src_distance, vector, route, threshold):
        dr, dc = vector

        if threshold is not None and src_distance > threshold:
            return pathloss_map

        if not self.is_in(r + dr, c + dc):
            return pathloss_map

        accumulated_pathloss = pathloss_map[r, c]
        accumulated_distance = src_distance

        for ref_coord, distance in route.items():
            accumulated_pathloss = self._calc_pathloss(accumulated_pathloss, accumulated_distance,
                                                       accumulated_distance + distance,
                                                       self._weights_at(*ref_coord))
            accumulated_distance += distance

        pathloss_map[r + dr, c + dc] = accumulated_pathloss



        self._fill_recur(pathloss_map, r + dr, c + dc, accumulated_distance, vector, route, threshold)

    def _calc_pathloss(self, src_pathloss, src_distance, dest_distance, weights):
        if src_distance == 0:
            return sum(weight * func(self._freq, dest_distance)
                       for weight, func in zip(weights, self._pathloss_functions))

        return src_pathloss * sum(weight * func(self._freq, dest_distance) / func(self._freq, src_distance)
                                  for weight, func in zip(weights, self._pathloss_functions))

    def _weights_at(self, r, c):
        return [landcover_map[r, c] for landcover_map in self._landcover_maps]

    def is_in(self, r, c):
        if 0 <= r < self.shape[0] and 0 <= c < self.shape[1]:
            return True

        return False


def main():
    landcover_map = np.loadtxt("data/landcover.txt", skiprows=6)
    antena_map = np.loadtxt("data/Antena.txt", skiprows=6)

    openland_map = np.zeros(landcover_map.shape)
    wood_with_leaves_map = np.zeros(landcover_map.shape)
    wood_without_leaves_map = np.zeros(landcover_map.shape)

    openland_map[landcover_map == 0] = 1
    wood_with_leaves_map[landcover_map == 1] = 1
    wood_without_leaves_map[landcover_map == 2] = 1

    # Egli model (openland or urban)
    openland_func = lambda freq, dist: 20 * np.log10(freq) + 40 * np.log10(dist / 1000) - 20 * np.log10(1) + 76.3 - 10 * np.log10(10)
    # COST235 with leaves
    wood_with_leaves_func = lambda freq, dist: 15.6 * freq ** (-0.009) * dist ** 0.26
    # COST235 without leaves
    wood_without_leaves_func = lambda freq, dist: 26.6 * freq ** (-0.2) * dist ** 0.5

    pathloss_calc = PathlossCalc()
    pathloss_calc.add_landcover(openland_map, openland_func)
    pathloss_calc.add_landcover(wood_with_leaves_map, wood_with_leaves_func)
    pathloss_calc.add_landcover(wood_without_leaves_map, wood_without_leaves_func)

    result = pathloss_calc.run(antena_map, 1000) # meter

    np.savetxt("results/result.txt", result, fmt="%.4f")

    fig, ax = plt.subplots()
    img = ax.imshow(result)
    plt.colorbar(img)
    plt.show()


if __name__ == "__main__":
    main()
