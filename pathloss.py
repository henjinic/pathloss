from copy import deepcopy

import numpy as np


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

        # negation
        temp_routes = deepcopy(self._routes)
        for (r, c), routes in temp_routes.items():
            for sign_r, sign_c in [(-1, 1), (1, -1), (-1, -1)]:
                if (r * sign_r, c * sign_c) in self._routes:
                    continue

                self._routes[r * sign_r, c * sign_c] = {(r * sign_r, c * sign_c): distance
                                                        for (r, c), distance in routes.items()}

        # transpose
        temp_routes = deepcopy(self._routes)
        for (r, c), routes in temp_routes.items():
            if (c, r) in self._routes:
                continue

            self._routes[c, r] = {(c, r): distance
                                  for (r, c), distance in routes.items()}

    @property
    def shape(self):
        return self._landcover_maps[0].shape

    def add_landcover(self, landcover_map, pathloss_function):
        self._landcover_maps.append(landcover_map)
        self._pathloss_functions.append(pathloss_function)

    def run(self, antenna_map, threshold=None):
        """`antena_map`: mask set 1 to antenna coordinations
        `threshold`: in meters
        """
        result = np.full(self.shape, 9999.0)

        for r, c in np.argwhere(antenna_map):
            pathloss_map = self._fill(r, c, threshold)
            result = np.minimum(result, pathloss_map)

        result[result == 9999] = -1

        return result

    def _fill(self, r, c, threshold):
        result = np.full(self.shape, 9999.0)
        result[r, c] = 0

        for vector, route in self._routes.items():
            self._fill_recur(result, r, c, 0, vector, route, threshold)

        return result

    def _fill_recur(self, pathloss_map, r, c, src_distance, vector, route, threshold):
        dr, dc = vector

        if threshold is not None and src_distance > threshold:
            return pathloss_map

        if not self._is_in(r + dr, c + dc):
            return pathloss_map

        accumulated_pathloss = pathloss_map[r, c]
        accumulated_distance = src_distance

        for (mr, mc), distance in route.items():
            accumulated_pathloss = self._calc_pathloss(accumulated_pathloss, accumulated_distance,
                                                       accumulated_distance + distance,
                                                       self._weights_at(r + mr, c + mc))
            accumulated_distance += distance

        pathloss_map[r + dr, c + dc] = accumulated_pathloss

        self._fill_recur(pathloss_map, r + dr, c + dc,
                         accumulated_distance, vector, route, threshold)

    def _calc_pathloss(self, src_pathloss, src_distance, dest_distance, weights):
        if src_distance == 0:
            return sum(weight * func(self._freq, dest_distance)
                       for weight, func in zip(weights, self._pathloss_functions))

        return src_pathloss * sum(weight * func(self._freq, dest_distance)
                                  / func(self._freq, src_distance)
                                  for weight, func in zip(weights, self._pathloss_functions))

    def _weights_at(self, r, c):
        weights = [landcover_map[r, c] for landcover_map in self._landcover_maps]

        if sum(weights) == 0:
            return weights

        return [weight / sum(weights) for weight in weights]

    def _is_in(self, r, c):
        if 0 <= r < self.shape[0] and 0 <= c < self.shape[1]:
            return True

        return False


def main():
    pass


if __name__ == "__main__":
    main()
