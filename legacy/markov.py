import numpy as np


class ProbMap:

    def __init__(self):
        self._prob_matrix = [
            [0.55, 0.35, 0.1],
            [0.15, 0.25, 0.6],
            [0.5, 0.2, 0.3]
        ]

        self._lc_map = np.loadtxt("data/landcover.txt", dtype=int)

    @property
    def shape(self):
        return self._lc_map.shape

    def get(self, from_coord, to_coord):
        return self._prob_matrix[self._lc_map[from_coord]][self._lc_map[to_coord]]

    def is_valid_coord(self, coord):
        r, c = coord

        if r < 0 or c < 0:
            return False

        if r >= self.shape[0] or c >= self.shape[1]:
            return False

        return True

    def __str__(self):
        return self._lc_map.__str__()


class SpreadMap:

    def __init__(self, start_coord, value):
        self._prob_map = ProbMap()

        self._result_map = np.zeros(self._prob_map.shape)
        self._result_map[start_coord] = value

        self._start_coord = start_coord

    @property
    def result_map(self):
        return self._result_map

    def spread(self):
        r, c = self._start_coord
        vectors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        to_coords = [(r + dr, c + dc) for dr, dc in vectors]
        to_coords = [(r, c) for r, c in to_coords if self._prob_map.is_valid_coord((r, c))]

        self.spread_recur(to_coords)

    def spread_recur(self, to_coords):
        next_coords = set()

        for to_coord in to_coords:
            next_coords = next_coords.union(self._update(to_coord))

        if not next_coords:
            return

        self.spread_recur(next_coords)

    def _update(self, to_coord):
        next_coords = []
        values = []

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            from_coord = (to_coord[0] + dr, to_coord[1] + dc)

            if not self._prob_map.is_valid_coord(from_coord):
                continue

            if self._result_map[from_coord] == 0:
                next_coords.append(from_coord)
                continue

            prob = self._prob_map.get(from_coord, to_coord)
            values.append(self._result_map[from_coord] * prob)

        self._result_map[to_coord] = max(values)

        return next_coords

    def __str__(self):
        return self._result_map.__str__()


def main():
    start_coords = [
        (2, 3),
        (5, 5),
        (8, 1)
    ]
    spread_maps = [SpreadMap(coord, 100) for coord in start_coords]

    for map in spread_maps:
        map.spread()

    merged_map = spread_maps[0].result_map
    for map in spread_maps[1:]:
        merged_map = np.maximum(merged_map, map.result_map)

    print(merged_map)

    np.savetxt("results/markov_result.txt", merged_map, "%8.4f")


if __name__ == "__main__":
    main()
