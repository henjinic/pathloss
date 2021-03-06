import math
import numpy as np
from statistics import mean


# COST235 without leaves
def pathloss_wood_withoutleaves(freq, dist): # frequency in MHz, distance in meter
    return 26.6 * freq ** (-0.2) * dist ** 0.5

# COST235 with leaves
def pathloss_wood_withleaves(freq, dist):
    return 15.6 * freq ** (-0.009) * dist ** 0.26

# Egli model (openland or urban)
def pathloss_openland(freq, dist):
    return 20*math.log10(freq) + 40*math.log10(dist/1000) - 20*math.log10(1) + 76.3 - 10*math.log10(10)


def landcover_to_func(landcover):
    if landcover == 0:
        return pathloss_openland
    elif landcover == 1:
        return pathloss_wood_withleaves
    elif landcover == 2:
        return pathloss_wood_withoutleaves
    else:
        raise Exception("invalidcode")


class LandcoverMap:

    def __init__(self):
        self._map = np.loadtxt("data/landcover.txt")

    @property
    def shape(self):
        return self._map.shape

    def codes_between(self, src, dest):
        codes = []

        if src[0] == dest[0] and src[1] > dest[1]: # left
            codes.append(self._map[src[0] - 1, src[1] - 1])
            codes.append(self._map[src[0], src[1] - 1])
        elif src[0] == dest[0] and src[1] < dest[1]: # right
            codes.append(self._map[src[0] - 1, src[1]])
            codes.append(self._map[src[0], src[1]])
        elif src[0] > dest[0] and src[1] == dest[1]: # up
            codes.append(self._map[src[0] - 1, src[1] - 1])
            codes.append(self._map[src[0] - 1, src[1]])
        elif src[0] < dest[0] and src[1] == dest[1]: # down
            codes.append(self._map[src[0], src[1] - 1])
            codes.append(self._map[src[0], src[1]])

        elif src[0] < dest[0] and src[1] < dest[1]: # lower right
            codes.append(self._map[src[0], src[1]])
        elif src[0] > dest[0] and src[1] < dest[1]: # upper right
            codes.append(self._map[src[0] - 1, src[1]])
        elif src[0] > dest[0] and src[1] > dest[1]: # upper left
            codes.append(self._map[src[0] - 1, src[1] - 1])
        elif src[0] < dest[0] and src[1] > dest[1]: # lower left
            codes.append(self._map[src[0], src[1] - 1])

        return codes


class PathlossFiller:

    def __init__(self, lora_freq, cell_size):
        self._landcover_map = LandcoverMap()
        self._lora_freq = lora_freq
        self._cell_size = cell_size
        self._result_map = np.zeros(self._landcover_map.shape) + 9999

    @property
    def height(self):
        return self._result_map.shape[0]

    @property
    def width(self):
        return self._result_map.shape[1]

    def fill(self, r, c):
        self._fill_nth_recur(r, c, 0, (0, 1))
        self._fill_nth_recur(r, c, 0, (1, 0))
        self._fill_nth_recur(r, c, 0, (0, -1))
        self._fill_nth_recur(r, c, 0, (-1, 0))

        self._fill_nth_recur(r, c, 0, (-1, -1))
        self._fill_nth_recur(r, c, 0, (-1, 1))
        self._fill_nth_recur(r, c, 0, (1, -1))
        self._fill_nth_recur(r, c, 0, (1, 1))

        return self._result_map

    def _fill_nth_recur(self, r, c, n, direction):
        target_coord = r + n * direction[0], c + n * direction[1]

        if not self._is_valid_result_coord(*target_coord):
            return

        if n == 0:
            result = 0
        else:
            pre_coord = r + (n - 1) * direction[0], c + (n - 1) * direction[1]

            codes = self._landcover_map.codes_between(pre_coord, target_coord)
            losses = [landcover_to_func(code)(self._lora_freq, self._cell_size * n * math.dist([0, 0], direction)) for code in codes]

            if n == 1:
                result = mean(losses)
            else:
                pre_losses = [landcover_to_func(code)(self._lora_freq, self._cell_size * (n - 1) * math.dist([0, 0], direction)) for code in codes]
                inc_rates = [loss / pre_loss for loss, pre_loss in zip(losses, pre_losses)]

                result = self._result_map[pre_coord] * mean(inc_rates)

        self._result_map[target_coord] = result

        self._fill_nth_recur(r, c, n + 1, direction)

    def _is_valid_result_coord(self, r, c):
        if r < 0 or c < 0:
            return False

        if r >= self.height or c >= self.width:
            return False

        return True


class PathlossMap:

    def __init__(self, lora_freq, cell_size):
        self._landcover_map = LandcoverMap()
        self._lora_freq = lora_freq
        self._cell_size = cell_size
        self._result_map = np.zeros(self._landcover_map.shape) + 9999

    @property
    def height(self):
        return self._result_map.shape[0]

    @property
    def width(self):
        return self._result_map.shape[1]

    def fill(self, r, c):
        if not self._is_valid_receiver_coord(r, c):
            raise Exception("Out of index")

        self._result_map = np.minimum(self._result_map, PathlossFiller(self._lora_freq, self._cell_size).fill(r, c))

    def print_map(self):
        print(self._result_map)

    def save_result(self, path):
        np.savetxt(path, self._result_map, "%7.2f")

    def _is_valid_receiver_coord(self, r, c):
        if r < 1 or c < 1:
            return False

        if r >= self.height or c >= self.width:
            return False

        return True


def main():

    for r, c in [[1, 1], [5, 5], [8, 11]]:
        pathloss_map = PathlossMap(lora_freq=900, cell_size=30)

        pathloss_map.fill(r, c)

        pathloss_map.save_result(f"results/result{r}_{c}.txt")

    pathloss_map = PathlossMap(lora_freq=900, cell_size=30)

    pathloss_map.fill(1, 1)
    pathloss_map.fill(5, 5)
    pathloss_map.fill(8, 11)

    pathloss_map.print_map()
    pathloss_map.save_result("results/result.txt")


if __name__ == "__main__":
    main()
