import numpy as np
from matplotlib import pyplot as plt

from pathloss import PathlossCalc


def loadasc(path, with_header=False):
    result = np.loadtxt(path, skiprows=6)

    if with_header:
        return result, np.loadtxt(path, dtype=str, max_rows=6)
    else:
        return result


def saveasc(path, data, header):
    header_string = "\n".join(f"{key:<14}{value}" for key, value in header)
    np.savetxt(path, data, fmt="%.4f", header=header_string, comments="")


def openland_func(freq, dist):
    """### Egli model (openland or urban)"""
    return 20 * np.log10(freq) + 40 * np.log10(dist / 1000) - 20 * np.log10(1) + 76.3 - 10 * np.log10(10)


def wood_with_leaves_func(freq, dist):
    """### COST235 with leaves"""
    return 15.6 * freq ** (-0.009) * dist ** 0.26


def wood_without_leaves_func(freq, dist):
    """### COST235 without leaves"""
    return 26.6 * freq ** (-0.2) * dist ** 0.5


def main():
    landcover_map, header = loadasc("data/jinju_landcover.txt", with_header=True)
    antenna_map = loadasc("data/jinju_antenna.txt")

    openland_map = np.zeros(landcover_map.shape)
    wood_with_leaves_map = np.zeros(landcover_map.shape)
    wood_without_leaves_map = np.zeros(landcover_map.shape)

    openland_map[landcover_map == 0] = 1
    wood_with_leaves_map[landcover_map == 1] = 1
    wood_without_leaves_map[landcover_map == 2] = 1

    pathloss_calc = PathlossCalc(cell_size=30, freq=900)
    pathloss_calc.add_landcover(openland_map, openland_func)
    pathloss_calc.add_landcover(wood_with_leaves_map, wood_with_leaves_func)
    pathloss_calc.add_landcover(wood_without_leaves_map, wood_without_leaves_func)

    result = pathloss_calc.run(antenna_map, threshold=20000)

    saveasc("results/result_jinju.asc", result, header)

    img = plt.imshow(result)
    plt.colorbar(img)
    plt.show()


if __name__ == "__main__":
    main()
