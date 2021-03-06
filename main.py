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


def insert_number(path, number):
    base, ext = path.split(".")
    base += f"_{number}"
    return f"{base}.{ext}"


def main():
    # # Jinju
    # landcover_path = "data/jinju_landcover.txt"
    # antenna_path = "data/jinju_antenna.txt"
    # result_path = "results/result_jinju.asc"

    # Sihwa
    landcover_path = "data/landcover.txt"
    antenna_path = "data/Antenna.txt"
    result_path = "results/result_sihwa.asc"

    landcover_map, header = loadasc(landcover_path, with_header=True)
    antenna_map = loadasc(antenna_path)

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

    pathloss_maps = pathloss_calc.run_each(antenna_map, threshold=20000)

    for i, pathloss_map in enumerate(pathloss_maps):
        saveasc(insert_number(result_path, i), pathloss_map, header)

    # for i in range(min(len(pathloss_maps), 8)):
    #     plt.subplot(2, 4, i + 1).imshow(pathloss_maps[i])
    # plt.show()


if __name__ == "__main__":
    main()
