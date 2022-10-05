# pathloss

`pathloss` is a module for calculating grid-based pathloss algorithm.

## Getting Started
1. Make sure you have `numpy>=1.21.2` installed.
2. Place `pathloss.py` in a path that Python can find.

## Usage
1. Define pathloss functions for each landcover
    ```py
    def pathloss_function_for_landcover_1(freq, dist):
        return ...

    def pathloss_function_for_landcover_2(freq, dist):
        return ...
    ```
2. Load weight maps for each landcover and antenna position map (2d list or ndarray)
    ```py
    weight_map_for_landcover_1 = ...
    weight_map_for_landcover_2 = ...
    antenna_map = ...
    ```
3. Run
    ```py
    from pathloss import PathlossCalc

    pathloss_calc = PathlossCalc(cell_size=30, freq=900)

    pathloss_calc.add_landcover(pathloss_function_for_landcover_1, weight_map_for_landcover_1)
    pathloss_calc.add_landcover(pathloss_function_for_landcover_2, weight_map_for_landcover_2)

    result = pathloss_calc.run(antenna_map, threshold=30)
    ```

## Contributors
The following is a list of the researchers who have helped to improve pathloss by constructing ideas and contributing code.
1. Chan Park (박찬)
2.
3.
4.
5. Hyeonjin Kim (김현진)

## License
The pathloss is distributed under the Spatial science lab in University of Seoul(UOS), a permissive open-source (free software) license.

![](https://lauos.or.kr/wp-content/uploads/2022/02/융합연구실로고.png)
