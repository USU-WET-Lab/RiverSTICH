## RiverSTICH

_River-STICH_ (**S**urvey **T**ransect **I**nterpolation to reconstruct 3D **Ch**annels)

Updated on 6/10/2025

RiverSTICH converts traditional transect-based survey data into descriptive reach-scale attributes and variability function parameters that can then be used by [RiverBuilder](https://github.com/Pasternack-Lab/RiverBuilder) to construct a modular 3D synthetic river channel 


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* numpy
* pandas
* openpyxl
* simpledbf
* matplotlib
* scipy

<!-- USAGE EXAMPLES -->
## Usage Examples

Here, we present two examples using different types of XS survey data to demonstrate how RiverSTICH works.

#### Example 1. main_SFE_Leggett.py


<!---
- Input
    - Outputs from the first script (1.) 
    - Topography used to run the 2D hydrodynamic model in raster format
        - 1-m DEM with bathymetry
    - Manning's n and channel slope 
    - Cross-section lines (.shp)
    - Thalweg points for each cross-section lines (.shp)
    - For each flow condition,
        - Water surface elevation (WSE) in raster format
        - Downstream WSE and flow discharge 
- Output
    - A plot of the HAND-based rating curve and the family of rating curves obtained from the benchmark dataset

-->


# <!-- ![output3](/codes/SFE_Leggett_hand_param_calc/HAND_BM/SRCs_extended.png) -->

<!---
<p align="center" width="100%">
<img width="50%" src="/SFE_Leggett_hand_param_calc/HAND_BM/SRCs_extended.png" alt="output3">
</p>
--.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- CONTACT -->
## Contact

Anzy Lee anzy.lee@usu.edu

Project Link: [https://github.com/USU-CIROH/HAND-FIM_Assessment_public](https://github.com/USU-CIROH/HAND-FIM_Assessment_public)


<!-- ACKNOWLEDGMENTS 
## Acknowledgments


<p align="right">(<a href="#readme-top">back to top</a>)</p>

