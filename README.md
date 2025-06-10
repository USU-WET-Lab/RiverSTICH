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

#### Example 1. Auto-level survey (main_SFE_Leggett.py)

- Input
    - Field survey data sheet (/survey/SFE_Leggett/SFE_Leggett.xlsx)
![Figure 1. Equal-space transect survey](/survey/SFE_Leggett/survey1.png)
![Figure 2. Longitudinal profile](/survey/SFE_Leggett/survey2.png)
![Figure 3. Additional riffle crests and pool troughs transect survey](/survey/SFE_Leggett/survey3.png)
- Output
    - X-Y contour plot (before and after transformation, /output/SFE_Leggett/XY_before_transformation.png and XY.png)
![Figure 4. X-Y contour plot, before transformation](/output/SFE_Leggett/XY_before_transformation.png) 
![Figure 5. X-Y contour plot, after transformation](/output/SFE_Leggett/XY.png)
    - X-Y and X-Z interpolated contour plot 
![Figure 6. X-Y and X-Z interpolated contour plot](/output/SFE_Leggett/XYZ_contours.png)
    - A channel attribute table of RiverSTICH channel (/output/SFE_Leggett/channel_attributes.xlsx)
    - Interpolated contour series, which will be used for RiverBuilder channel generation (/output/SFE_Leggett/SFE_Leggett_RB_metrics.xlsx)

# <!-- ![output3](/codes/SFE_Leggett_hand_param_calc/HAND_BM/SRCs_extended.png) -->

<!---
<p align="center" width="100%">
<img width="50%" src="/SFE_Leggett_hand_param_calc/HAND_BM/SRCs_extended.png" alt="output3">
</p>
-->


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

