from ftransect import *
import shutil
import warnings


warnings.filterwarnings('ignore')

# Define name of the site
site_name = 'M1'

# Open field survey data
survey_path = os.path.abspath('./survey/%s' % site_name)
survey_topo_path = os.path.join(survey_path, 'M1.shp') # BED_30_dx100_dy1_1sqm
survey_wse_path = os.path.join(survey_path, 'M1_base.shp') # WSE from 2d model result

## Get WSE at station points
wse_point = survey_wse_path.replace('shp', 'dbf')

Contours_XYZ = XYZ_topo_to_contours(
    site_name, 100, wse_point, 1)
# Note: Points with x < x_cutoff was removed for erroneous WSE values
# Note: y_flipped was set 1 as the y coordinate values were flipped

# Generate XYZ coordinate of surveyed data with XYZ coordinate
Contours, min_vertical_offset, method = generate_XZ_contours(
    Contours_XYZ, site_name, 'xyz_topo')

# Generate series for thalweg elevation and contours
Contours_series = generate_series(Contours, site_name, 'pchip')

# Write a GCS file
write_GCS(Contours_series, site_name)

# Calculate channel attributes
channel_att = calculate_att_table(Contours_series)
channel_att.to_excel('./output/%s/channel_attributes.xlsx' % (site_name))