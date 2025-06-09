from ftransect import *
import warnings

warnings.filterwarnings('ignore')

# Define name of the site
site_name = 'SFE_Leggett'

# Open field survey data
wb, ws, path_survey = open_field_survey_data(site_name)

# Generate Thalweg elevation
Thalweg_Z = ThalZ_profile(site_name, True)

# Generate XYZ coordinate of auto-level survey data
# 1. Transform XYZ by assuming asymmetric channel (X-Y)
Contours_XYZ, HT, HT_all, TR = trans_XY_contours(site_name, 'Asym')

# 2. Find the best-fit regression line for contours (X-Z)
Contours, min_vertical_offset, method = generate_XZ_contours(
    Contours_XYZ, site_name, 'autolevel', Thalweg_Z=Thalweg_Z)

# Generate series for thalweg elevation and contours
Contours_series = generate_series(site_name, Contours, 'pchip')

# Calculate channel attributes
channel_att = calculate_att_table(Contours_series)
channel_att.to_excel('./output/%s/channel_attributes.xlsx' % site_name)

