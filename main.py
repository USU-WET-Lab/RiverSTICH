from ftransect import *
import shutil
import warnings
from openpyxl import *
from openpyxl.utils.dataframe import dataframe_to_rows

warnings.filterwarnings('ignore')

# Define name of the site
site_name = 'SFE_Leggett'

# Open field survey data
wb, ws, dir_survey = open_field_survey_data(site_name)

# Generate Thalweg elevation
Thalweg_Z = ThalZ_profile(site_name, True)

# Generate XYZ coordinate of surveyed data
# 1. Transform XYZ by assuming asymmetric channel (X-Y)
XYZ, HT, HT_all, TR = trans_XY_contours(site_name, 'Asym')

# 2. Find the best-fit regression line for contours (X-Z)
contours, min_vertical_offset, method = generate_XZ_contours(XYZ, Thalweg_Z, site_name)

# Generate series for thalweg elevation and contours
contours_series = generate_series(site_name, contours, interp_method='pchip')

# Calculate channel attributes
channel_att = calculate_att_table(contours_series)
channel_att.to_excel('./output/%s_channel_attributes.xlsx' % site_name)

