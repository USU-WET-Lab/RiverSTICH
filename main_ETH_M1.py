from ftransect import *
import shutil
import warnings

warnings.filterwarnings('ignore')

write_gcs = 0
gcs_dir = 'D:\\usu-RiverBuilder\\tools\\gcs'

# Define name of the site
site_name = 'M1'
flow_condition = 'base'

survey_path = os.path.abspath('./survey/%s' % site_name)
survey_topo_path = os.path.join(survey_path, 'M1.shp') # BED_30_dx100_dy1_1sqm
survey_wse_path = os.path.join(survey_path, 'M1_base.shp') # WSE from 2d model result

## Get WSE at station points
wse_point = survey_wse_path.replace('shp', 'dbf')

Contours_XYZ = XYZ_topo_to_contours(
    site_name, 100, wse_point, 'base', 1)

Contours, min_vertical_offset, method = generate_XZ_contours(
    Contours_XYZ, site_name, 'xyz_topo')  # mean of BFZ

Contours_series = generate_series(site_name, Contours, 'pchip' )
