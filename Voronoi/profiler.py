import util
import divide_into_beds as dib
import numpy as np
import voronoi
from pyqtree import Index

path = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Schol\20190726_count_merged.shp"

plants, src_driver, src_crs, src_schema = util.open_shape_file(path)
missed_points_coord = []
      
n = 5000
overlap = 1000

beds = dib.divide(np.array(plants + missed_points_coord))

batches = []
for j in range(len(beds)):
    bed = np.array(beds[j])
    if len(bed) > 500:
        for i in range(int(np.ceil(len(bed)/n))):
            offset = n if (i + 1) * n < len(bed) else len(bed) - i * n
            offset = offset + overlap if i * n + offset + overlap < len(bed) else offset
            batches.append(bed[i * n: i * n + offset, :])
            
plants_i, mean_x_coord, mean_y_coord = util.readable_values(batches[22])
spindex = Index(bbox=(np.amin(plants_i[:,0]), np.amin(plants_i[:,1]), np.amax(plants_i[:,0]), np.amax(plants_i[:,1])))
for plant in plants_i:
    spindex.insert(plant, bbox=(plant[0], plant[1], plant[0], plant[1]))
missed_points, a, ci, adjacent_missed_regions, slopes, dists, vor = voronoi.get_missing_points(plants_i, spindex)
#%%
# =============================================================================
# import matplotlib.pyplot as plt
# points = np.array([vor.points[(np.where(vor.point_region == poly.id))[0][0]] for poly in adjacent_missed_regions[22]])
# lines = util.find_points_in_line(list(adjacent_missed_regions[22]), 0.5842593210109179, vor)
# plt.scatter(points[:,0], points[:,1])
# for line in lines:
#     plt.plot([p[0] for p in line],[p[1] for p in line])
# =============================================================================
