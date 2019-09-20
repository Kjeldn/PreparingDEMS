import util_voronoi as util
import divide_into_beds as dib
import numpy as np
from scipy.spatial import voronoi_plot_2d
from pyqtree import Index
import voronoi_diagram as vd
import matplotlib.pyplot as plt
import remove_outliers as ro

path = r"Z:\800 Operational\c08_biobrass\Duikerweg\20190909\1651\clipped_points._merged.shp"

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
            
plants_i, mean_x_coord, mean_y_coord = util.readable_values(batches[10])
spindex = Index(bbox=(np.amin(plants_i[:,0]), np.amin(plants_i[:,1]), np.amax(plants_i[:,0]), np.amax(plants_i[:,1])))
for plant in plants_i:
    spindex.insert(plant, bbox=(plant[0], plant[1], plant[0], plant[1]))
#missed_points, a, ci, adjacent_missed_regions, slopes, dists, vor = voronoi.get_missing_points(plants_i, spindex, plot=True, mean_dist=0.08)
convex_hull = util.get_convex_hull(np.array(plants_i))
id_plants, non_id_plants = ro.remove_outliers(plants_i, -0.706638627706045)
vor = vd.Voronoi_diagram(id_plants)
a, lengths = util.get_areas_and_lengths(vor, convex_hull)
vor.clip_regions(convex_hull.buffer(0.03))
ci = util.get_confidence_interval(a)
missed_regions, small_regions = vd.get_large_and_small_regions(vor, convex_hull, ci, True)         
adjacent_missed_regions = vd.find_adjacent_polygons(missed_regions)
slopes, dists = vd.get_slopes_and_distances_in_pairs_of_large_regions(vor, adjacent_missed_regions)

missed_points = vd.find_midpoints_in_pairs_of_large_regions(adjacent_missed_regions, vor, -0.7143475337058449, dists, True, 0.08)
missed_points = missed_points + vd.find_missed_points_in_regions(adjacent_missed_regions, vor, -0.7143475337058449, dists, spindex, True, 0.08)

voronoi_plot_2d(vor, show_vertices=False, show_points=False)
plt.scatter(np.array(id_plants)[:,0], np.array(id_plants)[:,1], color='b')
plt.scatter(np.array(non_id_plants)[:,0], np.array(non_id_plants)[:,1], color='purple', marker='x')
plt.scatter(np.array(missed_points)[:,0], np.array(missed_points)[:,1], color='k', marker='v')
for r in missed_regions:
    plt.fill(*zip(*r.vc), color="r", alpha = 0.5)
    
for r in small_regions:
    plt.fill(*zip(*r.vc), color="g", alpha = 1)
    
plt.show()
#%%
import matplotlib.pyplot as plt
points = np.array([vor.points[(np.where(vor.point_region == poly.id))[0][0]] for poly in adjacent_missed_regions[22]])
lines = util.find_points_in_line(list(adjacent_missed_regions[22]), -0.17404523952846995, vor)
plt.scatter(points[:,0], points[:,1])
for line in lines:
    plt.plot([p[0] for p in line],[p[1] for p in line])
