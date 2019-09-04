import numpy as np
from scipy.spatial import Voronoi, Delaunay
from shapely.geometry import Polygon
from collections import OrderedDict
from pyqtree import Index
import divide_into_beds as dib
from tqdm import tqdm
import warnings
import voronoi_diagram as vd
import util

warnings.simplefilter(action="ignore", category=RuntimeWarning)

path = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Schol\20190726_count_merged.shp"
dst = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Schol\20190726_count_merged_missed.shp"
    
def get_missing_points(plants, plot=False):
    convex_hull = util.get_convex_hull(np.array(plants))
    vor = Voronoi(plants)
    a, lengths = util.get_areas_and_lengths(vor, convex_hull)            
    #vor = vd.clip(vor, convex_hull.buffer(0.02), 0)
    ci = util.get_confidence_interval_areas(a)
    missed_regions, small_regions = vd.get_large_and_small_regions(vor, convex_hull, ci, lengths)         
    adjacent_missed_regions = vd.find_adjacent_polygons(missed_regions)
    slopes, dists = vd.get_slopes_and_distances_in_pairs_of_large_regions(vor, adjacent_missed_regions)
    ci_s = util.get_confidence_interval_slopes(slopes, 0.99)
    missed_points = vd.find_midpoints_in_pairs_of_large_regions(adjacent_missed_regions, vor, ci_s, dists)
    missed_points = missed_points + vd.find_missed_points_in_regions(adjacent_missed_regions, vor, ci_s, dists, spindex)
    if plot:
        vd.plot_voronoi_diagram(vor, np.array(missed_points), missed_regions, small_regions)
    return missed_points, slopes, ci_s, a, vor, adjacent_missed_regions

#%% plants
plants, src_driver, src_crs, src_schema = util.open_shape_file(path)
plants = np.array(plants)
beds = dib.divide(plants, plot=False)
missed_points_coord = []

t = 0
n = 5000
for bed in beds:
    t += int(np.ceil(len(bed)/n))
    
pbar = tqdm(total = t, desc="finding missing points")

#for j in range(len(beds)):
for j in range(1):
    bed = np.array(beds[j])

    overlap = 1000
    for i in range(1):
    #for i in range(int(np.ceil(len(bed)/n))):
        offset = n if (i + 1) * n < len(bed) else len(bed) - i * n
        offset = offset + overlap if i * n + offset + overlap < len(bed) else offset
        plants_i, mean_x_coord, mean_y_coord = util.readable_values(bed[i * n: i * n + offset, :])
        spindex = Index(bbox=(np.amin(plants_i[:,0]), np.amin(plants_i[:,1]), np.amax(plants_i[:,0]), np.amax(plants_i[:,1])))
        for plant in plants_i:
            spindex.insert(plant, bbox=(plant[0], plant[1], plant[0], plant[1]))
        
        missed_points, slopes, ci_s, a, vor, adjacent_missed_regions = get_missing_points(plants_i, True)
        missed_points_coord = missed_points_coord + list(util.readable_values_inv(np.array(missed_points), mean_x_coord, mean_y_coord))
        pbar.update(1)
        
missed_points_coord = util.remove_overlapping_points(np.array(missed_points_coord))
util.write_shape_file(dst, missed_points_coord, src_crs, src_driver, {'geometry': 'Point', 'properties': OrderedDict([('name', 'str')])})

#%%
points = util.readable_values(bed)
tri = Delaunay(points[0])
# =============================================================================
# plt.triplot(points[0][:,0], points[0][:,1], tri.simplices.copy())
# plt.scatter(points[0][:,0], points[0][:,1], color='r')
# plt.show()
# =============================================================================

from shapely.geometry import LineString
from scipy.signal import find_peaks

areas = []
lenghts = []
for t in tri.simplices:
    areas.append(Polygon([tri.points[t[0]], tri.points[t[1]], tri.points[t[2]]]).area)
    lenghts.append(LineString([tri.points[t[0]], tri.points[t[1]]]).length)
    lenghts.append(LineString([tri.points[t[0]], tri.points[t[2]]]).length)
    lenghts.append(LineString([tri.points[t[2]], tri.points[t[1]]]).length)
    
for i in range(len(lenghts) - 1, -1, -1):
    if lenghts[i] > 0.5:
        del lenghts[i]
        
hist = np.histogram(lenghts, bins=1000)
peaks, props = find_peaks(hist[0], distance = 100)


# =============================================================================
# plt.plot(peaks, hist[0][peaks], '*')
#     
# plt.plot(hist[0][0:1000])
# plt.show()
# =============================================================================

        
