import numpy as np
from collections import OrderedDict
from pyqtree import Index
import divide_into_beds as dib
from tqdm import tqdm
import warnings
import voronoi_diagram as vd
import util
import multiprocessing as mp
from functools import partial
import time

warnings.simplefilter(action="ignore", category=RuntimeWarning)

path = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Schol\20190726_count_merged.shp"
dst = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Schol\20190726_count_merged_missed.shp"
clip_voronoi = True
slope_field_mice = -0.7143475337058449
slope_field = 0.5842593210109179
n_ints = 2
n_processes = 4
    
def get_missing_points(plants, spindex, plot=False, first_it=True, mean_dist=None):
    convex_hull = util.get_convex_hull(np.array(plants))
    vor = vd.Voronoi_diagram(plants)
    a, lengths = util.get_areas_and_lengths(vor, convex_hull)
    ci = util.get_confidence_interval(a)
    missed_regions, small_regions = vd.get_large_and_small_regions(vor, convex_hull, ci, lengths, clip_voronoi)         
    adjacent_missed_regions = vd.find_adjacent_polygons(missed_regions)
    slopes, dists = vd.get_slopes_and_distances_in_pairs_of_large_regions(vor, adjacent_missed_regions)
    if clip_voronoi:        
        vor.clip_regions(convex_hull.buffer(0.04))
    ci_s = (slope_field - 0.03, slope_field + 0.03)
    missed_points = vd.find_midpoints_in_pairs_of_large_regions(adjacent_missed_regions, vor, ci_s, dists, first_it, mean_dist)
    missed_points = missed_points + vd.find_missed_points_in_regions(adjacent_missed_regions, vor, ci_s, dists, spindex, first_it, mean_dist)
    if plot:
        vd.plot_voronoi_diagram(vor, np.array(missed_points), missed_regions, small_regions)
    return missed_points, a, ci, adjacent_missed_regions, slopes, dists, ci_s, vor

def worker(batch, first_it, mean_dist, i):
    start = time.time()
    missed_points_coord = []
    
    plants_i, mean_x_coord, mean_y_coord = util.readable_values(batch)
    spindex = Index(bbox=(np.amin(plants_i[:,0]), np.amin(plants_i[:,1]), np.amax(plants_i[:,0]), np.amax(plants_i[:,1])))
    for plant in plants_i:
        spindex.insert(plant, bbox=(plant[0], plant[1], plant[0], plant[1]))
    if first_it:
        missed_points, a, ci, adjacent_missed_regions, slopes, dists, ci_s, vor = get_missing_points(plants_i, spindex)
    else:
        missed_points, a, ci, adjacent_missed_regions, slopes, dists, ci_s, vor = get_missing_points(plants_i, spindex, first_it = False, mean_dist = mean_dist)
    missed_points_coord = missed_points_coord + list(util.readable_values_inv(np.array(missed_points), mean_x_coord, mean_y_coord))
    
    print("batch", i, "done by", mp.current_process(), "in", time.time() - start, "seconds")
    return missed_points_coord, np.nanmedian(slopes), np.nanmedian(dists)

if __name__ == "__main__": 
    plants, src_driver, src_crs, src_schema = util.open_shape_file(path)
    missed_points_coord = []
    for z in range(n_ints):        
        slope_means = []
        dists_means = []
        n = 5000
        overlap = 1000
        
        beds = dib.divide(np.array(plants + missed_points_coord))
        p = mp.Pool(n_processes)
        
        batches = []
        for j in range(len(beds)):
            bed = np.array(beds[j])
            if len(bed) > 500:
                for i in range(int(np.ceil(len(bed)/n))):
                    offset = n if (i + 1) * n < len(bed) else len(bed) - i * n
                    offset = offset + overlap if i * n + offset + overlap < len(bed) else offset
                    batches.append(bed[i * n: i * n + offset, :])
        
        time1= time.time()
        results = [p.apply_async(worker, (batches[i], z == 0, np.nanmedian(dists_means), i)) for i in range(len(batches))]
        
        new_missed_points = []
        for res in results:
            new_missed_points += res.get()[0]
            slope_means.append(res.get()[1])
            dists_means.append(res.get()[2])
        
        missed_points_coord = util.remove_overlapping_points(np.array(new_missed_points))
        print("found points in", time.time() - time1, "seconds")
        print(dists_means)

    util.write_shape_file(dst, missed_points_coord, src_crs, src_driver, {'geometry': 'Point', 'properties': OrderedDict([('name', 'str')])})
