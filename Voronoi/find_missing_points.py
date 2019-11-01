"""
This script finds missed points based on the plants count shapefile. It then writes a new shapefile as dst.
"""

import numpy as np
from collections import OrderedDict
import divide_into_beds as dib
import warnings
import voronoi_diagram as vd
import util_voronoi as util
import multiprocessing as mp
import time
import remove_outliers as ro

warnings.simplefilter(action="ignore", category=RuntimeWarning)

path = r"Z:\800 Operational\c01_verdonk\Rijweg stalling 2\20190709\1156\Plant_count\bed2.gpkg"
dst = r"Z:\800 Operational\c01_verdonk\Rijweg stalling 2\20190709\1156\Plant_count\bed2_missed.gpkg"

clip_voronoi = True #clip voronoi polygons on the border to find missing points on the border
slope_field_mice = -0.7143475337058449
slope_field_schol = 0.5842593210109179
slope_field_weveroost = -0.8818719406757523
slope_field_jokevisser = -0.17404523952846995
slope_field_rijwegstalling1 = 0.36930600102426436
slope_field_rijwegstalling2 = -0.744001707930606
slope_field = slope_field_rijwegstalling2 #slope of the field, can be computed from slope_field.py, if 0 it is not used
n_its = 3 #number of iterations
n_processes = 4 #number of processes used
batch_size = [2000, 6000, 8000] #batch sizes used in each iteration, len should be n_its
overlap = 1000 #number of points overlap between batches

dist_between_two_crops = 0.101646283 #distance between two crops such that one missed point is between them, can be computed with slope_field.py, if 0 it is not used
delta1 = [0.02, 0.04, 0.04] #delta1 used in each iteration, used in util_voronoi.ci_slopes. should be 0.01 < delta1 < 0.06. Decrease if skewed lines are found.
delta2 = [0.03, 0.06, 0.06] #delta2 used in each iteration, used in util_voronoi.ci_slopes. should be 0.01 < delta1 < delta2 < 0.06. Decrease if skewed lines are found.
remove_outliers = True #Remove outliers from plant count (points which are not on a ridge) before finding missed points

def get_missing_points(plants, d1, d2, plot=False, first_it=True, mean_dist=None):
    convex_hull = util.get_convex_hull(np.array(plants))
    vor = vd.Voronoi_diagram(plants)
    a, _ = util.get_areas_and_lengths(vor, convex_hull)
    if clip_voronoi:
        vor.clip_regions(convex_hull.buffer(0.03))
    ci = util.get_confidence_interval(a)
    missed_regions, small_regions = vd.get_large_and_small_regions(vor, convex_hull, ci, clip_voronoi)
    adjacent_missed_regions = vd.find_adjacent_polygons(missed_regions)
    slopes, dists = vd.get_slopes_and_distances_in_pairs_of_large_regions(vor, adjacent_missed_regions)
    
    if slope_field:
        s = slope_field
    else:
        if slopes:
            s = np.nanmedian(slopes)
        else:
            s = 0
            
    if dist_between_two_crops:
        d = dist_between_two_crops
    else:
        if mean_dist and not np.isnan(mean_dist) and first_it:
            d = mean_dist
        elif dists:
            d = np.nanmedian(dists)
        else:
            d = 0
    
    missed_points = vd.find_midpoints_in_pairs_of_large_regions(adjacent_missed_regions, vor, s, d)
    missed_points = missed_points + vd.find_missed_points_in_regions(adjacent_missed_regions, vor, s, d, delta1 = d1, delta2 = d2)
    if plot:
        vd.plot_voronoi_diagram(vor, np.array(missed_points), missed_regions, small_regions)
    return missed_points, a, ci, adjacent_missed_regions, slopes, dists, vor

def worker(batch, first_it, mean_dist, i, total, d1, d2):
    start = time.time()
    missed_points_coord = []
    
    plants_i, mean_x_coord, mean_y_coord = util.readable_values(batch)
    if slope_field and remove_outliers:
        plants_i = np.array(ro.remove_outliers(plants_i, slope_field)[0])

    if first_it:
        missed_points, a, ci, adjacent_missed_regions, slopes, dists, vor = get_missing_points(plants_i, mean_dist = mean_dist, d1=d1, d2=d2)
    else:
        missed_points, a, ci, adjacent_missed_regions, slopes, dists, vor = get_missing_points(plants_i, first_it = False, mean_dist = mean_dist, d1=d1, d2=d2)
    missed_points_coord = missed_points_coord + list(util.readable_values_inv(np.array(missed_points), mean_x_coord, mean_y_coord))
    
    print("batch", i + 1, "/", total, "done by", mp.current_process(), "in", time.time() - start, "seconds")
    return missed_points_coord, np.nanmedian(slopes), np.nanmedian(dists), a, ci, adjacent_missed_regions, slopes, dists, vor, plants_i

if __name__ == "__main__":
    plants, src_driver, src_crs, src_schema = util.open_shape_file(path)
    missed_points_coord = []
    slope_means = []
    dists_means = []
    for z in range(n_its):
        beds = dib.divide(np.array(plants + missed_points_coord))
#        beds = [np.array(plants + missed_points_coord)] #if dividing into beds is not necessary
        p = mp.Pool(n_processes)
        
        batches = []
        for j in range(len(beds)):
            bed = np.array(beds[j])
            if len(bed) > 500:
                for i in range(int(np.ceil(len(bed)/batch_size[z]))):
                    offset = batch_size[z] if (i + 1) * batch_size[z] < len(bed) else len(bed) - i * batch_size[z]
                    offset = offset + overlap if i * batch_size[z] + offset + overlap < len(bed) else offset
                    if offset > 500:
                        batches.append(bed[i * batch_size[z]: i * batch_size[z] + offset, :])
        
        time1= time.time()
        
        results = [p.apply_async(worker, (batches[i], z == 0, np.nanmedian(dists_means), i, len(batches), delta1[z], delta2[z])) for i in range(len(batches))]
        
        new_missed_points = []
        for res in results:
            try:
                new_missed_points += res.get()[0]
            except:
                print(res.get()[0])
            if z == 0:
                slope_means.append(res.get()[1])
                dists_means.append(res.get()[2])
        
        missed_points_coord += util.remove_overlapping_points(np.array(new_missed_points), np.array(plants + missed_points_coord))
        print("found points in", time.time() - time1, "seconds")
        p.close()
        p = None

    util.write_shape_file(dst, missed_points_coord, src_crs, src_driver, {'geometry': 'Point', 'properties': OrderedDict([('name', 'str')])}, 'missed point')