import numpy as np
from collections import OrderedDict
from pyqtree import Index
import divide_into_beds as dib
import warnings
import voronoi_diagram as vd
import util_voronoi as util
import multiprocessing as mp
import time
import remove_outliers as ro

warnings.simplefilter(action="ignore", category=RuntimeWarning)

path = r"Z:\800 Operational\c01_verdonk\Rijweg stalling 1\20190709\1137\Plant_count\c01_verdonk-Rijweg stalling 1-201907091137-GR-count_KMV.shp"
dst = r"Z:\800 Operational\c01_verdonk\Rijweg stalling 1\20190709\1137\Plant_count\c01_verdonk-Rijweg stalling 1-201907091137-GR-count_KMV_missed.shp"

clip_voronoi = True
slope_field_mice = -0.7143475337058449
slope_field_schol = 0.5842593210109179
slope_field_weveroost = -0.8818719406757523
slope_field_jokevisser = -0.17404523952846995
slope_field_rijwegstalling1 = 0.36930600102426436
slope_field = slope_field_rijwegstalling1
n_its = 2
n_processes = 4
batch_size = 15000
overlap = 1000

dist_between_two_crops = 0

def get_missing_points(plants, spindex, plot=False, first_it=True, mean_dist=None):
    convex_hull = util.get_convex_hull(np.array(plants))
    vor = vd.Voronoi_diagram(plants)
    a, lengths = util.get_areas_and_lengths(vor, convex_hull)
    if clip_voronoi:        
        vor.clip_regions(convex_hull.buffer(0.03))
    ci = util.get_confidence_interval(a)
    missed_regions, small_regions = vd.get_large_and_small_regions(vor, convex_hull, ci, clip_voronoi)
    adjacent_missed_regions = vd.find_adjacent_polygons(missed_regions)
    slopes, dists = vd.get_slopes_and_distances_in_pairs_of_large_regions(vor, adjacent_missed_regions)
    
    missed_points = vd.find_midpoints_in_pairs_of_large_regions(adjacent_missed_regions, vor, slope_field or np.nanmedian(slopes), dists, first_it, mean_dist)
    missed_points = missed_points + vd.find_missed_points_in_regions(adjacent_missed_regions, vor, slope_field or np.nanmedian(slopes), dists, spindex, first_it, mean_dist)
    if plot:
        vd.plot_voronoi_diagram(vor, np.array(missed_points), missed_regions, small_regions)
    return missed_points, a, ci, adjacent_missed_regions, slopes, dists, vor

def worker(batch, first_it, mean_dist, i, total):
    start = time.time()
    missed_points_coord = []
    
    plants_i, mean_x_coord, mean_y_coord = util.readable_values(batch)
    spindex = Index(bbox=(np.amin(plants_i[:,0]), np.amin(plants_i[:,1]), np.amax(plants_i[:,0]), np.amax(plants_i[:,1])))
    for plant in plants_i:
        spindex.insert(plant, bbox=(plant[0], plant[1], plant[0], plant[1]))
    plants_i, _ = ro.remove_outliers(plants_i, slope_field)
    plants_i = np.array(plants_i)
    if first_it:
        missed_points, a, ci, adjacent_missed_regions, slopes, dists, vor = get_missing_points(plants_i, spindex, mean_dist = mean_dist)
    else:
        missed_points, a, ci, adjacent_missed_regions, slopes, dists, vor = get_missing_points(plants_i, spindex, first_it = False, mean_dist = mean_dist)
    missed_points_coord = missed_points_coord + list(util.readable_values_inv(np.array(missed_points), mean_x_coord, mean_y_coord))
    
    print("batch", i + 1, "/", total, "done by", mp.current_process(), "in", time.time() - start, "seconds")
    return missed_points_coord, np.nanmedian(slopes), np.nanmedian(dists)

if __name__ == "__main__":
    plants, src_driver, src_crs, src_schema = util.open_shape_file(path)
    missed_points_coord = []
    slope_means = []
    dists_means = []
    for z in range(n_its):
        beds = dib.divide(np.array(plants + missed_points_coord))
        p = mp.Pool(n_processes)
        
        batches = []
        for j in range(len(beds)):
            bed = np.array(beds[j])
            if len(bed) > 500:
                for i in range(int(np.ceil(len(bed)/batch_size))):
                    offset = batch_size if (i + 1) * batch_size < len(bed) else len(bed) - i * batch_size
                    offset = offset + overlap if i * batch_size + offset + overlap < len(bed) else offset
                    if offset > 500:
                        batches.append(bed[i * batch_size: i * batch_size + offset, :])
        
        time1= time.time()
        results = [p.apply_async(worker, (batches[i], z == 0, dist_between_two_crops or np.nanmedian(dists_means), i, len(batches))) for i in range(len(batches))]
        
        new_missed_points = []
        for res in results:
            try:
                new_missed_points += res.get()[0]
            except:
                print(res.get()[0])
            if z == 0:
                slope_means.append(res.get()[1])
                dists_means.append(res.get()[2])
        
        missed_points_coord = util.remove_overlapping_points(np.array(new_missed_points + missed_points_coord))
        print("found points in", time.time() - time1, "seconds")
        p.close()
        p = None

    util.write_shape_file(dst, missed_points_coord, src_crs, src_driver, {'geometry': 'Point', 'properties': OrderedDict([('name', 'str')])}, 'missed point')
