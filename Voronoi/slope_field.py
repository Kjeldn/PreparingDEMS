import util_voronoi as util
import voronoi_diagram as vd
import numpy as np
import multiprocessing as mp
import divide_into_beds as dib

clip_voronoi = True
path = r"C:\Users\VanBoven\Documents\DL Plant Count\PLANT COUNT - c08_biobrass-AZ74-201905171650-GR\POINTS.shp"
batch_size = 15000
overlap = 1000
n_processes = 4

def get_slopes(plants, index, size):
    plants_i, mean_x_coord, mean_y_coord = util.readable_values(plants)
    convex_hull = util.get_convex_hull(np.array(plants_i))
    vor = vd.Voronoi_diagram(plants_i)
    a, lengths = util.get_areas_and_lengths(vor, convex_hull)
    if clip_voronoi:        
        vor.clip_regions(convex_hull.buffer(0.03))
    ci = util.get_confidence_interval(a)
    missed_regions, small_regions = vd.get_large_and_small_regions(vor, convex_hull, ci, clip_voronoi)         
    adjacent_missed_regions = vd.find_adjacent_polygons(missed_regions)
    slopes, dists = vd.get_slopes_and_distances_in_pairs_of_large_regions(vor, adjacent_missed_regions)
    print("batch", index + 1, "of", size, "done")
    return slopes, dists

if __name__ == "__main__":
    plants, src_driver, src_crs, src_schema = util.open_shape_file(path)
    beds = dib.divide(plants)
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
    
    results = [p.apply_async(get_slopes, (batches[i], i, len(batches))) for i in range(len(batches))]
    
    slopes = []
    dists = []
    for res in results:
        if res.get():
            slopes += res.get()[0]
            dists += res.get()[1]
            print('mean slope:', np.nanmedian(slopes), 'cv score:', abs(np.nanstd(slopes))/np.nanmean(slopes), 'mean dist:', np.nanmedian(dists))