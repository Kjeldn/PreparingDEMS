import numpy as np
from pyqtree import Index
import divide_into_beds as dib
import voronoi_diagram as vd
import util_voronoi as util
import remove_outliers as ro
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from collections import deque

path = r"Z:\800 Operational\c01_verdonk\Rijweg stalling 2\20190709\1156\Plant_count\bed3.gpkg"
clip_voronoi = True
slope_field = -0.744001707930606
batch_size = 2000
overlap = 1000
dist_between_two_crops = 0.101646283
plot_voronoi_diagram = False

delta1 = 0.02 #0.01 < delta1 < 0.05
delta2 = 0.03 #0.01 < delta1 < delta2 <0.06
n_batch = 1

def get_missing_points(plants, plot=False, first_it=True, mean_dist=None):
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
    missed_points = missed_points + vd.find_missed_points_in_regions(adjacent_missed_regions, vor, s, d, delta1 = delta1, delta2 = delta2)
    if plot:
        vd.plot_voronoi_diagram(vor, np.array(missed_points), missed_regions, small_regions)
    return missed_points, a, ci, adjacent_missed_regions, slopes, dists, vor, missed_regions

if __name__ == "__main__":
    plants, src_driver, src_crs, src_schema = util.open_shape_file(path)
    missed_points_coord = []
    slope_means = []
    dists_means = []
    
    beds = dib.divide(np.array(plants + missed_points_coord), heighest_in_group = False, b=1, c=8)
#    beds = [np.array(plants + missed_points_coord)]
   
    batches = []
    for j in range(len(beds)):
        bed = np.array(beds[j])
        if len(bed) > 500:
            for i in range(int(np.ceil(len(bed)/batch_size))):
                offset = batch_size if (i + 1) * batch_size < len(bed) else len(bed) - i * batch_size
                offset = offset + overlap if i * batch_size + offset + overlap < len(bed) else offset
                if offset > 500:
                    batches.append(bed[i * batch_size: i * batch_size + offset, :])
    
    #%%
    n_batch = 1
    
    missed_points_coord = []
    
    for i in range(1):
        batch = batches[i]
        plants_i, mean_x_coord, mean_y_coord = util.readable_values(batch)
    #    plants_i, r = ro.remove_outliers(plants_i, slope_field)    
        plants_i = np.array(plants_i)
        
        missed_points, a, ci, adjacent_missed_regions, slopes, dists, vor, missed_regions = get_missing_points(plants_i, plot=plot_voronoi_diagram)
        s = sorted(adjacent_missed_regions, key=len, reverse=True)
      
    #%%
    n_hole = 0
    import util_voronoi as util
    l = util.find_points_in_line(s[n_hole], slope_field,vor, delta1=delta1, delta2=delta2)
    u = unary_union([Polygon(v.vc) for v in list(s[n_hole])])
    ss = []
    for ll in l:
        for i in range(len(ll) - 1):
            ss.append(util.get_slope_and_dist(ll[i], ll[i + 1])[0])
    l = util.find_points_in_line(s[n_hole], np.median(ss),vor, delta1=delta1, delta2=delta2)
    ss = []
    for ll in l:
        for i in range(len(ll) - 1):
            ss.append(util.get_slope_and_dist(ll[i], ll[i + 1])[0])
    l = util.find_points_in_line(s[n_hole], np.median(ss),vor, delta1=delta1, delta2=delta2)
# =============================================================================
#     for i in range(len(l)):
#         ll = deque(l[i])
#         
#         a = LineString([(ll[-1][0], ll[-1][1]), (ll[-1][0] +  2* np.cos(np.median(ss)), ll[-1][1] + 2*np.sin(np.median(ss)))]).intersection(u.exterior)
#         if a.type == "Point":
#             ll.append(np.array(list(a.coords[0])))
#         elif a.type == "MultiPoint":
#             ll.append(np.array(list(sorted(a, key=lambda aa : aa.distance(Point((ll[-1][0], ll[-1][1]))), reverse=True)[0].coords[0])))
#             
#         b = LineString([(ll[0][0] -  2* np.cos(np.median(ss)), ll[0][1] - 2*np.sin(np.median(ss))), (ll[0][0], ll[0][1])]).intersection(u.exterior)
#         if b.type == "Point":
#             ll.appendleft(np.array(list(b.coords[0])))
#         elif b.type == "MultiPoint":
#             ll.appendleft(np.array(list(sorted(b, key=lambda aa : aa.distance(Point((ll[-1][0], ll[-1][1]))), reverse=True)[0].coords[0])))
#             
#         l[i] = list(ll)
# =============================================================================
    
    mp = vd.find_missed_points_in_regions([s[n_hole]], vor, slope_field, np.mean(dists), delta1=delta1, delta2=delta2)
    
    plt.figure()
    for v in list(s[n_hole]):
        plt.plot(*Polygon(v.vc).exterior.xy, 'k')
        plt.fill(*zip(*v.vc), color="r", alpha = 0.25)
    for ll in l:
        plt.plot(np.array(ll)[:,0], np.array(ll)[:,1], '-o', color='b')
    plt.plot(np.array(mp)[:,0], np.array(mp)[:,1], 'o', color='r')
    plt.show()
