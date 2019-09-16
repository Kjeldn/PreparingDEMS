import util
from pyqtree import Index
import numpy as np

def remove_outliers(plants, slope_field):
    plants = np.array(plants)
    spindex = Index(bbox=(np.amin(plants[:, 0]), np.amin(plants[:,1]), np.amax(plants[:,0]), np.amax(plants[:, 1])))
    for p in plants:
        spindex.insert(p, bbox=(p[0], p[1], p[0], p[1]))
        
    identified_plants = []
    non_identified_plants = []
    for i in range(len(plants)):
        p = plants[i]
        dy = 0.1 #5 meters
        dx = 0.1 #5 meters
        if any(util.get_slope_and_dist(p, q)[0] > util.ci_slopes(p, q, slope_field, 0.01)[0] 
            and util.get_slope_and_dist(p, q)[0] < util.ci_slopes(p, q, slope_field, 0.01)[1] for q in spindex.intersect(bbox=(p[0] - dx, p[1] - dy, p[0] + dx, p[1] + dy))):
            identified_plants.append(p)
        else:
            non_identified_plants.append(p)
    return identified_plants, non_identified_plants