import util
from pyqtree import Index
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from collections import OrderedDict

slope_field = -0.17404523952846995
path = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Joke Visser\20190603_final_plant_count.gpkg"
dst = r"C:\Users\wytze\OneDrive\Documents\vanBoven\Joke Visser\20190603_final_plant_count_filtered.gpkg"

plants, src_driver, src_crs, src_schema = util.open_shape_file(path)
plants = np.array(plants)
plants_r, mean_x_coord, mean_y_coord = util.readable_values(plants)
spindex = Index(bbox=(np.amin(plants_r[:, 0]), np.amin(plants_r[:,1]), np.amax(plants_r[:,0]), np.amax(plants_r[:, 1])))
for p in plants_r:
    spindex.insert(p, bbox=(p[0], p[1], p[0], p[1]))
    
identified_plants = []
non_identified_plants = []
for i in trange(len(plants_r), desc="finding outliers"):
    p = plants_r[i]
    dy = 0.5
    dx = 0.5
    if any(util.get_slope_and_dist(p, q)[0] > util.ci_slopes(p, q, slope_field, 0.01)[0] 
        and util.get_slope_and_dist(p, q)[0] < util.ci_slopes(p, q, slope_field, 0.01)[1] for q in spindex.intersect(bbox=(p[0] - dx, p[1] - dy, p[0] + dx, p[1] + dy))):
        identified_plants.append(p)
    else:
        non_identified_plants.append(p)
        
plt.scatter(np.array(identified_plants)[:,0], np.array(identified_plants)[:,1], color='b')
plt.scatter(np.array(non_identified_plants)[:,0], np.array(non_identified_plants)[:,1], color='r')
plt.show()

util.write_shape_file(dst, identified_plants, src_crs, src_driver, {'geometry': 'Point', 'properties': OrderedDict([('name', 'str')])}, 'filtered point')