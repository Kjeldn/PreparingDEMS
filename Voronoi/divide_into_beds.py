from shapely.geometry import Polygon, Point, LineString
from shapely.geometry.polygon import LinearRing
import numpy as np
from pyqtree import Index
import matplotlib.pyplot as plt
from tqdm import tqdm
import util

def fill_points_in_line(p, q, n):
    ret = []
    for i in range(1, n + 1):
        point_to_add = [(i*p[0] + (n + 1 -i) * q[0])*(1/(n+1)), (i*p[1] + (n + 1 -i) * q[1])*(1/(n+1))]
        ret.append(point_to_add)
    return ret

"""
Divide list of points into beds. This is done with the following steps:
    1. 5000 equidistant points on the exterior of the convex hull of plants are made. For each point the distance
        to the closest point in plants is calculated.
    2. Points on the exterior of the convex hull with large distance to plants are grouped together if in the same neighbourhood.
    3. The point with the largest distance to plants in each group is calculated
    4. For each pair of points found in step 3 the line between them is evaluated. If points on this line have large distance to plants
        the likeliness of it being a line on a rijpad is large
    5. For each line with likeliness > 0.95 the points are split

Works better for wider rijpads.

Parameters
-----------
plants : list or numpy array of coordinates
    The plants that have to be divided into beds
plot : Boolean
    If True it plots the divided beds, the splitting lines and the points on the convex hull with large distance to plants
orentation : String
    If The orientation of the crop field is more North-South than East-West set this to North-South
    
Returns
-----------
beds : list of numpy arrays
    plants divided in beds
"""
def divide(plants, plot=False, orientation='East-West'):
    if orientation == 'North-South':
        plants = np.array(sorted(sorted(plants, key=lambda a : a[1]), key = lambda a: a[0]))
    else:    
        plants = np.array(sorted(sorted(plants, key=lambda a : a[0]), key = lambda a: a[1]))
    
    spindex = Index(bbox=(np.amin(plants[:,0]), np.amin(plants[:,1]), np.amax(plants[:,0]), np.amax(plants[:,1])))
    for plant in plants:
        spindex.insert(plant, bbox=(plant[0], plant[1], plant[0], plant[1]))

    convex_hulls = []
    
    n = 5000
    overlap = 500
    for i in range(int(np.ceil(len(plants)/n))):
        offset = n if (i + 1) * n < len(plants) else len(plants) - i * n
        offset = offset + overlap if i * n + offset + overlap < len(plants) else offset
        convex_hulls.append(util.get_convex_hull(plants[i * n: i * n + offset, :]))
        
    convex_hull = convex_hulls[0]
    for i in range(1, len(convex_hulls)):
        convex_hull = convex_hull.union(convex_hulls[i])
    
    ds = convex_hull.length / 5000
    dy = (np.max(plants[:,1]) - np.min(plants[:,1]))/2000
    dx = (np.max(plants[:,0]) - np.min(plants[:,0]))/2000
    
    x,y = convex_hull.exterior.xy
    
    dist = []
    for i in range(len(x) - 1):
        line = LineString([(x[i], y[i]), (x[i + 1], y[i + 1])])
        n = int(line.length / ds + 0.5) - 1
        points = fill_points_in_line(line.coords[0], line.coords[1], n) + [[x[i], y[i]]]
        for p in points:
            j = 1
            while j < 200 and not spindex.intersect((p[0] - j*dx, p[1] - j*dy, p[0] + j*dx, p[1] + j*dy)):
                j += 1
            dist.append({'coord': p, 'dist': j})
            
    lone_points = []
    dists = []
    indices = []
    for i in range(len(dist)):
        if dist[i]['dist'] > 4:
            if plot:
                plt.plot(dist[i]['coord'][0], dist[i]['coord'][1], '*')
                plt.text(dist[i]['coord'][0], dist[i]['coord'][1], dist[i]['dist'])
            lone_points.append(dist[i]['coord'])
            indices.append(i)
        
    for i in range(len(lone_points) - 1):
        dists.append(LineString([(lone_points[i][0], lone_points[i][1]), (lone_points[i + 1][0], lone_points[i + 1][1])]).length)
    
    groups = []
    groups.append([{'coord': lone_points[0], 'index': indices[0]}])
    for i in range(1, len(lone_points)):
        if indices[i] - indices[i - 1] < 4:
            for j in range(len(groups)):
                if {'coord': lone_points[i - 1], 'index': indices[i - 1]} in groups[j]:
                    groups[j].append({'coord': lone_points[i], 'index': indices[i]})
        else:
            groups.append([{'coord': lone_points[i], 'index': indices[i]}])
            
            
    for i in range(len(groups)-1, -1, -1):
        if len(groups[i]) < 4:
            del groups[i]
    
    max_points = []
    for i in range(len(groups)):
        max_index = groups[i][0]['index']
        max_dist = dist[max_index]['dist']
        for j in range(1, len(groups[i])):
            if dist[groups[i][j]['index']]['dist'] > max_dist:
                max_index = groups[i][j]['index']
                max_dist = dist[max_index]['dist']
                
        max_points.append(dist[max_index]['coord'])
    
    likeliness = []
    for i in range(len(max_points)):
        for j in range(i, len(max_points)):
            p = max_points[i]
            q = max_points[j]
            if p != q:
                l = LineString([(p[0], p[1]), (q[0], q[1])])
                if convex_hull.exterior.distance(Point((p[0] + q[0])/2, (p[1] + q[1])/2)) > ds:
                    points = fill_points_in_line(l.coords[0], l.coords[1], int(l.length / ds + 0.5) - 1)
                    n = 0
                    for point in points:
                        if spindex.intersect((point[0] - 3*dx, point[1] - 3*dy, point[0] + 3*dx, point[1] + 3*dy)):
                            n += 1
                    likeliness.append({'line': (i, j), 'likeliness': 1 - n/len(points)})
                else:
                    likeliness.append({'line': (i, j), 'likeliness': 0 })
    
    splitting_lines = []
    for l in likeliness:
        if l['likeliness'] > 0.95:
            splitting_lines.append([max_points[l['line'][0]], max_points[l['line'][1]]])
            if plot:
                plt.plot([max_points[l['line'][0]][0], max_points[l['line'][1]][0]], [max_points[l['line'][0]][1], max_points[l['line'][1]][1]])
    
    if plot:
        for p in max_points:
            plt.plot(p[0], p[1], 'o')
        plt.plot(x,y)
        plt.show()
    
    beds = [plants]
    
    pbar = tqdm(desc="dividing into beds", total = len(splitting_lines))
    for l in splitting_lines:
        newbeds = []
        for bed in beds:
            p1 = []
            p2 = []
            for plant in bed:
                if LinearRing([(l[0][0], l[0][1]), (l[1][0], l[1][1]), (plant[0], plant[1])]).is_ccw:
                    p1.append(plant)
                else:
                    p2.append(plant)
            if p1:
                newbeds.append(p1)
            if p2:
                newbeds.append(p2)
        beds = newbeds
        pbar.update(1)
    pbar.close()
    
    if plot:    
        for bed in beds:
            plt.scatter(np.array(bed)[:,0], np.array(bed)[:,1])
        plt.show()
    
    return beds
