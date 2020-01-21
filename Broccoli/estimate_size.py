from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from shapely.geometry import Polygon, Point, LineString
import pandas as pd
import PIL.ExifTags
import scipy.stats
from shapely.strtree import STRtree as rt

img = Image.open(r"Z:\100 Testing\190716 Iron Man AZ74 10m difuse light\Haalbaarheidstest\detected_broccolis.jpg")
bands = img.getbands()

def calculate_distance_from_pixels(width: bool, n_pixels: int, flown_height, picsize_x, picsize_y):
    return ((flown_height * np.tan(np.arcsin(14/35)) * 2) / picsize_x if width else picsize_y) * n_pixels;

pix = np.array(img)
red = pix[:,:,0]
red[red < 250] = 0
red[red >= 250] = 1
#plt.imshow(red)

red = np.array(red)

_,contours, hier = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

polys = []

for c in contours:
    if len(c) > 3 :
        poly = Polygon([c[i][0] for i in range(len(c))])
        if poly.area > 10 and max([poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]]) > 20:
            polys.append(poly)


#plt.hist([p.area for p in polys], bins=400)


points = []
sizes = []
for p in polys:
    plt.plot(*p.exterior.xy)
    points.append(p.centroid)
    sizes.append(max([p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]]))
    
plt.scatter([p.x for p in points], [p.y for p in points])


sizes_in_m = []
for s in sizes:
    sizes_in_m.append(calculate_distance_from_pixels(True, s, 10, img.size[1], img.size[0]))
    
    
def get_slope(p, q):
    [p, q] = sorted([p, q], key=lambda v : v[0])
    return np.arctan((p[1] - q[1])/ (p[0] - q[0]))

def get_dist(p, q):
    return np.sqrt((p[1] - q[1])**2 + (p[0] - q[0])**2)

tree = rt(points)
slopes = []
for p in points:
    for q in tree.query(p.buffer(500)):
        if p.x != q.x and p.y != q.y:
            slopes.append(get_slope([p.x, p.y],[q.x, q.y]))
            
a,b,_ = plt.hist(slopes, bins=200)
slope = b[np.argmax(a)]

def ci_slopes(p, q, slope_field, delta):
    dist = get_dist(p, q)
    if dist != 0:
        return (slope_field - np.arctan(delta/dist), slope_field + np.arctan(delta/dist))
    else:
        return (0,0)


def get_if_in_range(p, q, slope_field, delta):
    slope = get_slope(p,q)
    ci = ci_slopes(p,q, slope_field, delta)
    if (slope < ci[1] and slope > ci[0]):
        return True
    return False

#%%
def find_points_in_line(points, slope_field, delta1 = 100, delta2 = 300):
    def get_slope_float(p, q):
        [p, q] = sorted([p, q], key=lambda v : v[0])
        return np.arctan((p[1] - q[1])/ (p[0] - q[0]))
    v = np.vectorize(get_slope_float, otypes=[np.float], signature='(2),(2)->()')
    indices = [i for i in range(len(points))]
    m = np.array([indices,]*len(indices)).flatten()
    n = np.array([indices,]*len(indices)).T.flatten()
    p = [points[i] for i in m]
    q = [points[i] for i in n]
    slopes = v(p,q).reshape((len(points), len(points)))
    lines = []
    for i, p in enumerate(points):
        closest_slope_p = None
        closest_slope = 0
        for j, q in enumerate(points):
            if p != q:
                if not closest_slope_p:
                    closest_slope_p = q
                    closest_slope = slopes[i,j]
                else:
                    slope = slopes[i,j]
                    if abs(slope_field - slope) < abs(slope_field - closest_slope):
                        closest_slope_p = q
                        closest_slope = slope
                        
        ci_s = ci_slopes(p, q, slope_field, delta1)
        
        if closest_slope > ci_s[0] and closest_slope < ci_s[1]:
            lines.append([p, closest_slope_p])
        
    for i in range(len(lines)):
        found = False
        for j in range(len(lines)):
            for k in range(len(lines)):
                if k != j and any(p in lines[j] for p in lines[k]):
                    newline = list(set(lines[j] + lines[k]))
                    if j < k:
                        del lines[k]
                        del lines[j]
                    else:
                        del lines[j]
                        del lines[k]
                    lines.append(newline)
                    found = True
                    break
            if found:
                break
            
        
    for i in range(len(lines)):
        found = False
        for j in range(len(lines)):
            for k in range(len(lines)):
                if j !=k:
                    n = 0
                    t = 0
                    for p in lines[k]:
                        for q in lines[j]:
                            t+= 1
                            if get_if_in_range(p,q,slope_field, delta2):
                                n += 1
                    if n/t > 0.8:
                        newline = list(set(lines[j] + lines[k]))
                        if j < k:
                            del lines[k]
                            del lines[j]
                        else:
                            del lines[j]
                            del lines[k]
                        lines.append(newline)
                        found = True
                        break
            if found:
                break
     
    return lines

lines = find_points_in_line([(points[i].x, points[i].y) for i in range(len(points))], -1.4904805512171104)

m = []
c = []

for line in lines:
    line = sorted(line, key=lambda p:p[1])
    plt.plot(np.array(line)[:,0], np.array(line)[:,1])
    x = np.array(line)[:,0]
    y = np.array(line)[:,1]
    A = np.vstack([x, np.ones(len(x))]).T
    mm, cc = np.linalg.lstsq(A, y, rcond=None)[0]
    m.append(mm)
    c.append(cc)
    
    
xx0 = []
xxn = []
for i in range(len(m)):
    x0 = (img.size[1] - c[i])/m[i]
    xn = (-c[i])/m[i]
    xx0.append(x0)
    xxn.append(xn)
    plt.plot([x0, xn],[img.size[1], 0])
    
xx0 =sorted(xx0)
xxn = sorted(xxn)

plt.plot([LineString([(xx0[i], img.size[1]), (xxn[i], 0)]).distance(LineString([(xx0[i + 1], img.size[1]), (xxn[i + 1], 0)])) for i in range(len(xx0) - 1)])

xx05 = 0.5*(xx0[0] + xxn[0])
xxn5 = 0.5*(xx0[-1] + xxn[-1])
pix_per_row = np.sqrt(xx05**2 + xxn5**2) / (len(xx0) -1)
cnt_per_row = 2038 / 27
cnt_per_pix = cnt_per_row/pix_per_row

cnt_per_pix2 = calculate_distance_from_pixels(True, 1, 10, img.size[1], img.size[0])*100

for i,p in enumerate(points):
    plt.text(p.x/img.size[0], 1-p.y/img.size[1], np.around(cnt_per_pix * sizes[i], decimals=2))