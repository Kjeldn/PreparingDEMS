import METAA
import CANNY
import RECCM
import GEORE

"""
Finds orthomosaics in the inbox that have a corresponding DEM in the inbox as well, then finds a suitable 
orthomosaic plus DEM for georegistration, which is refered to as the base. Both sets are then processed 
to binary maps using METAA and CANNY functions. RECCM functions take the binary maps and find corresponding 
locations, which are used as GCPs to georegister the set found in the inbox.
---
METAA: Contains functions that navigate directories, open or move files, and meta-functions for all scripts.
CANNY: Contains a function that combines the CannyPF and CannyLines procedure for an input image.
RECCM: Contains functions that handle anything associated with RECC matching, and outlier removal.
GEORE: Contains a function that carries out the actual georegistration, using a .points to create .vrt's.
---
inbox    | str    | Path to inbox  
archive  | str    | Path to archive
rtu      | str    | Path to ready to upload 
nrtu     | str    | Path to not ready to upload
dstr     | str    | Path to rectified dems and points
rec      | str    | Path to receipt
plist    | list   | List for plots
pathlist | list   | List with paths to suitable files in the inbox
path     | str    | Path to the orthomosaic up for georegistration
base     | str    | Path to base orthomosaic for current inbox file
Img0C    | 3D arr | Rescaled and equalised orthomosaic
ImgB0C   | 2D arr | Rescaled, equalised, grayscaled and blurred orthomosaic
MaskB0C  | 2D arr | Binary map defining edges of orthomosaic
gt0C     | tuple  | Geotransform corresponding to Img0C, ImgB0C, MaskB0C and Edges0C
Edges0C  | 2D arr | Binary map with edges found by CannyPF and CannyLines
MaskB0F  | 2D arr | Binary map defining edges of orthomosaic
gt0F     | tuple  | Geotransform corresponding to MaskB0F and Edges0F
Edges0F  | 2D arr | Binary map with edges or ridges found in DEM data
x_off    | int    | X-offset found by OneMatch function using Edges0C and Edges1C
y_off    | int    | Y-offset found by OneMatch function using Edges0C and Edges1C
CV1      | float  | Concentration value for the match produced by OneMatch
Edges1Fa | 2D arr | Buffered version of Edges1F to prevent out of bounds
grid     | list   | List of 200 or 300 (x,y) tuples that form a grid in Edges0F
md       | int    | Maximum distance, the maximum feasable error in pixels
c1       | 2D arr | Binary circle for clipping a patch
c2       | 2D arr | Binary circle for clipping of a search map
x0       | 1D arr | Array containing x-pixel in Edges0F
y0       | 1D arr | Array containing y-pixel in Edges0F
x1       | 1D arr | Array containing x-pixel in Edges1F corresponding to x0
y1       | 1D arr | Array containing y-pixel in Edges1F corresponding to y0
CV       | 1D arr | Array with concentration values corresponding to each (x0,y0) - (x1,y1) match
dx       | 1D arr | Array containing x-offset in meters for each match
dy       | 1D arr | Array containing y-offset in meters for each match
GCPstat  | tuple  | Contains the status of matches or GCP's after outlier removal
f1       | int    | Flag for creation of .points file
f2       | int    | Flag for creation of .vrt files
f3       | int    | Flag for moving files to corresponding folders
""" 

inbox,archive,rtu,nrtu,dstr,rec = METAA.GetDirs("STAMPERTJE")
if __name__ == '__main__':
    plist,pathlist = METAA.FindFile(inbox)
    for path in pathlist:
        plist,base                               = METAA.FindBase(plist,pathlist,archive,rtu,path)
        plist,Img0C,ImgB0C,MaskB0C,gt0C          = METAA.OpenOrth(plist,base)
        plist,Edges0C                            = CANNY.CannyLin(plist,Img0C,ImgB0C,MaskB0C)
        plist,MaskB0F,gt0F,Edges0F               = METAA.OpenDEMs(plist,base) 
        plist,Img1C,ImgB1C,MaskB1C,gt1C          = METAA.OpenOrth(plist,path)
        plist,Edges1C                            = CANNY.CannyLin(plist,Img1C,ImgB1C,MaskB1C)
        plist,MaskB1F,gt1F,Edges1F               = METAA.OpenDEMs(plist,path)
        plist,x_off,y_off,CV1                    = RECCM.OneMatch(plist,Edges1C,gt1C,Edges0C,gt0C,MaskB0C)
        plist,Edges1Fa,x_off,y_off,grid,md,c1,c2 = RECCM.IniMatch(plist,Edges0F,Edges1F,MaskB0F,x_off,y_off,CV1)    
        plist,x0,y0,x1,y1,CV,dx,dy               = RECCM.MulMatch(plist,Edges0F,Edges1F,Edges1Fa,x_off,y_off,CV1,grid,md,c1,c2,gt0F,gt1F)
        plist,x0,y0,x1,y1,CV,dx,dy,GCPstat       = RECCM.RemovOut(plist,Edges0F,Edges1F,x0,y0,x1,y1,CV,dx,dy)
        plist                                    = METAA.SaveFigs(plist,path,base,rec,GCPstat)
        f1                                       = RECCM.MakePnts(path,x0,y0,x1,y1,gt0F,gt1F)
        f2                                       = GEORE.MakeVRTs(path)
        f3                                       = METAA.MoveFile(path,rtu,nrtu,dstr,rec,GCPstat)    