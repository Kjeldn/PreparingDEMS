import METAA
import CANNY
import RIDGE
import RECCM
import gdal

l             = 0
base          = METAA.SelectBase()
files         = METAA.SelectFiles(base)
baseR         = METAA.SelectBase()
filesR        = METAA.SelectFiles(baseR)
psC,psF,w,md  = METAA.Initialize()

print("[BASE]")
Ridges0,gt0,fx0,fy0,Contour0                                = RIDGE.DemOpening(psF,baseR)
 
for i in range(0,len(files)):
    print("[IMAGE "+str(i+1)+"/"+str(len(files))+"]")
    Ridges1,gt1,fx1,fy1,Contour1                            = RIDGE.DemOpening(psF,filesR[i])
    gto1                                                    = gdal.Open(files[i]).GetGeoTransform()
    X1Oa,Y1Oa,Lon0Oa,Lat0Oa,X1Fa,Y1Fa,X0Fa,Y0Fa,CVa,dx,dy   = RECCM.PatchMatch(psF,w,md,Ridges1,gt1,fx1,fy1,Ridges0,gt0,fx0,fy0,Contour0)
    X1Ob,Y1Ob,Lon0Ob,Lat0Ob,X1Fb,Y1Fb,X0Fb,Y0Fb,CVb,gcplist = RECCM.RemOutlier(X1Oa,Y1Oa,Lon0Oa,Lat0Oa,X1Fa,Y1Fa,X0Fa,Y0Fa,CVa,dx,dy,gt1,gto1)

    # Revert
    RECCM.Georegistr(i,files,gcplist)
    
    