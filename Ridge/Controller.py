import METAA
import CANNY
import RIDGE
import RECCM
import gdal

l             = 0
base          = METAA.SelectBase("/")
files         = METAA.SelectFiles(base)
baseR         = METAA.SelectBase(base)
filesR        = METAA.SelectFiles(base)
psC,psF,w,md  = METAA.Initialize()

print("[BASE]")
gt0C,Img0C,ImgB0C,MaskB0C,fx0C,fy0C                          = METAA.OrthOpenin(psC,base)
EdgeMap0C,OriMap0C,MaskMap0C,GradPoints0C                    = CANNY.CannyPFree(psC,ImgB0C,MaskB0C)
EdgesA0C,EdgesB0C,EdgesC0C,ChainsA0C                         = CANNY.CannyLines(psC,EdgeMap0C,OriMap0C,MaskMap0C,GradPoints0C)
Ridges0,gt0F,fx0F,fy0F,Contour0F                             = RIDGE.DemOpening(psF,baseR)

for i in range(0,len(files)):
    print("[IMAGE "+str(i+1)+"/"+str(len(files))+"]")
    gt1C,Img1C,ImgB1C,MaskB1C,fx1C,fy1C                     = METAA.OrthOpenin(psC,base)
    EdgeMap1C,OriMap1C,MaskMap1C,GradPoints1C               = CANNY.CannyPFree(psC,ImgB1C,MaskB1C)
    EdgesA1C,EdgesB1C,EdgesC1C,ChainsA1C                    = CANNY.CannyLines(psC,EdgeMap1C,OriMap1C,MaskMap1C,GradPoints1C)
    x_offset,y_offset,X1C,Y1C,X0C,Y0C,CV1                   = RECCM.SinglMatch(psC,w,md,EdgesA1C,gt1C,fx1C,fy1C,EdgesA0C,gt0C,fx0C,fy0C,MaskB0C)
    Ridges1,gt1F,fx1F,fy1F,Contour1F                        = RIDGE.DemOpening(psF,filesR[i])
    gto1                                                    = gdal.Open(files[i]).GetGeoTransform()
    X1Oa,Y1Oa,Lon0Oa,Lat0Oa,X1Fa,Y1Fa,X0Fa,Y0Fa,CVa,dx,dy   = RECCM.PatchMatch(psF,w,md,Ridges1,gt1F,fx1F,fy1F,Ridges0,gt0F,fx0F,fy0F,Contour0F,x_offset,y_offset,CV1)
    X1Ob,Y1Ob,Lon0Ob,Lat0Ob,X1Fb,Y1Fb,X0Fb,Y0Fb,CVb,gcplist = RECCM.RemOutlier(X1Oa,Y1Oa,Lon0Oa,Lat0Oa,X1Fa,Y1Fa,X0Fa,Y0Fa,CVa,dx,dy,gt1F,gto1)
    RECCM.Georegistr(i,files,gcplist)