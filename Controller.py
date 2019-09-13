import METAA
import CANNY
import RECCM

l             = 0
base          = METAA.SelectBase()
files         = METAA.SelectFiles(base)
psC,psF,w,md  = METAA.Initialize()

print("[BASE]")
gt0,Img0C,ImgB0C,MaskB0C,fx0C,fy0C,Img0F,fx0F,fy0F          = METAA.OrthOpenin(psC,psF,base)
EdgeMap0C,OriMap0C,MaskMap0C,GradPoints0C                   = CANNY.CannyPFree(psC,ImgB0C,MaskB0C)
EdgesA0C,EdgesB0C,EdgesC0C,ChainsA0C                        = CANNY.CannyLines(psC,EdgeMap0C,OriMap0C,MaskMap0C,GradPoints0C)

ImgB0F,MaskB0F,Contour0F                                    = METAA.OrthSwitch(psF,psC,Img0F,ChainsA0C)
EdgeMap0F,OriMap0F,MaskMap0F,GradPoints0F                   = CANNY.CannyPFree(psF,ImgB0F,MaskB0F)
EdgesA0F,EdgesB0F,EdgesC0F,ChainsA0F                        = CANNY.CannyLines(psF,EdgeMap0F,OriMap0F,MaskMap0F,GradPoints0F)
 
for i in range(0,len(files)):
    print("[IMAGE "+str(i+1)+"/"+str(len(files))+"]")
    o                                                       = METAA.LogStarter(l,i,files)
    gt1,Img1C,ImgB1C,MaskB1C,fx1C,fy1C,Img1F,fx1F,fy1F      = METAA.OrthOpenin(psC,psF,files[i])
    EdgeMap1C,OriMap1C,MaskMap1C,GradPoints1C               = CANNY.CannyPFree(psC,ImgB1C,MaskB1C)
    EdgesA1C,EdgesB1C,EdgesC1C,ChainsA1C                    = CANNY.CannyLines(psC,EdgeMap1C,OriMap1C,MaskMap1C,GradPoints1C)
    x_offset,y_offset,X1C,Y1C,X0C,Y0C,CV1                   = RECCM.SinglMatch(psC,w,md,EdgesA1C,gt1,fx1C,fy1C,EdgesA0C,gt0,fx0C,fy0C,MaskB0C)

    ImgB1F,MaskB1F,Contour1F                                = METAA.OrthSwitch(psF,psC,Img1F,ChainsA1C)
    EdgeMap1F,OriMap1F,MaskMap1F,GradPoints1F               = CANNY.CannyPFree(psF,ImgB1F,MaskB1F)
    EdgesA1F,EdgesB1F,EdgesC1F,ChainsA1F                    = CANNY.CannyLines(psF,EdgeMap1F,OriMap1F,MaskMap1F,GradPoints1F)
    X1Oa,Y1Oa,Lon0Oa,Lat0Oa,X1Fa,Y1Fa,X0Fa,Y0Fa,CVa,dx,dy   = RECCM.PatchMatch(psF,w,md,EdgesA1F,gt1,fx1F,fy1F,EdgesA0F,gt0,fx0F,fy0F,Contour0F,x_offset,y_offset,CV1)
    X1Ob,Y1Ob,Lon0Ob,Lat0Ob,X1Fb,Y1Fb,X0Fb,Y0Fb,CVb,gcplist = RECCM.RemOutlier(X1Oa,Y1Oa,Lon0Oa,Lat0Oa,X1Fa,Y1Fa,X0Fa,Y0Fa,CVa,dx,dy)
   
    METAA.CreateFigs(l,i,files,psC,psF,w,Img0C,Img1C,ImgB0C,ImgB1C,Img0F,Img1F,ImgB0F,ImgB1F,EdgesA0C,EdgesB0C,EdgesC0C,EdgesA1C,EdgesB1C,EdgesC1C,EdgesA0F,EdgesB0F,EdgesC0F,EdgesA1F,EdgesB1F,EdgesC1F,X0C,Y0C,X1C,Y1C,X0Fa,Y0Fa,X1Fa,Y1Fa,X0Fb,Y0Fb,X1Fb,Y1Fb,CVb)
    RECCM.Georegistr(i,files,gcplist)
    METAA.LogStopper(o)