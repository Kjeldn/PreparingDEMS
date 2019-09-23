import METAA
import CANNY
import RECCM

path,plist = METAA.SelectFiles()

plist,Img0C,ImgB0C,MaskB0C,gt0C,fx0C,fy0C = METAA.OrtOpening(plist,path[0])
plist,Edges0C                             = CANNY.CannyLines(plist,Img0C,ImgB0C,MaskB0C)
plist,gt0F,fx0F,fy0F,MaskB0F,Edges0F      = METAA.DemOpening(plist,path[0],Img0C)

for i in range(1,len(path)):
    plist,Img1C,ImgB1C,MaskB1C,gt1C,fx1C,fy1C = METAA.OrtOpening(plist,path[i])
    plist,Edges1C                             = CANNY.CannyLines(plist,Img1C,ImgB1C,MaskB1C)
    plist,gt1F,fx1F,fy1F,MaskB1F,Edges1F      = METAA.DemOpening(plist,path[i],Img1C)

    plist,x_off,y_off,x0,y0,xog,yog,x1,y1,CV1                                           = RECCM.SinglMatch(plist,Edges1C,gt1C,fx1C,fy1C,Edges0C,gt0C,fx0C,fy0C,MaskB0C)
    plist,origin_x,origin_y,target_lon,target_lat,x0,y0,xog,yog,xof,yof,x1,y1,CVa,dx,dy = RECCM.PatchMatch(plist,Edges1F,gt1F,fx1F,fy1F,Edges0F,gt0F,fx0F,fy0F,MaskB0F,x_off,y_off,CV1)
    plist,origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,gcplist               = RECCM.RemOutlier(plist,origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,dx,dy,gt1F,path,i)
    plist                                                                               = METAA.CapFigures(i,path,plist)
    RECCM.Georegistr(i,path,gcplist)
    
    
#path = [r"E:\LOCALORTHO\T2.tif",r"E:\LOCALORTHO\T0.tif"]
