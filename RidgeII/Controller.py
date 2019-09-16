import METAA
import CANNY
import RECCM

l    = 0
path = METAA.SelectFiles() 

Img0C,ImgB0C,MaskB0C,gt0C,fx0C,fy0C = METAA.OrtOpening(path[0])
Edges0C                             = CANNY.CannyLines(ImgB0C,MaskB0C)
ps0F,gt0F,fx0F,fy0F,MaskB0F,Edges0F = METAA.DemOpening(path[0],0)

for i in range(1,len(path)):
    Img1C,ImgB1C,MaskB1C,gt1C,fx1C,fy1C = METAA.OrtOpening(path[i])
    Edges1C                             = CANNY.CannyLines(ImgB1C,MaskB1C)
    ps1F,gt1F,fx1F,fy1F,MaskB1F,Edges1F = METAA.DemOpening(path[i],ps0F)

    x_off,y_off,x0,y0,xog,yog,x1,y1,CV1                                           = RECCM.SinglMatch(Edges1C,gt1C,fx1C,fy1C,Edges0C,gt0C,fx0C,fy0C,MaskB0C)
    origin_x,origin_y,target_lon,target_lat,x0,y0,xog,yog,xof,yof,x1,y1,CVa,dx,dy = RECCM.PatchMatch(ps0F,Edges1F,gt1F,fx1F,fy1F,Edges0F,gt0F,fx0F,fy0F,MaskB0F,x_off,y_off,CV1)
    origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,gcplist               = RECCM.RemOutlier(origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,dx,dy,gt1F,path,i)
    RECCM.Georegistr(i,path,gcplist)
