import METAA
import CANNY
import RECCM

if __name__ == '__main__':
    path,plist = METAA.SelectFiles()
    
    plist,Img0C,ImgB0C,MaskB0C,gt0C,fx0C,fy0C = METAA.OrtOpenDow(plist,path[0])
    plist,Edges0C                             = CANNY.CannyLines(plist,Img0C,ImgB0C,MaskB0C)
    plist,gt0F,fx0F,fy0F,MaskB0F,Edges0F      = METAA.DemOpening(plist,path[0],Img0C)
    
    for i in range(1,len(path)):
        plist,Img1C,ImgB1C,MaskB1C,gt1C,fx1C,fy1C = METAA.OrtOpenDow(plist,path[i])
        plist,Edges1C                             = CANNY.CannyLines(plist,Img1C,ImgB1C,MaskB1C)
        plist,gt1F,fx1F,fy1F,MaskB1F,Edges1F      = METAA.DemOpening(plist,path[i],Img1C)
    
        plist,x_off,y_off,x0,y0,xog,yog,x1,y1,CV1                                           = RECCM.SinglMatch(plist,Edges1C,gt1C,fx1C,fy1C,Edges0C,gt0C,fx0C,fy0C,MaskB0C)
        plist,pbar,inp,pool                                                                 = RECCM.OooneMatch(plist,Edges0F,Edges1F,MaskB0F,CV1,gt0F,gt1F,fx0F,fy0F,fx1F,fy1F,x_off,y_off)       
        plist,results                                                                       = RECCM.TwoooMatch(plist,pbar,inp,pool)
        plist,origin_x,origin_y,target_lon,target_lat,x0,y0,xog,yog,xof,yof,x1,y1,CVa,dx,dy = RECCM.ThreeMatch(plist,pbar,results,Edges0F,Edges1F)
        plist,origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,gcplist1,gcplist2     = RECCM.RemOutSlop(plist,origin_x,origin_y,target_lon,target_lat,x0,y0,x1,y1,CVa,dx,dy,gt1F,path,i)
        plist                                                                               = METAA.CapFigures(i,path,plist)
        RECCM.Georegistr(i,path,gcplist1,gcplist2)
        RECCM.GeoPointss(i,path,target_lon,target_lat,origin_x,origin_y,gt1F)