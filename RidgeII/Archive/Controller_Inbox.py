import METAA
import CANNY
import RECCM

if __name__ == '__main__':
    metapath,plist = METAA.ChrInbFiles(1)
    
    for path in metapath:
        plist,Img0C,ImgB0C,MaskB0C,gt0C           = METAA.OrtOpenDow(plist,path[0])
        plist,Edges0C                             = CANNY.CannyLines(plist,Img0C,ImgB0C,MaskB0C)
        plist,MaskB0F,gt0F,Edges0F                = METAA.DemOpenDow(plist,path[0],Img0C) 
        
        for i in range(1,len(path)):
            plist,Img1C,ImgB1C,MaskB1C,gt1C           = METAA.OrtOpenDow(plist,path[i])
            plist,Edges1C                             = CANNY.CannyLines(plist,Img1C,ImgB1C,MaskB1C)
            plist,MaskB1F,gt1F,Edges1F                = METAA.DemOpenDow(plist,path[i],Img1C)
            
            plist,x_off,y_off,CV1                                                               = RECCM.SinglMatch(plist,Edges1C,gt1C,Edges0C,gt0C,MaskB0C)
            plist,Edges1Fa,x_off,y_off,grid,md,c1,c2                                            = RECCM.InitiMatch(plist,Edges0F,Edges1F,MaskB0F,CV1,x_off,y_off)    
            plist,x0,y0,x1,y1,CVa,dx,dy                                                         = RECCM.MultiMatch(plist,Edges0F,Edges1F,Edges1Fa,CV1,x_off,y_off,grid,md,c1,c2,gt0F,gt1F)
            plist,x0,y0,x1,y1,CVa,dx,dy                                                         = RECCM.RemOutSlop(plist,x0,y0,x1,y1,CVa,dx,dy)
            plist                                                                               = METAA.CapFigures(plist,path,i)
            RECCM.GeoPointss(i,path,x0,y0,x1,y1,gt0F,gt1F)
