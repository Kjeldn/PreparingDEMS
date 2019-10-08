import METAA
import CANNY
import RECCM
import GEORE

# STAMPERTJE:
inbox   = r"C:\Users\VanBoven\Documents\100 Ortho Inbox"
archive = r"D:\VanBovenDrive\VanBoven MT\Archive"
rtu     = r"C:\Users\VanBoven\Documents\100 Ortho Inbox\2_ready_to_upload"
nrtu    = r"C:\Users\VanBoven\Documents\100 Ortho Inbox\4_not_ready_to_upload"
dstr    = r"C:\Users\VanBoven\Documents\100 Ortho Inbox\ ... "

if __name__ == '__main__':
    plist,pathlist = METAA.SelectFile(inbox)
    
    for path in pathlist:
        plist,base                               = METAA.FirsttBase(plist,archive,rtu,path)
        plist,Img0C,ImgB0C,MaskB0C,gt0C          = METAA.OrtOpenDow(plist,base)
        plist,Edges0C                            = CANNY.CannyLines(plist,Img0C,ImgB0C,MaskB0C)
        plist,MaskB0F,gt0F,Edges0F               = METAA.DemOpenDow(plist,base,Img0C) 
        plist,Img1C,ImgB1C,MaskB1C,gt1C          = METAA.OrtOpenDow(plist,path)
        plist,Edges1C                            = CANNY.CannyLines(plist,Img1C,ImgB1C,MaskB1C)
        plist,MaskB1F,gt1F,Edges1F               = METAA.DemOpenDow(plist,path,Img1C)
        plist,x_off,y_off,CV1                    = RECCM.SinglMatch(plist,Edges1C,gt1C,Edges0C,gt0C,MaskB0C)
        plist,Edges1Fa,x_off,y_off,grid,md,c1,c2 = RECCM.InitiMatch(plist,Edges0F,Edges1F,MaskB0F,CV1,x_off,y_off)    
        plist,x0,y0,x1,y1,CVa,dx,dy              = RECCM.MultiMatch(plist,Edges0F,Edges1F,Edges1Fa,CV1,x_off,y_off,grid,md,c1,c2,gt0F,gt1F)
        plist,x0,y0,x1,y1,CVa,dx,dy              = RECCM.RemOutSlop(plist,Edges0F,Edges1F,x0,y0,x1,y1,CVa,dx,dy)
        plist                                    = METAA.CapFigures(plist,path)
        f1                                       = RECCM.MakePoints(path,x0,y0,x1,y1,gt0F,gt1F)
        f2                                       = GEORE.Georegistr(path)
        f3                                       = METAA.TrafficPol(path,rtu,nrtu,dstr,grid,x0)