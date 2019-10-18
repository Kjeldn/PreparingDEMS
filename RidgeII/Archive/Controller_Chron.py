import METAA
import CANNY
import RECCM
import GEORE

inbox,archive,rtu,nrtu,dstr,rec = METAA.GetDirs("STAMPERTJE")

if __name__ == '__main__':
    plist,pathlist = METAA.FindFile(inbox)
    
    for path in pathlist:
        plist,base                               = METAA.FindBase(plist,pathlist,archive,rtu,path)
        plist,Img0C,ImgB0C,MaskB0C,gt0C          = METAA.OpenOrth(plist,base)
        plist,Edges0C                            = CANNY.CannyLin(plist,Img0C,ImgB0C,MaskB0C)
        plist,MaskB0F,gt0F,Edges0F               = METAA.OpenDEMs(plist,base,Img0C) 
        plist,Img1C,ImgB1C,MaskB1C,gt1C          = METAA.OpenOrth(plist,path)
        plist,Edges1C                            = CANNY.CannyLin(plist,Img1C,ImgB1C,MaskB1C)
        plist,MaskB1F,gt1F,Edges1F               = METAA.OpenDEMs(plist,path,Img1C)
        plist,x_off,y_off,CV1                    = RECCM.OneMatch(plist,Edges1C,gt1C,Edges0C,gt0C,MaskB0C)
        plist,Edges1Fa,x_off,y_off,grid,md,c1,c2 = RECCM.IniMatch(plist,Edges0F,Edges1F,MaskB0F,CV1,x_off,y_off)    
        plist,x0,y0,x1,y1,CVa,dx,dy              = RECCM.MulMatch(plist,Edges0F,Edges1F,Edges1Fa,CV1,x_off,y_off,grid,md,c1,c2,gt0F,gt1F)
        plist,x0,y0,x1,y1,CVa,dx,dy,GCPstat      = RECCM.RemovOut(plist,Edges0F,Edges1F,x0,y0,x1,y1,CVa,dx,dy)
        plist                                    = METAA.SaveFigs(plist,path,base,rec,GCPstat)
        f1                                       = RECCM.MakePnts(path,x0,y0,x1,y1,gt0F,gt1F)
        f2                                       = GEORE.MakeVRTs(path)
        f3                                       = METAA.MoveFile(path,rtu,nrtu,dstr,rec,grid,f2)
        
# 