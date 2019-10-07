import METAA
import CANNY
import RECCM
import GEORE

# Martijn:
# inbox = r"\\STAMPERTJE\100 Ortho Inbox"
# archive = r"\\STAMPERTJE\Data\VanBovenDrive\VanBoven MT\Archive"

# STAMPERTJE:
inbox = r"C:\Users\VanBoven\Documents\100 Ortho Inbox"
archive = r"D:\VanBovenDrive\VanBoven MT\Archive"

# Manual METAA.SelectFile()
plist = []
path = [r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190513\1240\Orthomosaic\c07_hollandbean-Aart Maris-201905131240.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190522\0956\Orthomosaic\c07_hollandbean-Aart Maris-201905220956.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190527\1152\Orthomosaic\c07_hollandbean-Aart Maris-201905271152.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190603\0946\Orthomosaic\c07_hollandbean-Aart Maris-201906030946.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190617\1333\Orthomosaic\c07_hollandbean-Aart Maris-201906171333.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190624\0942\Orthomosaic\c07_hollandbean-Aart Maris-201906240942.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190703\0945\Orthomosaic\c07_hollandbean-Aart Maris-201907030945.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190724\1028\Orthomosaic\c07_hollandbean-Aart Maris-201907241028.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190806\1005\Orthomosaic\c07_hollandbean-Aart Maris-201908061005.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190823\1048\Orthomosaic\c07_hollandbean-Aart Maris-201908231048.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190829\1123\Orthomosaic\c07_hollandbean-Aart Maris-201908291123.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190906\0917\Orthomosaic\c07_hollandbean-Aart Maris-201909060917.tif"]

path = [#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190617\1419\Orthomosaic\c07_hollandbean-Hein de Schutter-201906171419-GR.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190624\1432\Orthomosaic\c07_hollandbean-Hein de Schutter-201906241432-GR.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190703\1301\Orthomosaic\c07_hollandbean-Hein de Schutter-201907031301-GR.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190708\1101\Orthomosaic\c07_hollandbean-Hein de Schutter-201907081101-GR.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190806\1119\Orthomosaic\c07_hollandbean-Hein de Schutter-201908061119-GR.tif",
        r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190823\1146\Orthomosaic\c07_hollandbean-Hein de Schutter-201908231146.tif"]

if __name__ == '__main__':
    #plist,path = METAA.SelectFile(inbox)
    
    for file in path:
        plist,base                               = METAA.FirsttBase(plist,archive,file)
        plist,Img0C,ImgB0C,MaskB0C,gt0C          = METAA.OrtOpenDow(plist,base)
        plist,Edges0C                            = CANNY.CannyLines(plist,Img0C,ImgB0C,MaskB0C)
        plist,MaskB0F,gt0F,Edges0F               = METAA.DemOpenDow(plist,base,Img0C) 
        plist,Img1C,ImgB1C,MaskB1C,gt1C          = METAA.OrtOpenDow(plist,file)
        plist,Edges1C                            = CANNY.CannyLines(plist,Img1C,ImgB1C,MaskB1C)
        plist,MaskB1F,gt1F,Edges1F               = METAA.DemOpenDow(plist,file,Img1C)
        plist,x_off,y_off,CV1                    = RECCM.SinglMatch(plist,Edges1C,gt1C,Edges0C,gt0C,MaskB0C)
        plist,Edges1Fa,x_off,y_off,grid,md,c1,c2 = RECCM.InitiMatch(plist,Edges0F,Edges1F,MaskB0F,CV1,x_off,y_off)    
        plist,x0,y0,x1,y1,CVa,dx,dy              = RECCM.MultiMatch(plist,Edges0F,Edges1F,Edges1Fa,CV1,x_off,y_off,grid,md,c1,c2,gt0F,gt1F)
        plist,x0,y0,x1,y1,CVa,dx,dy              = RECCM.RemOutSlop(plist,Edges0F,Edges1F,x0,y0,x1,y1,CVa,dx,dy)
        plist                                    = METAA.CapFigures(plist,file)
        f1                                       = RECCM.MakePoints(file,x0,y0,x1,y1,gt0F,gt1F)
        f2                                       = GEORE.Georegistr(file)