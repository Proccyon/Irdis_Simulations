#-----Header-----#
#This file finds the data points used in figure c1,c2 and c3 of Holstein et al. 
#The data points are found from calibration measurements, half of which use the calibration polarizer.
#--/--Header--/--#


#-----Imports-----#
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import warnings
import pickle

import Methods as Mt
import Irdis_Model
import IrdapFunctions
#--/--Imports--/--#

warnings.filterwarnings('ignore', category=UserWarning, append=True)

#-----Functions-----#

def CircleMean(Theta1,Theta2):
        '''
        Summary:     
            Takes the mean of two angles, ex. makes sure Mean(7/4pi,0) returns (15/8)pi not (7/8)pi
        Input:
            Theta1: First Angle (radians)
            Theta2: Second Angle (radians), switching Theta1 and Theta2 gives the same result

        Output:
            CircleMean: Mean of the the two angles
        '''
    return np.arctan( (np.sin(Theta1)+np.sin(Theta2)) / (np.cos(Theta1)+np.cos(Theta2)) )
    

def CreateAperture(Shape,x0,y0,R):
        '''
        Summary:     
            Creates a circular aperture array of ones within the circle and 0's outside the circle.
        Input:
            Shape: Shape of the array
            x0: x coordinate of aperture centre
            y0: y coordinate of aperture centre
            R: Radius of aperture

        Output:
            Aperture: 2d aperture array
        '''
    Aperture = np.zeros(Shape)
    for y in range(Shape[0]):
        for x in range(Shape[1]):
            dr = np.sqrt((x-x0)**2 + (y-y0)**2)
            if(dr<=R):
                Aperture[y,x]=1
    
    return Aperture


#--/--Functions--/--#

#-----IrdisCalibrationClass-----#

class IrdisCalibration:
    
    def __init__(self,PolFileList,UnpolFileList,DarkFileF,DarkFileS,FlatFile):
        self.PolFileList = PolFileList
        self.UnpolFileList = UnpolFileList
        self.DarkFileF = DarkFileF
        self.DarkFileS = DarkFileS
        self.FlatFile = FlatFile

        self.HwpTargetList = [(0,45),(11.25,56.25),(22.5,67.5),(33.75,78.75)]
        self.DerTargetList = [(0,45),(11.25,56.25),(22.5,67.5),(33.75,78.75)]
        
        self.ApertureSize=50 #Radius of the apertures
        self.ApertureXList = [250,450,650,450,450,250,250,650,650]#Aperture positions
        self.ApertureYList = [500,500,500,700,300,300,700,300,700]

        #These variables will be saved as text files using the pickle module
        self.PolParamValueArray_FileName = "C:/Users/Gebruiker/Desktop/BRP/Figures/Irdis_Results/PickleData/PicklePolParamValueArray.txt"
        self.UnpolParamValueArray_FileName = "C:/Users/Gebruiker/Desktop/BRP/Figures/Irdis_Results/PickleData/PickleUnpolParamValueArray.txt"
        self.AlternativeParamValueArray_FileName = "C:/Users/Gebruiker/Desktop/BRP/Figures/Irdis_Results/PickleData/PickleAlternativeParamValueArray.txt"

        self.PolDerList_FileName = "C:/Users/Gebruiker/Desktop/BRP/Figures/Irdis_Results/PickleData/PicklePolDerList.txt"
        self.UnpolDerList_FileName = "C:/Users/Gebruiker/Desktop/BRP/Figures/Irdis_Results/PickleData/PickleUnpolDerList.txt"
        self.AlternativeHwpList_FileName = "C:/Users/Gebruiker/Desktop/BRP/Figures/Irdis_Results/PickleData/PickleAlternativeHwpList.txt"
    
    def RunCalibration(self):

        print("Reading files...")
        #Get the images and header parameters of calibration images
        self.RawPolImageList,self.PolDerListTotal,self.PolHwpListTotal,self.PolExpTimeList = self.ReadFileList(PolFileList)
        self.RawUnpolImageList,self.UnpolDerListTotal,self.UnpolHwpListTotal,self.UnpolExpTimeList = self.ReadFileList(UnpolFileList)

        #Get images and header parameters of dark images
        #DarkImageF is the dark image subtracted from the flat, DarkImageS is for the calibration images
        #The dark hwp and der angles are not used for anything
        self.DarkImageListF,DarkDerF,DarkHwpF,self.DarkExpTimeF = self.ReadFile(DarkFileF,True)
        self.DarkImageF = np.mean(self.DarkImageListF,axis=0)
        self.DarkImageListS,DarkDerS,DarkHwpS,self.DarkExpTimeS = self.ReadFile(DarkFileS,True)
        self.DarkImageS = np.mean(self.DarkImageListS,axis=0)

        #Get flat image and header parameters
        self.FlatImage,FlatDer,FlatHwp,self.FlatExpTime = self.ReadFile(FlatFile)

        print("Calibrating files...")
        #Calibrate for noise using the darks and flat
        self.PolImageList,self.PolBadPixelMap = self.CalibrateNoise(self.RawPolImageList,self.PolExpTimeList,self.FlatImage,self.FlatExpTime,self.DarkImageS,self.DarkImageF)
        self.UnpolImageList,self.UnpolBadPixelMap = self.CalibrateNoise(self.RawUnpolImageList,self.UnpolExpTimeList,self.FlatImage,self.FlatExpTime,self.DarkImageS,self.DarkImageF)

        print("Splitting files...")
        #Splits the images in two
        self.PolImageListL,self.PolImageListR = self.SplitImageList(self.PolImageList)
        self.UnpolImageListL,self.UnpolImageListR = self.SplitImageList(self.UnpolImageList)

        print("Creating double difference images...")
        #Creates the double difference images    
        self.PolDDImageArray,self.PolDSImageArray,self.PolDerList = self.CreateHwpDoubleDifferenceImges(self.HwpTargetList,self.PolHwpListTotal,self.PolDerListTotal,self.PolImageListL,self.PolImageListR)
        self.UnpolDDImageArray,self.UnpolDSImageArray,self.UnpolDerList = self.CreateHwpDoubleDifferenceImges(self.HwpTargetList,self.UnpolHwpListTotal,self.UnpolDerListTotal,self.UnpolImageListL,self.UnpolImageListR)
        self.AlternativeDDImageArray,self.AlternativeDSImageArray,self.AlternativeHwpList = self.CreateDerDoubleDifferenceImges(self.DerTargetList,self.UnpolHwpListTotal,self.UnpolDerListTotal,self.UnpolImageListL,self.UnpolImageListR)

        print("Finding parameter values...")
        #Gets double difference values using aperatures
        self.PolParamValueArray = self.GetDoubleDifferenceValue(self.PolDDImageArray,self.PolDSImageArray)
        self.UnpolParamValueArray = self.GetDoubleDifferenceValue(self.UnpolDDImageArray,self.UnpolDSImageArray)
        self.AlternativeParamValueArray = self.GetDoubleDifferenceValue(self.AlternativeDDImageArray,self.AlternativeDSImageArray)

        print("Calculating fitted curve...")
        self.PolParamFitArray,self.PolFitDerList = self.FindFittedStokesParameters(Irdis_Model.BB_H_a,self.HwpTargetList,True)
        self.UnpolParamFitArray,self.UnpolFitDerList = self.FindFittedStokesParameters(Irdis_Model.BB_H_a,self.HwpTargetList,False)
        self.AlternativeParamFitArray,self.AlternativeFitHwpList = self.FindFittedStokesParameters(Irdis_Model.BB_H_a,self.DerTargetList,False,True)

        pickle.dump(self.PolParamValueArray,open(self.PolParamValueArray_FileName,"wb"))
        pickle.dump(self.UnpolParamValueArray,open(self.UnpolParamValueArray_FileName,"wb"))
        pickle.dump(self.AlternativeParamValueArray,open(self.AlternativeParamValueArray_FileName,"wb"))

        pickle.dump(self.PolDerList,open(self.PolDerList_FileName,"wb"))
        pickle.dump(self.UnpolDerList,open(self.UnpolDerList_FileName,"wb"))
        pickle.dump(self.AlternativeHwpList,open(self.AlternativeHwpList_FileName,"wb"))

    #---ReadFunctions---#
    def ReadFile(self,File,IsDark=False):
        '''
        Summary:     
            Reads a given file and returns the images and header data.
            Is used in reading dark and flat files.
        Input:
            File: A .fits file containing irdis data
            IsDark: DarkFiles contain multiple images. If File is a dark set IsDark to True,
            then all images will be returned as a list

        Output:
            RawImage: The unaltered image from the .fits file
            DerAngle: Derotator angle (degrees)
            HwpAngle: Half-wave plate angle (degrees)
            ExpTime: Exposure time (seconds) (is 1s or 2s in this case)
        '''    
        Header = File[0].header
        if(IsDark):
            RawImage = File[0].data
        else:
            RawImage = File[0].data[0]

        Der1 = Header["ESO INS4 DROT2 BEGIN"]*np.pi/180
        Der2 = Header["ESO INS4 DROT2 END"]*np.pi/180
        DerAngle = np.round(CircleMean(Der1,Der2)*180/np.pi,2)
        
        Hwp1 = (Header["ESO INS4 DROT3 BEGIN"]-152.15)*np.pi/180
        Hwp2 = (Header["ESO INS4 DROT3 END"]-152.15)*np.pi/180
        HwpAngle = np.round(CircleMean(Hwp1,Hwp2)*180/np.pi,2)
       
        ExpTime = Header["EXPTIME"]

        return RawImage,DerAngle,HwpAngle,ExpTime

    def ReadFileList(self,FileList):
        '''
        Summary:     
            The same function as ReadFile but now for a list of files
            Is used in reading calibration files.
        Input:
            FileList: List of .fits files containing irdis data.
        Output:
            RawImageList: Unaltered images from the .fits file
            DerAngleList: Derotator angles (degrees)
            HwpAngleList: Half-wave plate angles (degrees)
            ExpTimeList: Exposure times (seconds) (is 1s or 2s in this case)
        '''  

        RawImageList = []
        Derlist = []
        HwpList = []
        ExpTimeList = []
        for File in FileList:
            RawImage,DerAngle,HwpAngle,ExpTime = self.ReadFile(File)
            RawImageList.append(RawImage)
            Derlist.append(DerAngle)
            HwpList.append(HwpAngle)
            ExpTimeList.append(ExpTime)
        
        return np.array(RawImageList),np.array(Derlist),np.array(HwpList),np.array(ExpTimeList)

    #-/-ReadFunctions-/-#

    #---EssentialFunctions---#
    def CalibrateNoise(self,RawImageList,ImageExpTimeList,FlatImage,FlatExpTime,DarkImageS,DarkImageF):
        '''
        Summary:     
            Does dark subtraction and flat scaling. Also uses functions from irdap to do bad pixel correction.
        Input:
            RawImageList: List of images that will be corrected.
            ImageExpTimeList: List of exposure times of raw images. Should be the same for all images.
            FlatImage: Image of flat measurement.
            FlatExpTime: Exposure time of the flat image.
            DarkImageS: Dark image with same exposure time as calibration images.
            DarkImageF: Dark image with same exposure time as flat image.
        Output:
            CalibratedImageList: List of images that have been corrected.
            BadPixelMap: Array of bad pixels. 1 is a bad pixel, 0 a normal pixel(or other way around...?).
        '''
        MasterFlat = IrdapFunctions.process_dark_flat_frames(DarkImageF,FlatImage,FlatExpTime)
        BadPixelMap = IrdapFunctions.create_bpm_darks(DarkImageF) * IrdapFunctions.create_bpm_darks(DarkImageS)
        CalibratedImageList = (1 /ImageExpTimeList[0])*(RawImageList - DarkImageS) / MasterFlat

        return IrdapFunctions.remove_bad_pixels(CalibratedImageList,BadPixelMap),BadPixelMap

    def SplitImageList(self,ImageList):
        '''
        Summary:     
            Splits images in a left and right part.
        Input:
            ImageList: List of images that have already been dark subtracted etc.
        Output:
            ImageL: List of the left part of images.
            ImageR: List of the right part of images.
        '''  

        ImageL = ImageList[:,11:1024, 36:932]
        ImageR = ImageList[:,5:1018, 1062:1958]
        return ImageL,ImageR

    def CreateHwpDoubleDifferenceImages(self,HwpTargetList,TotalHwpList,TotalDerList,ImageListL,ImageListR):
        '''
        Summary:     
            Does double difference method by combining images with hwp angles differing 45 degrees.
            Goes through TotalHwpList to combine images.
        Input:
            HwpTargetList: List of Hwp tuples indicating which hwp angles should be combined.
            TotalHwpList: Hwp angles of images in ImageListL and ImageListR.
            TotalDerList: Derotator angles of images in ImageListL and ImageListR.
            ImageListL: List of the left part of calibration images.
            ImageListR: List of the right part of calibration images.

        Output:
            DDImageArray: Array of double difference images per hwp combination and derotator angle.
            DSImageArray: Array of double sum images per hwp combination and derotator angle.
            DerList: Derotator angles of DDimages,DsImages in DDImageArray and DSImageArray
        '''  

        DDImageArray = []
        DSImageArray = []
        for HwpTarget in HwpTargetList:
            HwpPlusTarget = HwpTarget[0]
            HwpMinTarget = HwpTarget[1]
            DerList = []
            DDImageList = []
            DSImageList = []
            for i in range(len(TotalHwpList)):
                if(TotalHwpList[i] == HwpMinTarget):
                    for j in range(len(TotalHwpList)):
                        if(TotalHwpList[j] == HwpPlusTarget and TotalDerList[i] == TotalDerList[j]):
                            ThetaDer = TotalDerList[i]
                            if(ThetaDer < 0):
                                ThetaDer += 180
                            DerList.append(ThetaDer)
                            PlusDifference = ImageListL[j]-ImageListR[j]
                            MinDifference = ImageListL[i]-ImageListR[i]
                            PlusSum = ImageListL[j]+ImageListR[j]
                            MinSum = ImageListL[i]+ImageListR[i]
                            DDImage = 0.5*(PlusDifference - MinDifference) 
                            DSImage = 0.5*(PlusSum + MinSum)
                            DDImageList.append(DDImage)
                            DSImageList.append(DSImage)
                            break
                            
            DDImageArray.append(np.array(DDImageList))
            DSImageArray.append(np.array(DSImageList))

        return np.array(DDImageArray),np.array(DSImageArray),np.array(DerList)

    #Same as the above function but combines images differing 45 degrees derotator angle
    def CreateDerDoubleDifferenceImges(self,DerTargetList,TotalHwpList,TotalDerList,ImageListL,ImageListR):
        '''
        Summary:     
            Does double difference method by combining images with derotator angles differing 45 degrees.
            Goes through TotalDerList to combine images.
        Input:
            DerTargetList: List of derotator tuples indicating which derotator angles should be combined.
            TotalHwpList: Hwp angles of images in ImageListL and ImageListR.
            TotalDerList: Derotator angles of images in ImageListL and ImageListR.
            ImageListL: List of the left part of calibration images.
            ImageListR: List of the right part of calibration images.

        Output:
            DDImageArray: Array of double difference images per derotator combination and hwp angle.
            DSImageArray: Array of double sum images per derotator combination and hwp angle.
            HwpList: Hwp angles of DDimages,DsImages in DDImageArray and DSImageArray
        ''' 

        DDImageArray = []
        DSImageArray = []
        for DerTarget in DerTargetList:
            DerPlusTarget = DerTarget[0]
            DerMinTarget = DerTarget[1]
            HwpList = []
            DDImageList = []
            DSImageList = []
            for i in range(len(TotalDerList)):
                if(TotalDerList[i] == DerMinTarget):
                    for j in range(len(TotalDerList)):
                        if(TotalDerList[j] == DerPlusTarget and TotalHwpList[i] == TotalHwpList[j]):
                            ThetaHwp = TotalHwpList[i]
                            if(ThetaHwp < 0):
                                ThetaHwp += 180
                            HwpList.append(ThetaHwp)
                            PlusDifference = ImageListL[j]-ImageListR[j]
                            MinDifference = ImageListL[i]-ImageListR[i]
                            PlusSum = ImageListL[j]+ImageListR[j]
                            MinSum = ImageListL[i]+ImageListR[i]
                            DDImage = 0.5*(PlusDifference - MinDifference) 
                            DSImage = 0.5*(PlusSum + MinSum)
                            DDImageList.append(DDImage)
                            DSImageList.append(DSImage)
                            break
                            
            DDImageArray.append(np.array(DDImageList))
            DSImageArray.append(np.array(DSImageList))

        return np.array(DDImageArray),np.array(DSImageArray),np.array(HwpList)


    #Uses aperatures to get a single value for the double differance
    def GetDoubleDifferenceValue(self,DDImageArray,DSImageArray):
        '''
        Summary:     
            Finds a (float) normalized parameter value from the double difference and sum images.
            Uses 9 circular apertures to find 9 different (but similar)values.
        Input:
            DDImageArray: Array of double difference images 
            DSImageArray: Array of double sum images
        Output:
            ParamValueArray: 3d array that lists parameter values per aperture, hwp combination and derotator angle.
        ''' 

        ParamValueArray = []
        for i in range(len(self.ApertureXList)):
            ApertureX = self.ApertureXList[i]
            ApertureY = self.ApertureYList[i] 
            Shape = DDImageArray[0][0].shape
            Aperture = CreateAperture(Shape,ApertureX,ApertureY,self.ApertureSize)
            ParamValue = np.nanmedian(DDImageArray[:,:,Aperture==1],axis=2) / np.nanmedian(DSImageArray[:,:,Aperture==1],axis=2)
            ParamValueArray.append(ParamValue)

        return np.array(ParamValueArray)
        

    def FindFittedStokesParameters(self,Model,HwpTargetList,UsePolarizer,DerMethod=False):
        '''
        Summary:     
            Finds parameter values expected from the model fitted in Holstein et al.
            Uses methods in Irdis_Model.py
        Input:
            Model: Class defined in Irdis_Model.py. Spefifies model parameters used.
            HwpTargetList: List of hwp anlges that are combined in double difference.
            UsePolarizer: Wheter or not the calibration polarizer is used.
            DerMethod: If true, double difference is taken with der angles differing 45 degrees instead of hwp angles.
        Output:
            ParamFitValueArray: 2d array that lists parameter values per hwp combination and derotator angle.
            FitDerList: List of derotator angles of parameter values in ParamFitValueArray
        ''' 

        FitDerList = np.linspace(-2*np.pi/180,103.25*np.pi/180,200)    
        ParmFitValueArray = []
        S_In = np.array([1,0.009480,0.000406,0]) #Not completely unpolarized because of M4, see q_in and u_in

        for HwpTarget in HwpTargetList:
            HwpPlusTarget = HwpTarget[0]*np.pi/180
            HwpMinTarget = HwpTarget[1]*np.pi/180
            ParmFitValueList = []
            for FitDer in FitDerList:
                X_Matrix,I_Matrix = Model.ParameterMatrix(HwpPlusTarget,HwpMinTarget,FitDer,0,UsePolarizer,True,DerMethod)
                X_Out = np.dot(X_Matrix,S_In)[0]
                I_Out = np.dot(I_Matrix,S_In)[0]
                X_Norm = X_Out/I_Out
                ParmFitValueList.append(X_Norm)
        
            ParmFitValueArray.append(np.array(ParmFitValueList))
        
        return np.array(ParmFitValueArray),FitDerList


    #-/-EssentialFunctions-/-#

    #---PlotFunctions---#
    #Shows one of the raw images. 
    def ShowRawImage(self,ImageNumber,FromPolarizedImages=True):
        plt.figure()
        plt.xlabel("x(pixels)")
        plt.ylabel("y(pixels)")
        
        if(FromPolarizedImages):
            Image = self.RawPolImageList[ImageNumber]
            plt.title("Raw polarized calibration image")
        else:
            Image = self.RawUnpolImageList[ImageNumber]
            plt.title("Raw unpolarized calibration image")

        plt.imshow(Image,vmin=0.7*np.median(Image),vmax=1.3*np.median(Image))
        plt.colorbar()

    def ShowDark(self,DarkNumber):

        plt.figure()
        plt.title("Dark used with science measurement")
        plt.xlabel("x(pixels)")
        plt.ylabel("y(pixels)")
        Image = self.DarkImageListS[DarkNumber]
        plt.imshow(Image,vmin=0.7*np.mean(Image),vmax=1.3*np.mean(Image))
        plt.colorbar()

        plt.figure()
        plt.title("Dark used with flat measurement")
        plt.xlabel("x(pixels)")
        plt.ylabel("y(pixels)")
        Image = self.DarkImageListF[DarkNumber]
        plt.imshow(Image,vmin=0.7*np.mean(Image),vmax=1.3*np.mean(Image))
        plt.colorbar()

    #Shows one of the raw images. 
    def ShowCalibratedImage(self,ImageNumber,FromPolarizedImages=True):
        plt.figure()
        plt.xlabel("x(pixels)")
        plt.ylabel("y(pixels)")
        
        if(FromPolarizedImages):
            Image = self.PolImageList[ImageNumber]
            plt.title("Dark,Flat calibrated polarized calibration image")
        else:
            Image = self.UnpolImageList[ImageNumber]
            plt.title("Dark,Flat calibrated unpolarized calibration image")

        plt.imshow(Image,vmin=0.7*np.nanmean(Image),vmax=1.3*np.nanmean(Image))
        plt.colorbar()
            
    def ShowBadPixelMap(self):
        plt.figure()
        plt.xlabel("x(pixels)")
        plt.ylabel("y(pixels)")
        plt.title("Bad pixel map of polarized images")
        plt.imshow(self.PolBadPixelMap)

        plt.figure()
        plt.xlabel("x(pixels)")
        plt.ylabel("y(pixels)")
        plt.title("Bad pixel map of unpolarized images")
        plt.imshow(self.UnpolBadPixelMap)


    #Shows one of the raw images. 
    def ShowLeftRightImage(self,ImageNumber,FromPolarizedImages=True):

        if(FromPolarizedImages):
            plt.figure()
            plt.title("Left polarized image")
            plt.xlabel("x(pixels)")
            plt.ylabel("y(pixels)")
            plt.imshow(self.PolImageListL[ImageNumber],vmin=4.8E3,vmax=5.2E3)
            plt.colorbar()

            plt.figure()
            plt.title("Right polarized image")
            plt.xlabel("x(pixels)")
            plt.ylabel("y(pixels)")
            plt.imshow(self.PolImageListR[ImageNumber],vmin=4.8E3,vmax=5.2E3)
            plt.colorbar()

            
        else:
            plt.figure()
            plt.title("Left polarized image")
            plt.xlabel("x(pixels)")
            plt.ylabel("y(pixels)")
            plt.imshow(self.UnpolImageListL[ImageNumber],vmin=4.8E3,vmax=5.2E3)
            plt.colorbar()

            plt.figure()
            plt.title("Right polarized image")
            plt.xlabel("x(pixels)")
            plt.ylabel("y(pixels)")
            plt.imshow(self.UnpolImageListR[ImageNumber],vmin=4.8E3,vmax=5.2E3)
            plt.colorbar()

    
    #Shows a double difference or sum image
    def ShowDoubleDifferenceImage(self,ImageNumber,HwpNumber,FromPolarizedImages=True,DoubleSum=False,ShowApertures=False):
        plt.figure()
        plt.xlabel("x(pixels)")
        plt.ylabel("y(pixels)")
        PlottedImage = []

        if(DoubleSum):
            if(FromPolarizedImages):
                PlottedImage = self.PolDSImageArray[HwpNumber][ImageNumber]
                plt.title("Polarized double sum image")
            else:
                PlottedImage = self.UnpolDSImageArray[HwpNumber][ImageNumber]
                plt.title("Unpolarized double sum image")
        else:
            if(FromPolarizedImages):
                PlottedImage = self.PolDDImageArray[HwpNumber][ImageNumber]
                plt.title("Polarized double difference image")
            else:
                PlottedImage = self.UnpolDDImageArray[HwpNumber][ImageNumber]
                plt.title("Unpolarized double difference image")

        if(ShowApertures):
            for i in range(len(self.ApertureXList)):
                ApertureX = self.ApertureXList[i]
                ApertureY = self.ApertureYList[i]
                Aperture = CreateAperture(PlottedImage.shape,ApertureX,ApertureY,self.ApertureSize)
                PlottedImage*= (Aperture == 0)

        
        plt.imshow(PlottedImage,vmin=np.nanmean(PlottedImage)/1.2,vmax=np.nanmean(PlottedImage)*1.2)
        plt.colorbar()
        

    def PlotStokesParameters(self,ColorList,ImageType=0):
        
        plt.figure()
        plt.ylabel("Normalized Stokes parameter (%)")
        plt.xticks(np.arange(-11.25,112.5,11.25))
        plt.axhline(y=0,color="black")

        if(ImageType==0):      
            plt.title("Stokes parameter vs derotator angle (polarizer)")           
            plt.xlabel("Derotator angle(degrees)")
            plt.yticks(np.arange(-120,120,20))
            plt.xlim(left=-1,right=91)
            plt.ylim(bottom=-110,top=110)

            for i in range(len(self.HwpTargetList)):
                #Plot data
                HwpPlusTarget = self.HwpTargetList[i][0]
                ParamValueList = 100*self.PolParamValueArray[:,i,:]
                plt.scatter(self.PolDerList*np.ones(ParamValueList.shape),ParamValueList,label="HwpPlus = "+str(HwpPlusTarget),zorder=100,color=ColorList[i],s=18,edgecolors="black")

                #Plot fitted curve
                plt.plot(self.PolFitDerList*180/np.pi,self.PolParamFitArray[i]*100,color=ColorList[i])

        if(ImageType==1):
            plt.title("Stokes parameter vs derotator angle (no polarizer)")           
            plt.xlabel("Derotator angle(degrees)")
            plt.yticks(np.arange(-1.2,1.2,0.2))
            plt.xlim(left=-1,right=102.25)
            plt.ylim(bottom=-1.1,top=1.1)

            for i in range(len(self.HwpTargetList)):
                #Plot data
                HwpPlusTarget = self.HwpTargetList[i][0]
                ParamValueList = 100*self.UnpolParamValueArray[:,i,:]
                plt.scatter(self.UnpolDerList*np.ones(ParamValueList.shape),ParamValueList,label="HwpPlus = "+str(HwpPlusTarget),zorder=100,color=ColorList[i],s=18,edgecolors="black")

                #Plot fitted curve
                plt.plot(self.UnpolFitDerList*180/np.pi,self.UnpolParamFitArray[i]*100,color=ColorList[i])

        if(ImageType==2):
            plt.title("Stokes parameter vs hwp angle (no polarizer)")           
            plt.xlabel("Hwp angle(degrees)")
            plt.yticks(np.arange(-0.96,0.96,0.16))
            plt.xlim(left=-1,right=102.25)
            plt.ylim(bottom=-0.88,top=0.88)

            for i in range(len(self.DerTargetList)):
                #Plot data
                DerPlusTarget = self.DerTargetList[i][0]
                ParamValueList = 100*self.AlternativeParamValueArray[:,i,:]
                plt.scatter(self.AlternativeHwpList*np.ones(ParamValueList.shape),ParamValueList,label="DerPlus = "+str(DerPlusTarget),zorder=100,color=ColorList[i],s=18,edgecolors="black")

                #Plot fitted curve
                plt.plot(self.AlternativeFitHwpList*180/np.pi,self.AlternativeParamFitArray[i]*100,color=ColorList[i])

        plt.grid(linestyle="--")
        plt.legend(fontsize=8)


    #---PlotFunctions---#

#--/--IrdisCalibrationClass--/--#

#-----Parameters-----#
Prefix = "C:/Users/Gebruiker/Desktop/BRP/CalibrationData/Internal Source/raw/irdis_internal_source_h/"
FileType=".fits"

PrefixUnpol = Prefix+"SPHERE_IRDIS_TEC165_0"
PrefixPol = Prefix+"SPHERE_IRDIS_TEC227_0"
DarkNameF = Prefix+"SPHERE_IRDIS_CAL_DARK164_0002"+FileType
DarkNameS = Prefix+"SPHERE_IRDIS_CAL_DARK164_0003"+FileType
FlatName = Prefix+"SPHERE_IRDIS_CAL_FLAT165_0002"+FileType

NumberListUnpol= np.arange(368,568,2)
NumberListPol = np.arange(400,664,2)

#--/--Parameters--/--#

#-----FindFiles-----#
DarkFileF = fits.open(DarkNameF)
DarkFileS = fits.open(DarkNameS)
FlatFile = fits.open(FlatName)

PolFileList  = []
for Number in NumberListPol:
    PolFile = fits.open(PrefixPol+str(Number)+".fits")
    PolFileList.append(PolFile)

UnpolFileList  = []
for Number in NumberListUnpol:
    UnpolFile = fits.open(PrefixUnpol+str(Number)+".fits")
    UnpolFileList.append(UnpolFile)

#--/--FindFiles--/--#

#-----Main-----#

IrdisCalibrationObject = IrdisCalibration(PolFileList,UnpolFileList,DarkFileF,DarkFileS,FlatFile)
IrdisCalibrationObject.RunCalibration()

IrdisCalibrationObject.PlotStokesParameters(["blue","lightblue","red","orange"],0)
IrdisCalibrationObject.PlotStokesParameters(["blue","lightblue","red","orange"],1)
IrdisCalibrationObject.PlotStokesParameters(["blue","lightblue","red","orange"],2)

IrdisCalibrationObject.ShowDark(0)
IrdisCalibrationObject.ShowDoubleDifferenceImage(0,0,False,True,True)

IrdisCalibrationObject.ShowRawImage(0)
IrdisCalibrationObject.ShowCalibratedImage(0)
IrdisCalibrationObject.ShowBadPixelMap()

plt.show()
#--/--Main--/--#

