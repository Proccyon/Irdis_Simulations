#-----Header-----#
#This file contains a Mueller matrix model for Irdis/SPHERE.
#A class MatrixModel is created that stores all model parameters.
#The class simulates the propagation of light through the system.
#--/--Header--/--#

#-----Imports-----#
import matplotlib.pyplot as plt
import numpy as np
import Methods as Mt
#--/--Imports--/--#

#-----Classes-----#

#The optical system of SPHERE for a specific wavelength
class MatrixModel():
    
    def __init__(self,Name,E_Hwp,R_Hwp,DeltaHwp,E_Der,R_Der,DeltaDer,d,DeltaCal,E_UT,R_UT):
        self.Name = Name #Name of filter
        self.E_Hwp = E_Hwp #Diattenuation of half-waveplate
        self.R_Hwp = R_Hwp *np.pi/180 #Retardance of half-waveplate
        self.DeltaHwp = DeltaHwp*np.pi/180 #Offset of half-waveplate
        self.E_Der = E_Der #Diattenuation of derotator
        self.R_Der = R_Der*np.pi/180 #Retardance of derotator
        self.DeltaDer = DeltaDer*np.pi/180 #Offset of derotator
        self.d = d #Diattenuation of polarizers
        self.DeltaCal = DeltaCal*np.pi/180 #Offset of calibration polarizer
        
        self.E_UT = E_UT
        self.R_UT = R_UT *np.pi/180

        
    def MakeIntensityMatrix(self,ThetaHwp,D_Sign,ThetaDer,Altitude,UsePolarizer=False,UseSource=False):
        '''
        Summary:     
            Creates the total mueller matrix of the system(No single/double difference).
            This matrix can only be used to measure the intensity of outcoming light for a 
            specific setup.
        Input:
            ThetaHwp: Half-waveplate angle (radians)
            D_Sign: Sign of the P0-90 polarizer matrix, should be either 1 or -1
            ThetaDer: Derotator angle (radians)
            Altitude: Altitude angle (radians). Not used when UseSource is True
            UsePolarizer: Wheter or not the calibration polarizer is used.
            UseSource: Wheter or not the internal light source is used. 
            When used S_In needs to take Mirror 4 into acount.
        Output:
            IntensityMatrix: Use like this --> Intensity = np.dot(IntensityMatrix,S_In)[0]
            where intensity is a float and S_In is the incoming stokes vector.
            Don't take [1],[2] or [3], these are not actually measured.
        '''
        M_CI = 0.5*Mt.ComMatrix(self.d*D_Sign,0) #Polarizer for single difference
        
        T_DerMin = Mt.RotationMatrix(-(ThetaDer+self.DeltaDer)) #Derotator with rotation
        M_Der = Mt.ComMatrix(self.E_Der,self.R_Der)
        T_DerPlus = Mt.RotationMatrix(ThetaDer+self.DeltaDer)
        
        T_HwpMin = Mt.RotationMatrix(-(ThetaHwp+self.DeltaHwp)) #Half-waveplate with rotation
        M_Hwp = Mt.ComMatrix(self.E_Hwp,self.R_Hwp)
        T_HwpPlus = Mt.RotationMatrix(ThetaHwp+self.DeltaHwp)
        
        T_Cal = Mt.RotationMatrix(-self.DeltaCal)#Optional polarizer      
        M_Polarizer = 0.5*Mt.ComMatrix(self.d,0) 
        
        Ta = Mt.RotationMatrix(Altitude) #Telescope mirrors with rotation
        M_UT = Mt.ComMatrix(self.E_UT,self.R_UT)

        IntensityMatrix
        if(UseSource): #If using internal calibration source
 
            if(UsePolarizer):
                IntensityMatrix = np.linalg.multi_dot([M_CI,T_DerMin,M_Der,T_DerPlus,T_HwpMin,M_Hwp,T_HwpPlus,T_Cal,M_Polarizer])
            else:
                IntensityMatrix = np.linalg.multi_dot([M_CI,T_DerMin,M_Der,T_DerPlus,T_HwpMin,M_Hwp,T_HwpPlus])
        else:
            if(UsePolarizer):
                IntensityMatrix = np.linalg.multi_dot([M_CI,T_DerMin,M_Der,T_DerPlus,T_HwpMin,M_Hwp,T_HwpPlus,T_Cal,M_Polarizer,Ta,M_UT])
            else:
                IntensityMatrix = np.linalg.multi_dot([M_CI,T_DerMin,M_Der,T_DerPlus,T_HwpMin,M_Hwp,T_HwpPlus,Ta,M_UT])

        return IntensityMatrix
        

    def MakeParameterMatrix(self,ThetaHwpPlus,ThetaHwpMin,ThetaDer,Altitude,UsePolarizer=False,UseSource=False,DerotatorMethod=False):

        '''
        Summary:     
            Creates a matrix that is used to find a stokes parameter (Q or U).
            Parameter matrix is found using double difference method.
        Input:
            ThetaHwpPlus: Smallest Half-wave plate angle (radians). 0 for Q, (1/8)pi for U.
            ThetaHwpMin: Biggest Half-wave plate angle (radians). (1/2)pi for Q, (3/8)pi for U.
            ThetaDer: Derotator angle (radians).
            Altitude: Altitude angle (radians).
            UsePolarizer: Wheter or not the calibration polarizer is used.
            UseSource: Wheter or not the internal light source is used
            DerotatorMethod: If true then take double difference using measurements 
            with different derotator angles. ThetaHwpPlus and ThetaHwpMin are now derotator angles.
            ThetaDer is now a Hwp angle.

        Output:
            X_Matrix: Use Like this --> X = np.dot(X_Matrix,S_In)[0]
            This returns a stokes parameter like Q or U.
            I_Matrix: Use Like this --> I = np.dot(I_Matrix,S_In)[0]
            X/I is then the normalized stokes parameter.
        '''

        MakeIntensityMatrix = lambda ThetaHwp,D_Sign : self.MakeIntensityMatrix(ThetaHwp,D_Sign,ThetaDer,Altitude,UsePolarizer,UseSource)
        if(DerotatorMethod):
            MakeIntensityMatrix = lambda ThetaHwp,D_Sign : self.MakeIntensityMatrix(ThetaDer,D_Sign,ThetaHwp,Altitude,UsePolarizer,UseSource)
        
        #Find stokes Q
        XPlus = MakeIntensityMatrix(ThetaHwpPlus,1) - MakeIntensityMatrix(ThetaHwpPlus,-1) #Single difference for two hwp angles
        XMin = MakeIntensityMatrix(ThetaHwpMin,1) - MakeIntensityMatrix(ThetaHwpMin,-1)

        IPlus = IntensityMatrix(ThetaHwpPlus,1) + MakeIntensityMatrix(ThetaHwpPlus,-1) #Single sum for two hwp angles
        IMin = MakeIntensityMatrix(ThetaHwpMin,1) + MakeIntensityMatrix(ThetaHwpMin,-1)
        X_Matrix = 0.5*(XPlus-XMin) #Double difference
        I_Matrix = 0.5*(IPlus+IMin) #Double sum

        return X_Matrix, I_Matrix
    
    def DoubleDifferenceMatrix(self,ThetaDer,Altitude,UsePolarizer=False,UseSource=False):
        '''
        Summary:     
            Create a total Mueller matrix acounting for the double difference.
            This way S_Out = np.dot(PolarizationMatrix,S_In) where we can fully
            use S_Out except for stokes V(S_Out[3]).
        Input:
            ThetaDer: Derotator angle (radians).
            Altitude: Altitude angle (radians).
            UsePolarizer: Wheter or not the calibration polarizer is used.
            UseSource: Wheter or not the internal light source is used

        Output:
            Polarization Matrix: 4x4 matrix where every entry related to V is 0.
        '''
        
        #Find stokes Q and IQ matrices
        Q_Matrix,IQ_Matrix = self.ParameterMatrix(0,(1/4)*np.pi,ThetaDer,Altitude,UsePolarizer,UseSource)

        #Find stokes Q and IQ matrices
        U_Matrix,IU_Matrix = self.ParameterMatrix((1/8)*np.pi,(3/8)*np.pi,ThetaDer,Altitude,UsePolarizer,UseSource)

        I_Matrix = 0.5*(IQ_Matrix+IU_Matrix) #IQ should be the same as IU but we average anyway
        
        PolarizationMatrix = np.zeros((4,4))
        PolarizationMatrix[0,0:3] = I_Matrix[0,0:3]
        PolarizationMatrix[1,0:3] = Q_Matrix[0,0:3]
        PolarizationMatrix[2,0:3] = U_Matrix[0,0:3]
        
        return PolarizationMatrix

    #Double difference is done using hwp combinations in HwpTargetList
    def FindParameterArray(self,DerList,S_In,Altitude=0,HwpTargetList=[(0,45),(11.25,56.25),(22.5,67.5),(33.75,78.75)],UsePolarizer=False,DerMethod=False):
 
         '''
        Summary:     
            Returns a 2d array of normalized stokes parameters over derotator angle. 
            This is used to make plots c1,c2,c3 in Holstein et al.
        Input:
            DerList: Derotator angles (radians).
            S_In: Incoming stokes vector. 
            Altitude: Altitude angle (radians).
            HwpTargetList: Each tupple in this list is two hwp angles with which to do the double difference.
            UsePolarizer: Wheter or not the calibration polarizer is used.
            DerMethod: If True instead array is over hwp angles. HwpTargetList is then DerTargetList.

        Output:
            ParmValueArray: 2d array of normalized parameter values. First dimension is which Hwp combination is used.
            second dimension is derotator angle.
        '''
        ParmValueArray = []

        for HwpTarget in HwpTargetList:
            HwpPlusTarget = HwpTarget[0]*np.pi/180
            HwpMinTarget = HwpTarget[1]*np.pi/180
            ParmValueList = []
            for Der in DerList:
                X_Matrix,I_Matrix = self.ParameterMatrix(HwpPlusTarget,HwpMinTarget,Der,Altitude,UsePolarizer,True,DerMethod)
                X_Out = np.dot(X_Matrix,S_In)[0]
                I_Out = np.dot(I_Matrix,S_In)[0]
                X_Norm = X_Out/I_Out
                ParmValueList.append(X_Norm)
        
            ParmValueArray.append(np.array(ParmValueList))
        
        return np.array(ParmValueArray)


    #Measures stokes vector for different derotator angles
    def RunPE_Measurement(self,Steps):
        I_Out = []
        Q_Out = []
        U_Out = []
        S_In = np.array([1,0,0,0])
        ThetaDerList = np.linspace(0,np.pi,Steps) #List of derotator angles

        for ThetaDer in ThetaDerList:
            TotalMatrix = self.DoubleDifferenceMatrix(ThetaDer,0,True)
            S_Out = np.dot(TotalMatrix,S_In)
            I_Out.append(S_Out[0])
            Q_Out.append(S_Out[1])
            U_Out.append(S_Out[2])

        return np.array(I_Out),np.array(Q_Out),np.array(U_Out)

        #Measures stokes vector for different derotator angles
    def RunIP_Measurement(self,Steps):
        IOut = []
        QOut = []
        UOut = []
        S_In = np.array([1,0,0,0])
        AltitudeList = np.linspace(0,0.5*np.pi,Steps) #List of derotator angles

        for Altitude in AltitudeList:
            TotalMatrix = self.DoubleDifferenceMatrix(0,Altitude,False)
            S_Out = np.dot(TotalMatrix,S_In)
            IOut.append(S_Out[0])
            QOut.append(S_Out[1])
            UOut.append(S_Out[2])

        return np.array(IOut),np.array(QOut),np.array(UOut)

#--/--Classes--/--#

#-----Methods-----#
#Plots Polarimetric efficiency vs derotator angle found using setup 1
def PlotEfficiencyDiagram(ModelList,Steps,PlotStokes=False):
    
    ThetaDerList = np.linspace(0,180,Steps) #List of derotator angles

    plt.figure()

    plt.title("Polarimetric efficiency vs Derotator angle")
    plt.xlabel("Derotator angle")
    plt.ylabel("Polarimetric efficiency")

    plt.xlim(left=0,right=180)
    plt.ylim(bottom=0,top=1.1)
    plt.xticks(np.arange(0,180,step=22.5))
    plt.yticks(np.arange(0,1.1,step=0.1))   
    
    plt.grid() 
 
    for Model in ModelList:
        I_Out,Q_Out,U_Out = Model.RunPE_Measurement(Steps)
        PE = np.sqrt(Q_Out**2 + U_Out**2) / I_Out
        plt.plot(ThetaDerList,PE, label=Model.Name)    
    
    plt.legend()
    

def PlotIpDiagram(ModelList,Steps,PlotStokes=False):
    
    AltitudeList = np.linspace(0,90,Steps) #List of derotator angles

    plt.figure()
    plt.title("Altitude vs IP")
    plt.xlabel("Altitude")
    plt.ylabel("Instrumental polarization(%)")

    plt.xlim(left=0,right=90)
    plt.ylim(bottom=0,top=5)
    plt.xticks(np.arange(0,90,step=10))
    plt.yticks(np.arange(0,5,step=0.5))   
    
    plt.grid() 

    for Model in ModelList:
        I_Out,Q_Out,U_Out = Model.RunIP_Measurement(Steps)
        IP = 100*np.sqrt(Q_Out**2 + U_Out**2) / I_Out     
        plt.plot(AltitudeList,IP, label=Model.Name)

    plt.legend()
    
    if(PlotStokes):
        plt.figure()

        plt.title("SCExAO Stokes Q and U over altitude")
        plt.xlabel("Altitude")
        plt.ylabel("Parameter value")
    
        plt.xlim(left=0,right=90)
        plt.ylim(bottom=-0.1,top=0.1)
        plt.xticks(np.arange(0,90,step=10))
        plt.yticks(np.arange(-0.1,0.1,step=0.02))   
        
        plt.grid() 
    
        #print(Q_Out)
        plt.plot(AltitudeList,Q_Out, label=Model.Name+"_Q")
        plt.plot(AltitudeList,U_Out, label=Model.Name+"_U")
    
        plt.legend()
    
    
#--/--Methods--/--#

#-----Main-----#
Steps = 100
BB_Y = MatrixModel("BB_Y",-0.00021,184.2,-0.6132,-0.00094,126.1,0.50007,0.9802,-1.542,0.0236,171.9)
BB_J = MatrixModel("BB_J",-0.000433,177.5,-0.6132,-0.008304,156.1,0.50007,0.9895,-1.542,0.0167,173.4)
BB_H = MatrixModel("BB_H",-0.000297,170.7,-0.6132,-0.002260,99.32,0.50007,0.9955,-1.542,0.01293,175)
BB_K = MatrixModel("BB_K",-0.000415,177.6,-0.6132,0.003552,84.13,0.50007,0.9842,-1.542,0.0106,176.3)

BB_H_a = MatrixModel("BB_H",-0.000297,170.7,-0.6132,-0.002260,99.32,0.50007,0.9955,-1.542,0.0090,175)

ModelList = [BB_Y,BB_J,BB_H,BB_K]
#PlotEfficiencyDiagram(ModelList,Steps)
#PlotIpDiagram(ModelList,Steps,True)
#plt.show()
#--/--Main--/--#