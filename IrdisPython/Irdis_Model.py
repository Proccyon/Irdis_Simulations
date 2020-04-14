import matplotlib.pyplot as plt
import numpy as np
import Methods as Mt

#-----Classes-----#

#The optical system of sphere for a specific wavelength
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

        
    #Does intensity measurement with given system parameters
    def IntensityMatrix(self,ThetaHwp,D_Sign,ThetaDer,Altitude,UsePolarizer=False,UseSource=False):
        M_CI = 0.5*Mt.ComMatrix(self.d*D_Sign,0) #Polarizer for double difference
        
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

        #TEST - REMOVE LATER
        #M_M4 = Mt.ComMatrix(0.00985,175*np.pi/180)

        if(UseSource): #If using internal calibration source
 
            if(UsePolarizer):
                return np.linalg.multi_dot([M_CI,T_DerMin,M_Der,T_DerPlus,T_HwpMin,M_Hwp,T_HwpPlus,T_Cal,M_Polarizer])
            else:
                return np.linalg.multi_dot([M_CI,T_DerMin,M_Der,T_DerPlus,T_HwpMin,M_Hwp,T_HwpPlus])
        else:

            if(UsePolarizer):
                return np.linalg.multi_dot([M_CI,T_DerMin,M_Der,T_DerPlus,T_HwpMin,M_Hwp,T_HwpPlus,T_Cal,M_Polarizer,Ta,M_UT])
            else:
                return np.linalg.multi_dot([M_CI,T_DerMin,M_Der,T_DerPlus,T_HwpMin,M_Hwp,T_HwpPlus,Ta,M_UT])
        

    #Calculates a stokes parameter using the double difference method
    def ParameterMatrix(self,ThetaHwpPlus,ThetaHwpMin,ThetaDer,Altitude,UsePolarizer=False,UseSource=False,DerotatorMethod=False):
        IntensityMatrix = lambda ThetaHwp,D_Sign : self.IntensityMatrix(ThetaHwp,D_Sign,ThetaDer,Altitude,UsePolarizer,UseSource)
        if(DerotatorMethod):
            IntensityMatrix = lambda ThetaHwp,D_Sign : self.IntensityMatrix(ThetaDer,D_Sign,ThetaHwp,Altitude,UsePolarizer,UseSource)
        
        #Find stokes Q
        XPlus = IntensityMatrix(ThetaHwpPlus,1) - IntensityMatrix(ThetaHwpPlus,-1) #Single difference for two hwp angles
        XMin = IntensityMatrix(ThetaHwpMin,1) - IntensityMatrix(ThetaHwpMin,-1)

        IPlus = IntensityMatrix(ThetaHwpPlus,1) + IntensityMatrix(ThetaHwpPlus,-1) #Single sum for two hwp angles
        IMin = IntensityMatrix(ThetaHwpMin,1) + IntensityMatrix(ThetaHwpMin,-1)
        X_Matrix = 0.5*(XPlus-XMin) #Double difference
        I_Matrix = 0.5*(IPlus+IMin) #Double sum

        return X_Matrix, I_Matrix
    

    #Performs double difference with given system parameters
    def DoubleDifferenceMatrix(self,ThetaDer,Altitude,UsePolarizer=False):
        
        #Find stokes Q and IQ matrices
        Q_Matrix,IQ_Matrix = self.ParameterMatrix(0,(1/4)*np.pi,ThetaDer,Altitude,UsePolarizer)

        #Find stokes Q and IQ matrices
        U_Matrix,IU_Matrix = self.ParameterMatrix((1/8)*np.pi,(3/8)*np.pi,ThetaDer,Altitude,UsePolarizer)

        I_Matrix = 0.5*(IQ_Matrix+IU_Matrix) #IQ should be the same as IU but we average anyway
        
        PolarizationMatrix = np.zeros((4,4))
        PolarizationMatrix[0,0:3] = I_Matrix[0,0:3]
        PolarizationMatrix[1,0:3] = Q_Matrix[0,0:3]
        PolarizationMatrix[2,0:3] = U_Matrix[0,0:3]
        
        return PolarizationMatrix

    
    #Returns an array of normalized stokes parameters over derotator angle.
    #Double difference is done using hwp combinations in HwpTargetList
    def FindParameterArray(self,DerList,S_In,Altitude=0,HwpTargetList=[(0,45),(11.25,56.25),(22.5,67.5),(33.75,78.75)],UsePolarizer=False,DerMethod=False):
 
        ParmFitValueArray = []

        for HwpTarget in HwpTargetList:
            HwpPlusTarget = HwpTarget[0]*np.pi/180
            HwpMinTarget = HwpTarget[1]*np.pi/180
            ParmFitValueList = []
            for Der in DerList:
                X_Matrix,I_Matrix = self.ParameterMatrix(HwpPlusTarget,HwpMinTarget,Der,Altitude,UsePolarizer,True,DerMethod)
                X_Out = np.dot(X_Matrix,S_In)[0]
                I_Out = np.dot(I_Matrix,S_In)[0]
                X_Norm = X_Out/I_Out
                ParmFitValueList.append(X_Norm)
        
            ParmFitValueArray.append(np.array(ParmFitValueList))
        
        return np.array(ParmFitValueArray)


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