import aslam_backend as aopt
import aslam_splines as asp
from . import IccUtil as util
import incremental_calibration as inc
import kalibr_common as kc
import sm

import gc
import numpy as np
import multiprocessing
import sys

# make numpy print prettier
np.set_printoptions(suppress=True)

CALIBRATION_GROUP_ID = 0
HELPER_GROUP_ID = 1

def addSplineDesignVariables(problem, dvc, setActive=True, group_id=HELPER_GROUP_ID):
    for i in range(0,dvc.numDesignVariables()):
        dv = dvc.designVariable(i)
        dv.setActive(setActive)
        problem.addDesignVariable(dv, group_id)

def getPartialBlocks(bsm, brows):
    return np.block([[bsm.getBlock(r, c) for c in brows] for r in brows])

class IccCalibrator(object):
    def __init__(self):
        self.ImuList = []

    def initDesignVariables(self, problem, poseSpline, noTimeCalibration, noChainExtrinsics=True, \
                            estimateGravityLength=False, initialGravityEstimate=np.array([0.0,9.81,0.0])):        
        # Initialize the system pose spline (always attached to imu0) 
        self.poseDv = asp.BSplinePoseDesignVariable( poseSpline )
        addSplineDesignVariables(problem, self.poseDv)

        # Add the calibration target orientation design variable. (expressed as gravity vector in target frame)
        if estimateGravityLength:
            self.gravityDv = aopt.EuclideanPointDv( initialGravityEstimate )
        else:
            self.gravityDv = aopt.EuclideanDirection( initialGravityEstimate )
        self.gravityExpression = self.gravityDv.toExpression()  
        self.gravityDv.setActive( True )
        problem.addDesignVariable(self.gravityDv, HELPER_GROUP_ID)
        
        #Add all DVs for all IMUs
        for imu in self.ImuList:
            imu.addDesignVariables( problem )
        
        #Add all DVs for the camera chain    
        self.CameraChain.addDesignVariables( problem, noTimeCalibration, noChainExtrinsics )

    def addPoseMotionTerms(self, problem, tv, rv):
        wt = 1.0/tv;
        wr = 1.0/rv
        W = np.diag([wt,wt,wt,wr,wr,wr])
        asp.addMotionErrorTerms(problem, self.poseDv, W, errorOrder)
        
    #add camera to sensor list (create list if necessary)
    def registerCamChain(self, sensor):
        self.CameraChain = sensor

    def registerImu(self, sensor):
        self.ImuList.append( sensor )
            
    def buildProblem( self, 
                      splineOrder=6, 
                      poseKnotsPerSecond=70, 
                      biasKnotsPerSecond=70, 
                      doPoseMotionError=False, 
                      mrTranslationVariance=1e6,
                      mrRotationVariance=1e5,
                      doBiasMotionError=True,
                      blakeZisserCam=-1,
                      huberAccel=-1,
                      huberGyro=-1,
                      noTimeCalibration=False,
                      noChainExtrinsics=True,
                      maxIterations=20,
                      gyroNoiseScale=1.0,
                      accelNoiseScale=1.0,
                      timeOffsetPadding=0.02,
                      verbose=False  ):

        print("\tSpline order: %d" % (splineOrder))
        print("\tPose knots per second: %d" % (poseKnotsPerSecond))
        print("\tDo pose motion regularization: %s" % (doPoseMotionError))
        print("\t\txddot translation variance: %f" % (mrTranslationVariance))
        print("\t\txddot rotation variance: %f" % (mrRotationVariance))
        print("\tBias knots per second: %d" % (biasKnotsPerSecond))
        print("\tDo bias motion regularization: %s" % (doBiasMotionError))
        print("\tBlake-Zisserman on reprojection errors %s" % blakeZisserCam)
        print("\tAcceleration Huber width (sigma): %f" % (huberAccel))
        print("\tGyroscope Huber width (sigma): %f" % (huberGyro))
        print("\tDo time calibration: %s" % (not noTimeCalibration))
        print("\tMax iterations: %d" % (maxIterations))
        print("\tTime offset padding: %f" % (timeOffsetPadding))


        ############################################
        ## initialize camera chain
        ############################################
        #estimate the timeshift for all cameras to the main imu
        self.noTimeCalibration = noTimeCalibration
        if not noTimeCalibration:
            for cam in self.CameraChain.camList:
                cam.findTimeshiftCameraImuPrior(self.ImuList[0], verbose)
        
        #obtain orientation prior between main imu and camera chain (if no external input provided)
        #and initial estimate for the direction of gravity
        self.CameraChain.findOrientationPriorCameraChainToImu(self.ImuList[0])
        estimatedGravity = self.CameraChain.getEstimatedGravity()

        ############################################
        ## init optimization problem
        ############################################
        #initialize a pose spline using the camera poses in the camera chain
        poseSpline = self.CameraChain.initializePoseSplineFromCameraChain(splineOrder, poseKnotsPerSecond, timeOffsetPadding)
        
        # Initialize bias splines for all IMUs
        for imu in self.ImuList:
            imu.initBiasSplines(poseSpline, splineOrder, biasKnotsPerSecond)
        
        # Now I can build the problem
        problem = inc.CalibrationOptimizationProblem()

        # Initialize all design variables.
        self.initDesignVariables(problem, poseSpline, noTimeCalibration, noChainExtrinsics, initialGravityEstimate = estimatedGravity)
        
        ############################################
        ## add error terms
        ############################################
        #Add calibration target reprojection error terms for all camera in chain
        self.CameraChain.addCameraChainErrorTerms(problem, self.poseDv, blakeZissermanDf=blakeZisserCam, timeOffsetPadding=timeOffsetPadding)
        
        # Initialize IMU error terms.
        for imu in self.ImuList:
            imu.addAccelerometerErrorTerms(problem, self.poseDv, self.gravityExpression, mSigma=huberAccel, accelNoiseScale=accelNoiseScale)
            imu.addGyroscopeErrorTerms(problem, self.poseDv, mSigma=huberGyro, gyroNoiseScale=gyroNoiseScale, g_w=self.gravityExpression)

            # Add the bias motion terms.
            if doBiasMotionError:
                imu.addBiasMotionTerms(problem)
            
        # Add the pose motion terms.
        if doPoseMotionError:
            self.addPoseMotionTerms(problem, mrTranslationVariance, mrRotationVariance)
        
        # Add a gravity prior
        self.problem = problem


    def optimize(self, options=None, maxIterations=30, recoverCov=False):

        if options is None:
            options = aopt.Optimizer2Options()
            options.verbose = True
            options.doLevenbergMarquardt = True
            options.levenbergMarquardtLambdaInit = 10.0
            options.nThreads = max(1,multiprocessing.cpu_count()-1)
            options.convergenceDeltaX = 1e-5
            options.convergenceDeltaJ = 1e-2
            options.maxIterations = maxIterations
            options.trustRegionPolicy = aopt.LevenbergMarquardtTrustRegionPolicy(options.levenbergMarquardtLambdaInit)
            options.linearSolver = aopt.BlockCholeskyLinearSystemSolver() #does not have multi-threading support

        #run the optimization
        self.optimizer = aopt.Optimizer2(options)
        self.optimizer.setProblem(self.problem)

        optimizationFailed=False
        try:
            retval = self.optimizer.optimize()
            if retval.linearSolverFailure:
                optimizationFailed = True
        except Exception as e:
            sm.logError(str(e))
            optimizationFailed = True

        if optimizationFailed:
            sm.logError("Optimization failed!")
            raise RuntimeError("Optimization failed!")
        
        if recoverCov:
            self.recoverCovariance()
        #free some memory
        del self.optimizer
        gc.collect()
        
    
    def recoverCovariance(self):
        print("Recovering covariance...")
        imu_covariance_blocks = [
            (imu, [
                imu.q_i_b_Dv.blockIndex(),
                imu.r_b_Dv.blockIndex(),
            ])
            for imu in self.ImuList
            if not imu.isReferenceImu
        ]
        time_blocks = [
            self.CameraChain.camList[i].cameraTimeToImuTimeDv.blockIndex()
            for i in range(len(self.CameraChain.camList))
        ]
        brows = [
            self.CameraChain.camList[0].T_c_b_Dv.q.blockIndex(),
            self.CameraChain.camList[0].T_c_b_Dv.t.blockIndex(),
            *time_blocks,
            *[
                block_index
                for _, imu_blocks in imu_covariance_blocks
                for block_index in imu_blocks
            ],
        ]
        P = aopt.SparseBlockMatrixXd()
        try:
            self.optimizer.computePartialCovariances(brows, P, 1e-6)
        except Exception as e:
            print("Fail to recover covariance:")
            print(e)
            self.cov_trafo_ic = np.ones((6, 6), dtype=float) * np.nan
            self.cov_times = np.ones((len(self.CameraChain.camList), len(self.CameraChain.camList)), dtype=float) * np.nan
            self.std_trafo_ic = np.sqrt(self.cov_trafo_ic.diagonal())
            self.std_times = np.sqrt(self.cov_times.diagonal())
            for imu, _ in imu_covariance_blocks:
                imu.cov_T_i_b = np.ones((6, 6), dtype=float) * np.nan
            return

        #split and store the variance
        self.cov_trafo_ic = getPartialBlocks(P, brows[:2])
        self.cov_times = getPartialBlocks(P, brows[2:2 + len(time_blocks)])
        self.std_trafo_ic = np.sqrt(self.cov_trafo_ic.diagonal())
        self.std_times = np.sqrt(self.cov_times.diagonal())
        for imu, imu_blocks in imu_covariance_blocks:
            cov_q_r = getPartialBlocks(P, imu_blocks)
            R_i_b = imu.q_i_b_Dv.toRotationMatrix()
            r_b = imu.r_b_Dv.toEuclidean()

            skew = lambda v: np.cross(np.eye(3), v)
            J_T_i_b = np.block([[np.eye(3), np.zeros((3, 3))],
                                [skew(R_i_b @ r_b), -R_i_b]])
            imu.cov_T_i_b = J_T_i_b @ cov_q_r @ J_T_i_b.T
    
    def saveImuSetParametersYaml(self, resultFile):
        imuSetConfig = kc.ImuSetParameters(resultFile, True)
        for imu in self.ImuList:
            imuConfig = imu.getImuConfig()
            imuSetConfig.addImuParameters(imu_parameters=imuConfig)
        imuSetConfig.writeYaml(resultFile)

    def saveCamChainParametersYaml(self, resultFile, withCov=False):
        chain = self.CameraChain.chainConfig
        nCams = len(self.CameraChain.camList)
    
        # Calibration results
        for camNr in range(0,nCams):
            #cam-cam baselines           
            if camNr > 0:
                T_cB_cA, baseline = self.CameraChain.getResultBaseline(camNr-1, camNr)
                chain.setExtrinsicsLastCamToHere(camNr, T_cB_cA)

            #imu-cam trafos
            T_ci = self.CameraChain.getResultTrafoImuToCam(camNr)
            cov = self.cov_trafo_ic if withCov and camNr == 0 else None
            chain.setExtrinsicsImuToCam(camNr, T_ci, cov)

            if not self.noTimeCalibration:
                #imu to cam timeshift
                timeshift = float(self.CameraChain.getResultTimeShift(camNr))
                cov_timeshift = self.cov_times if withCov and camNr == 0 else None
                chain.setTimeshiftCamImu(camNr, timeshift, cov_timeshift)
             
        try:
            chain.writeYaml(resultFile)
        except:
            raise RuntimeError("ERROR: Could not write parameters to file: {0}\n".format(resultFile))
