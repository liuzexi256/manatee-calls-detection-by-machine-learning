# manatee-calls-detection-by-machine-learning  
The purpose of is to detect manatee calls from real hydrophone recordings taken in an estuary. The data used for this project was collected by the Department of Biology at UF. Given data set are (1) a file (train_signal.wav) with 10 different manatee calls segmented by the biologist that represent the signal class we would like to detect; (2) a 2 second noise background file (noise_signal.wav) that represents the acoustic noise picked up by the hydrophone; (3) the continuous file (test_signal.wav) with unsegmented manatee calls mixed with background noise that lasts approximately 30 seconds. The sampling rate is 48 KHz. The purpose of this project is to design and evaluate a machine learning detection approach to distinguish the manatee calls from the background.

There are many alternative ways to solve this problem, and here we will compare two procedures: A- Create two adaptive models (linear or nonlinear), one to model the manatee calls and the other to model the noisy background. Each of these models will be trained as a predictor in the files that contains the 10 examples of the manatee calls, and the noisy background respectively. Once the two models are developed, apply both of them in parallel to predict the test data set. Since they were trained for different time structures, the predictor that has the smallest error should represent the corresponding class (noise or manatee). There is still a little problem because we need to smooth the prediction error in time because, as you can expect, the noise is high frequency, so a running average smoother is needed. The output created by the system will be a square wave for the full duration of the test set, with high meaning manatee and low meaning background. B- Alternatively, implement the SPRT and CUSUM tests to implement the segmentation in the two classes. The issue that one have to check is if the time series is i.i.d. or requires a model to make the statistical test based on the prediction error.

The purpose is to compare the accuracy and the computational complexity of the two methods. To select in a principled manner the free parameters of both methods, i.e. in A: the predictors, the error smoother and to evaluate the performance of this system in the test set; In B: the size of the segments, the threshold and the number of log likelihood tests in CUSUM.

Start with linear models in A and use either LMS, RLS trained with MSE or MCC in the input space or RKHS. Regarding the error smoother use either a window or a recurrent estimator. As a first important step is to use DSP tools to help understanding the data structure and appropriately set the free parameters. The big difficulty is the variability of both the background noise and the manatee calls, which calls for small model orders. Find a way to handle the data nonstationary in both classes by assemble averaging. There is not a criterion to see if the predictors are correct or not, except by hearing the data (there are 16 calls)! Since you have the calls your ear can judge the quality of the decisions. You will have two types of errors, false and missed detections. The Receiver Operating Characteristics (ROC) is the best way to compare different solutions.
