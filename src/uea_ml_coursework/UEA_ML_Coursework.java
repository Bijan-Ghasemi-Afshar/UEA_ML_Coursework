/**
 * This class is used for generating the test cases and execuring them on the 
 * KNN classifier.
 */
package uea_ml_coursework;

import weka.core.Instances;
import weka.tools.WekaTools;

/**
 * 10/04/2019
 * @author Bijan Ghasemi Afshar (100125463)
 */
public class UEA_ML_Coursework {
    
    /**
     * Tests the results of Part 1
     * @param dataLocation The location of the file containing Train File
     */
    public static void testPart1(String dataLocation){
        
        Instances trainData = null, testData = null;
        
        // Loading the data
        try{
            trainData = WekaTools.loadData(dataLocation, false);
        } catch (Exception e){
            System.out.println("There was an issue loading the data \n" + e );
        }
        
        if (trainData != null){
            
            // Print dataset information
            WekaTools.printDatasetInfo(trainData);
            
            // Instantiate classifier
            KNN knn = new KNN();
            knn.setK(3);
            
            // Build the classifier using the training data
            try{
                knn.buildClassifier(trainData);
            } catch (Exception e){
                System.out.println("There was an issue building classifier\n"
                        + e);
            }
            
            // Classifying the unclassified objects from Part 1
            try {
                testData = WekaTools.loadData("./data/Pitcher_Plants_TEST.arff", 
                        false);
            } catch (Exception e){
                System.out.println("Error loading test data\n" + e);
            }
            
            // Classify test data and show distribution for each class
            // Clone test data for getting the distribution 
            // to avoid re-standardising
            /** Expected Results
             * 5, 13, N.raja     | 1.0  | Standardised
             * 6, 14, N.truncata | 0.66 | Standardised
             * 7, 15, N.truncata | 0.66 | Standardised
             * 8, 12, N.truncata | 0.66 | Standardised
             */
            System.out.println("------Classification Results------");
            Instances clonedTest = new Instances (testData);
            double[] instDist = new double[trainData.numClasses()];
            for (int i = 0; i < testData.numInstances(); i++){
                System.out.println("\n" + testData.get(i));
                System.out.println("Result: " + 
                        trainData.classAttribute().value((int)knn.classifyInstance(testData.get(i))));
                System.out.println(testData.get(i));
                instDist = knn.distributionForInstance(clonedTest.get(i));
                for (int j = 0; j < instDist.length; j++){
                    System.out.println("Class " + (j+1) + ": " + instDist[j]);
                }
            }

        }
        
    }
    
    /**
     * Tests the standardisation functionality.
     * @param dataLocation The location of the file containing Train File
     */
    public static void testStandardisation(String dataLocation){
     
        Instances trainData = null, testData = null;
        
        // Loading the data
        try{
            trainData = WekaTools.loadData(dataLocation, false);
        } catch (Exception e){
            System.out.println("There was an issue loading the data \n" + e );
        }
        
        if (trainData != null){
            
            // Print dataset information
            WekaTools.printDatasetInfo(trainData);
            
            // Instantiate classifier
            KNN knn = new KNN();
            
            // Build the classifier using the training data
            try{
                knn.buildClassifier(trainData);
            } catch (Exception e){
                System.out.println("There was an issue building classifier\n"
                        + e);
            }
            
            // Loading the unclassified objects from Part 1
            try {
                testData = WekaTools.loadData("./data/Pitcher_Plants_TEST.arff", 
                        false);
            } catch (Exception e){
                System.out.println("Error loading test data\n" + e);
            }
            
            // Standardising the values
            /** Expected Results. (+- 0.01)
             * 1.51, -0.23
	     * -1.07, 0.46
             * 0.22, 0.46
             * -0.42, -1.62
             * -0.42, -0.93
             * -1.07, -0.93
             * 0.87, 1.16
             * -0.42, 1.16
             * 0.87, -0.23
             * 0.22, -0.93
	     * -1.72, -0.23
	     * 1.51, 1.86
             */
            System.out.println("------Standardisation Results------");
            System.out.println(trainData);

        }
        
    }
    
    /**
     * Tests the functionality that sets K through LOOCV.
     * @param dataLocation The location of the file that contains training data.
     */
    public static void testSettingKByLOOCV(String dataLocation){
        
        Instances trainData = null, testData = null;
        
        // Loading the data
        try{
            trainData = WekaTools.loadData(dataLocation, false);
        } catch (Exception e){
            System.out.println("There was an issue loading the data \n" + e );
        }
        
        if (trainData != null){
            
            // Print dataset information
            WekaTools.printDatasetInfo(trainData);
            
            // Instantiate classifier
            KNN knn = new KNN();
            knn.setK(3);
            System.out.println("K: " + knn.getK());
            
            // Build the classifier using the training data
            try{
                knn.buildClassifier(trainData);
            } catch (Exception e){
                System.out.println("There was an issue building classifier\n"
                        + e);
            }
            
            // Classifying the unclassified objects from Part 1
            try {
                testData = WekaTools.loadData("./data/Pitcher_Plants_TEST.arff", 
                        false);
            } catch (Exception e){
                System.out.println("Error loading test data\n" + e);
            }
            
            // Classify test data and show distribution for each class
            // Clone test data for getting the distribution 
            // to avoid re-standardising
            /** Expected Results
             * 5, 13, N.raja     | 1.0  | Standardised
             * 6, 14, N.truncata | 0.66 | Standardised
             * 7, 15, N.truncata | 0.66 | Standardised
             * 8, 12, N.truncata | 0.66 | Standardised
             */
            System.out.println("------Classification Results------");
            Instances clonedTest = new Instances (testData);
            double[] instDist = new double[trainData.numClasses()];
            for (int i = 0; i < testData.numInstances(); i++){
                System.out.println("\n" + testData.get(i));
                System.out.println("Result: " + 
                        trainData.classAttribute().value((int)knn.classifyInstance(testData.get(i))));
                System.out.println(testData.get(i));
                instDist = knn.distributionForInstance(clonedTest.get(i));
                for (int j = 0; j < instDist.length; j++){
                    System.out.println("Class " + (j+1) + ": " + instDist[j]);
                }
            }

        }
        
    }
    
    /**
     * The main function for testing the KNN classifier.
     * @param args Terminal arguments passed to the program
     */
    public static void main(String[] args) {
        
//        testPart1("./data/Pitcher_Plants_TRAIN.arff");
        
//        testStandardisation("./data/Pitcher_Plants_TRAIN.arff");

        testSettingKByLOOCV("./data/Pitcher_Plants_TRAIN.arff");

    }
    
}
