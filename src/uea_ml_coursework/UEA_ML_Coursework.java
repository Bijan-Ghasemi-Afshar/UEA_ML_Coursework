/**
 * This class is used for generating the test cases and execuring them on the 
 * KNN classifier.
 */
package uea_ml_coursework;

import weka.classifiers.Evaluation;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.tools.WekaTools;
import static weka.tools.WekaTools.confusionMatrix;
import static weka.tools.WekaTools.printConfusionMatrix;


/**
 * 10/04/2019
 * @author Bijan Ghasemi Afshar (100125463)
 */
public class UEA_ML_Coursework {
    
    /**
     * Function for getting the classification results of KNN Ensemble.
     * @param classifier The knn ensemble.
     * @param testData The test data.
     * @return The classification results.
     */
    public static int[] getEnsembleResults(KnnEnsemble classifier,
            Instances testData){
        
        int[] classificationResults = new int[testData.numInstances()];
        
        for (int i = 0; i < testData.numInstances(); i++){
            
            try {
                
                classificationResults[i] = (int)classifier.classifyInstance(testData.instance(i));
                
            } catch (Exception e) {
                
                System.out.println("There was an issue classifying\n" + e);
                
            }
            
        }
        
        return classificationResults;
        
    }
    
    /**
     * Tests the results of Part 1
     * @param dataLocation The location of the file containing Train File
     * @param testDataLocation The location of the file containing Test File
     */
    public static void testPart1(String dataLocation, String testDataLocation){
        
        Instances trainData = null, testData = null;
        
        // Loading the data
        try{
            trainData = WekaTools.loadData(dataLocation, false);
        } catch (Exception e){
            System.out.println("There was an issue loading the data \n" + e );
        }
        
        if (trainData != null){
            
            System.out.println("------Testing The Results of Part 1------\n");
            
            // Print dataset information
            System.out.println("------Training data properties------");
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
                testData = WekaTools.loadData(testDataLocation, false);
            } catch (Exception e){
                System.out.println("Error loading test data\n" + e);
            }
            
            // Classify test data and show distribution for each class
            /** Expected Results
             * 5, 13, N.raja     | 1.0  | Standardised
             * 6, 14, N.truncata | 0.66 | Standardised
             * 7, 15, N.truncata | 0.66 | Standardised
             * 8, 12, N.truncata | 0.66 | Standardised
             */
            System.out.println("------Classification Results------");
            System.out.println("K is: " + knn.getK());
            double[] instDist = new double[trainData.numClasses()];
            for (int i = 0; i < testData.numInstances(); i++){
                System.out.println("\n" + testData.get(i));
                System.out.println("Result: " + 
                        trainData.classAttribute().value((int)knn.classifyInstance(testData.get(i))));
                instDist = knn.distributionForInstance(testData.get(i));
                for (int j = 0; j < instDist.length; j++){
                    System.out.println("Class " + (j) + ": " + instDist[j]);
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
            
            System.out.println("------Testing Standardisation------\n");
            
            // Print dataset information
            System.out.println("------Training data properties------");
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
            
            System.out.println("------Testing Automated Setting of K------\n");
            
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
            
            // Set K automatically through LOOCV
            /**
             * Expected Results.
             * Range of K: 1-2 (Kmax = 12 * 0.2 = 2)
             * Since when using K=2 uses randomness when there are ties between
             * instances, there might be times where K=2 produces higher resutls
             * Result of K: 1 or 2
             */
            System.out.println("------Automated Setting of K Result------");
            System.out.println("Setting Automatically K: " + knn.getSetKAuto());
            System.out.println("K is: " + knn.getK());
        }
    }
    
    /**
     * Tests the functionality that uses a weighted scheme for voting.
     * @param dataLocation The location of the file that contains training data.
     * @param testDataLocation The location of the file containing Test File.
     */
    public static void testWeightedScheme(String dataLocation, String testDataLocation){
        
        Instances trainData = null, testData = null;
        
        // Loading the data
        try{
            trainData = WekaTools.loadData(dataLocation, false);
        } catch (Exception e){
            System.out.println("There was an issue loading the data \n" + e );
        }
        
        if (trainData != null){
        
            System.out.println("------Testing Weighted Voting Scheme------\n");
            
            // Print dataset information
            System.out.println("------Training data properties------");
            WekaTools.printDatasetInfo(trainData);
            
            // Instantiate classifier
            KNN knn = new KNN(true, false, true);
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
                testData = WekaTools.loadData(testDataLocation, false);
            } catch (Exception e){
                System.out.println("Error loading test data\n" + e);
            }
            
            // Uses the weighted voting scheme for classification.
            /**
             * Expected Results.
             * 5, 13, N.raja     | Votes class 0: 0.0  | Votes class 0: 2.05
             * 6, 14, N.truncata | Votes class 0: 1.37  | Votes class 0: 0.7
             * 7, 15, N.truncata | Votes class 0: 1.37  | Votes class 0: 0.7
             * 8, 12, N.truncata | Votes class 0: 1.22  | Votes class 0: 0.67
             */
            System.out.println("------Results of Classification------");
            System.out.println("Use weighted voting scheme: " 
                    + knn.getWeightedScheme());
            for (int i = 0; i < testData.numInstances(); i++){
                System.out.println("\n" + testData.get(i));
                double[] weightedVotes = knn.getWeightedVotes(testData.get(i));
                if (weightedVotes != null){
                    for (int j = 0; j < weightedVotes.length; j++){
                        System.out.println("Votes for class " + j + ": " 
                                + weightedVotes[j]);
                    }
                }
            }
        }
    }
    
    /**
     * Tests the KNN classifier and KNN ensemble if flag is set.
     * @param dataset The name of the dataset to be tested.
     * * @param testEnsemble The flag for testing KNN Ensemble.
     */
    public static void testDataset(String dataset, boolean testEnsemble){
        
        Instances trainData = null, testData = null;
        KnnEnsemble knnEnsem = null;
        KNN knn = null;
        
        // Loading the data
        try{
            String trainDataLocation = "./data/" + dataset + "/" + dataset + 
                    "_TRAIN.arff";
            trainData = WekaTools.loadData(trainDataLocation, true);
        } catch (Exception e){
            System.out.println("There was an issue loading the data \n" + e );
        }
        
        if (trainData != null){
            
            System.out.println("\n------Testing The KNN------\n");
            
            // Print dataset information
            System.out.println("------Training data properties------");
            WekaTools.printDatasetInfo(trainData);
            
            // Instantiate classifiers
            knn = new KNN();
            if(testEnsemble){
                knnEnsem = new KnnEnsemble();
            }
            
            // Build the classifiers using the training data
            try{
                knn.buildClassifier(trainData);
                knn.setSetKAuto(true);
                knn.setWeightedScheme(true);
                if (testEnsemble){
                    knnEnsem.setBestK(knn.getK());
                    knnEnsem.buildClassifier(trainData);
                }
                
//                Evaluation eval = new Evaluation(trainData);
//                eval.crossValidateModel(knn, trainData, 10, new Random(100));
//                System.out.printf("Estimated Accuracy: %.2f%%\n",
//                        eval.pctCorrect());
            } catch (Exception e){
                System.out.println("There was an issue building classifier\n"
                        + e);
            }
            
            
            try {
                String testDataLocation = "./data/" + dataset + "/" + dataset + 
                    "_TEST.arff";
                testData = WekaTools.loadData(testDataLocation, true);
                System.out.println("------Testing data properties------");
                WekaTools.printDatasetInfo(testData);
            } catch (Exception e){
                System.out.println("Error loading test data\n" + e);
            }
            
            // Classify test data
            System.out.println("------Classification Results------");
            System.out.println("------Single KNN------");
            int[] actualResults = WekaTools.getClassValues(testData);
            int[] classifiedInstances = WekaTools.classifyInstances(knn,
                    testData);
              
            // Get Accuracy
            System.out.printf("KNN Accuracy: %.2f%%\n",  
                    WekaTools.getAccuracy(actualResults, classifiedInstances));
            
            // Get Confusion Matrix
            int[][] confMatrix = confusionMatrix(classifiedInstances,
                    actualResults, trainData.numClasses());
            printConfusionMatrix(confMatrix);
    
            // Test ensemble if flag is set
            if (testEnsemble){
                System.out.println("------KNN Ensemle------");
                classifiedInstances = getEnsembleResults(knnEnsem, testData);
                
                // Get Accuracy
                System.out.printf("KNN Ensemble Accuracy: %.2f%%\n",  
                    WekaTools.getAccuracy(actualResults, classifiedInstances));
                
                // Get Confusion Matrix
                confMatrix = confusionMatrix(classifiedInstances,
                    actualResults, trainData.numClasses());
                printConfusionMatrix(confMatrix);
            }


        }
    }
    
    /**
     * The main function for testing the KNN classifier.
     * @param args Terminal arguments passed to the program
     */
    public static void main(String[] args) {
        
//        testPart1("./data/Pitcher_Plants_TRAIN.arff",
//                "./data/Pitcher_Plants_TEST.arff");
        
//        testStandardisation("./data/Pitcher_Plants_TRAIN.arff");

//        testSettingKByLOOCV("./data/Pitcher_Plants_TRAIN.arff");;

//        testWeightedScheme("./data/Pitcher_Plants_TRAIN.arff", 
//                "./data/Pitcher_Plants_TEST.arff");

//        testDataset("iris", true);
        testDataset("ecoli", true);
//        testDataset("libras", true);
//        testDataset("optical", false);
//        testDataset("blood", true);
//        testDataset("bank", false);


    }
    
}
