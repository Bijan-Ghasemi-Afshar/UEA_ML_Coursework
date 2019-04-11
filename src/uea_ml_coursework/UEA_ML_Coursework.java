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

    public static void test1(String dataLocation){
        
        Instances allData = null;
        
        // Loading the data
        try{
            allData = WekaTools.loadData(dataLocation);
        } catch (Exception e){
            System.out.println("There was an issue loading the data \n" + e );
        }
        
        if (allData != null){
            
            // Print dataset information
            System.out.println("Attributes: " + allData.numAttributes() 
                    + "\nInstances: " + allData.numInstances() + 
                    "\nClasses: " + allData.numClasses());
            double[] classDist = WekaTools.classDistribution(allData);
            for (int i = 0; i < classDist.length; i++){
                System.out.println("Class index " + i + " distribution: "
                        + classDist[i]);
            }
            
            // Spliting the data
            Instances[] splitedData = WekaTools.splitData(allData, 0.3);
            Instances trainData = splitedData[0];
            Instances testData = splitedData[1];
            
            // Instantiate classifier
            KNN oneNN = new KNN();
            oneNN.setK(5);
            
            // Build the classifier using the training data
            try{

                oneNN.buildClassifier(trainData);

            } catch (Exception e){

                System.out.println("There was an issue building classifier\n" 
                        + e);

            }

            // Test classifiers
//            double accuracy = oneNN.classifyInstance(testData.get(0));
//            oneNN.distributionForInstance((testData.get(0)));
            double accuracy = WekaTools.accuracy(oneNN, testData);
            System.out.println("The One Neares Neighbor accuracy is: " 
                    + accuracy + "%");

        }
        
    }
    
    /**
     * The main function for testing the KNN classifier.
     * @param args Terminal arguments passed to the program
     */
    public static void main(String[] args) {
        
        test1("./data/Height_Sex.arff");
        
    }
    
}