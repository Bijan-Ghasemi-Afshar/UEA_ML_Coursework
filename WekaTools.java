/*
 * These are some useful methods to simplify working with Weka
 */
package weka.tools;

import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * 18/02/2019
 * @author Bijan
 */
public class WekaTools {
    
    /**
     * Function to calculate the accuracy of a classifier.
     * @param classifier The classifier to be tested.
     * @param test The test dataset for calculating the accuracy.
     * @return The accuracy of the classification results on the test dataset.
     */
    public static double accuracy(Classifier classifier, Instances test){
        
        int numOfCorrect = 0;
        for (int i = 0; i < test.numInstances(); i++){
            try{
                
                double classificationResult = classifier.classifyInstance(test.instance(i));
                double instanceClassValue = test.instance(i).classValue();
                
                if (classificationResult == instanceClassValue){

                    numOfCorrect++;
                }
                
            } catch (Exception e){
                System.out.println("There was an issue clasifying\n" + e);
            }
            
        }
        
        int total = test.numInstances();
        
//        System.out.println("Corrects: " + numOfCorrect);
//        System.out.println("Total: " + total);
        
        return ((double)numOfCorrect / (double)total);
        
    }
    
    /**
     * Function that calculates accuracy based on given classification results
     * and actual results.
     * @param actualResults The expected results.
     * @param classifiedInstances The classification results.
     * @return The accuracy.
     */
    public static double getAccuracy(int[] actualResults, 
            int[] classifiedInstances){
        
        double accuracy = 0.0;
        
        for (int i = 0; i < actualResults.length; i++){
            if (actualResults[i] == classifiedInstances[i]){
                accuracy++;
            }
        }
        
        System.out.println("Corrects: " + accuracy + "\nTotal: " 
                + actualResults.length);
        
        return (accuracy/actualResults.length) * 100;
    }
    
    /**
     * Function to load the dataset.
     * @param dataLocation The dataset to be loaded.
     * @param shuffle Flag to whether shuffle the order of instances.
     * @return The dataset (Instances) object
     */
    public static Instances loadData(String dataLocation, boolean shuffle){

        try{
            FileReader reader = new FileReader(dataLocation);
            Instances data = new Instances(reader);
            // Set the last arrtibute as the class value
            data.setClassIndex(data.numAttributes() - 1);
            if (shuffle)
                data.randomize(data.getRandomNumberGenerator((long)(Math.random() * 1000)));
            return data;
        }catch(Exception e){
            System.out.println("There was an issue loading the data\n" + e);
            return null;
        }
        
    }
    
    /**
     * Function to print the general information of the dataset.
     * @param dataset The dataset provided for displaying its information.
     */
    public static void printDatasetInfo(Instances dataset){
        
        System.out.println("Attributes: " + (dataset.numAttributes() - 1)
                + "\nInstances: " + dataset.numInstances() + 
                "\nClasses: " + dataset.numClasses());
        double[] classDist = WekaTools.classDistribution(dataset);
        for (int i = 0; i < classDist.length; i++){
            System.out.println("Class index " + i + " distribution: "
                    + classDist[i]);
        }
        System.out.println("");
        
    }
    
    /**
     * Function to split data into two parts.
     * @param all The original data to be split.
     * @param proportion The proportion of the data that needs to be split.
     * @return The array with the length of two containing split data.
     */
    public static Instances[] splitData(Instances all, double proportion){
        
        Instances[] split = new Instances[2];
        
        // Copy all data to split[0]
        split[0] = new Instances(all);
        
        // Copy header data but no instances to split[1]
        split[1] = new Instances(all, 0);
        
        int dataProportion = (int)(proportion * all.numInstances());
        
        for (int i = dataProportion-1; i >= 0; i--){
            
            split[1].add(split[0].instance(i));
            split[0].delete(i);
            
        }
        
        return split;
    }
    
    /**
     * Function for getting the distribution of classes in a dataset.
     * @param data The dataset for finding its class distribution.
     * @return The array containing class distributions for classes in indexes.
     */
    public static double[] classDistribution(Instances data){
        
        double[] results = new double[data.numClasses()];
        int total = data.numInstances();
        
        for (int i = 0; i < data.numInstances(); i++){
            
            results[(int)data.instance(i).classValue()]++;
            
        }
        
        for (int i = 0; i < results.length; i++){
            
            results[i] = results[i] / total;
            
        }
        
        return results;
        
    }
    
    /**
     * Function for creating a confusion matrix.
     * @param predicted The array containing predicted values of a classification.
     * @param actual The array containing actual classes of a dataset.
     * @param numberOfClasses The number of classes used for classification.
     * @return The confusion matrix.
     */
    public static int[][] confusionMatrix(int[] predicted, int[] actual, int numberOfClasses){
        
        
        int[][] results = new int[numberOfClasses][numberOfClasses];
        
        for (int i = 0; i < numberOfClasses; i++){
            
            for (int j = 0; j < numberOfClasses; j++){
                
                if (i == j){
                    results[i][j] = countNumberOfCorrectClassResults(predicted, actual, j);
                } else {
                    results[i][j] = countNumberOfWrongClassResults(predicted, actual, i, j);
                }
                
            }
        }
        
        return results;
    }
    
    /**
     * Function for printing the confusion matrix.
     * @param confusionMtx The two dimensional array of confusion matrix.
     */
    public static void printConfusionMatrix(int[][]confusionMtx){
        
        System.out.println("------The Confusion Matrix------");
        System.out.print("  ");
        for (int i = 0; i < confusionMtx.length; i++){
            
            System.out.print("\t" + i);
            
        }
        System.out.println("");
        
        for (int i = 0; i < confusionMtx.length; i++){
            
            System.out.print(i + "\t");
            for (int j = 0; j < confusionMtx.length; j++){
                
                System.out.print(confusionMtx[j][i] + "\t");
                
            }
            System.out.println("");
        }
        
    }
    
    /**
     * Function that counts the number of correct classifications.
     * @param predicted The array containing class prediction of classification.
     * @param actual The array containing actual class values of a classification.
     * @param classIndex The index of the correct class index.
     * @return Number of correct classifications.
     */
    private static int countNumberOfCorrectClassResults(int[] predicted, int[] actual, int classIndex){
        
        int result = 0;
        
        for (int i = 0; i < predicted.length; i++){
            
            if (predicted[i] == classIndex && actual[i] == predicted[i]){
                
                result++;
                
            }

        }
        
        return result;
        
    }
    
    /**
     * Function that counts the number of wrong classifications.
     * @param predicted The array containing class prediction of classification.
     * @param actual The array containing actual class values of a classification.
     * @param classIndexColumn The index of the column of the class in a confusion matrix.
     * @param classIndexRow The index of the row of the class in a confusion matrix.
     * @return Number of correct classifications.
     */
    private static int countNumberOfWrongClassResults(int[] predicted, int[] actual, int classIndexColumn, int classIndexRow){
        
        int result = 0;
        
        for (int i = 0; i < predicted.length; i++){
            
            if (actual[i] == classIndexColumn && classIndexRow == predicted[i]){
                
                result++;
                
            }

        }
        
        return result;
        
    }

    /**
     * Function that classifies a test dataset.
     * @param classifier The classifier for testing.
     * @param testData The dataset to be used on the classifier.
     * @return The predictions of the classifier.
     */
    public static int[] classifyInstances(Classifier classifier, Instances testData){
        
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
     * Function to find the actual class values of a dataset.
     * @param testInstances The test dataset.
     * @return The an array containing actual class values of a dataset.
     */
    public static int[] getClassValues(Instances testInstances){
        
        int[] results = new int[testInstances.numInstances()];
        
        for (int i = 0; i < testInstances.numInstances(); i++){
            
            results[i] = (int)testInstances.instance(i).classValue();
            
        }
        
        return results;
        
    }
    
    /**
     * Main function for testing functions of this tool set.
     * @param args Arguments passes from terminal.
     */
    public static void main(String[] args) {
        
        // Testing the confusion matrix
        int[] actual = new int[]{0,0,1,1,1,0,0,1,1,1};
        int[] predicted = new int[]{0,1,1,1,1,1,1,1,1,1};
        int[][] confMatrix = confusionMatrix(predicted, actual, 2);
        printConfusionMatrix(confMatrix);
        
        actual = new int[]{0,1,2,0,1,2};
        predicted = new int[]{0,1,2,1,2,0};
        confMatrix = confusionMatrix(predicted, actual, 3);
        printConfusionMatrix(confMatrix);
        
    }
    
}
