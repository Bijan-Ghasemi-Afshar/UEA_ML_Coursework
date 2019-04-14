/*
 * This is the implementation of the K-Nearest Neighbour classifier with 
 * additional functionalities. It extends the AbstractClassifier class from
 * the Weka machine learning package. This implementation assumes all attributes
 * are real values.
 */
package uea_ml_coursework;

import java.util.Arrays;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 10/04/2019
 * @author Bijan Ghasemi Afshar (100125463)
 */
public class KNN extends AbstractClassifier{

    // Class properties
    private Instances dataModel;
    private boolean standardise;
    private boolean setKAuto;
    private boolean weightedScheme;
    private double[] means;
    private double[] standardDeviations;
    private int k;
    private double[] votes;
    private int[] votesDiscrete;
    
    /**
     * Constructor for initialising the KNN object.
     */
    public KNN(){
        this.k = 1;
        this.standardise = true;
        this.setKAuto = false;
        this.weightedScheme = false;
    }
    
    /**
     * Constructor for initialising the KNN object.
     * @param standardise Flag to whether standardise values of not.
     */
    public KNN(boolean standardise){
        this.k = 1;
        this.standardise = standardise;
        this.setKAuto = false;
        this.weightedScheme = false;
    }
    
    /**
     * Constructor for initialising the KNN object.
     * @param standardise Flag to whether standardise values of not.
     * @param setK Flag to whether set K automatically through LOOCV.
     */
    public KNN(boolean standardise, boolean setKAuto){
        this.k = 1;
        this.standardise = standardise;
        this.setKAuto = setKAuto;
        this.weightedScheme = false;
    }
    
    /**
     * Constructor for initialising the KNN object.
     * @param standardise Flag to whether standardise values of not.
     * @param setKAuto Flag to whether set K automatically through LOOCV.
     * @param weightedScheme Flag to whether use weighted voting scheme.
     */
    public KNN(boolean standardise, boolean setKAuto, boolean weightedScheme){
        this.k = 1;
        this.standardise = standardise;
        this.setKAuto = setKAuto;
        this.weightedScheme = weightedScheme;
    }
    
    /**
     * Sets the K for KNN classifier.
     * @param k The number of closest neighbours considered for classification.
     */
    public void setK(int k){
        if (dataModel != null){
            this.k = k;
            testKLimit();
        } else {
            this.k = k;
        }
    }
    
    /**
     * Builds the classifier by storing the training data.
     * @param data The classified training data
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        dataModel = data;
        votes = new double[dataModel.numClasses()];
        votesDiscrete = new int[dataModel.numClasses()];
        
        // Delete instances with no class value
        dataModel.deleteWithMissingClass();
        
        // Deleting attributes that are not supported (not numeric)
        for (int i = dataModel.numAttributes() - 2; i >= 0; i--){
            if (!getCapabilities().test(dataModel.attribute(i))){
                dataModel.deleteAttributeAt(i);
            }
        }

        // Set K to highest value if K is larger than number of data model
        testKLimit();
     
        // Standardise attributes if flag is set
        if (standardise){   
            standardiseDataModelAttr();
        }
        // Set K through LOOCV
        if (setKAuto){
            setKWithLOOCV();
        } 
        System.out.println("K is " + k);
    }
    
    /**
     * This function returns the capabilities of this implementation of this 
     * classifier
     * @return The Capabilities object with allowed capabilities.
     */
    @Override
    public Capabilities getCapabilities(){
        
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();

        // Enabling attributes supported by this classifier
        capabilities.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // Enabling class attributes supported by this classifier
        capabilities.enable(Capabilities.Capability.NOMINAL_CLASS);
        capabilities.enable(Capabilities.Capability.NUMERIC_CLASS);

        // Minimum requried instances
        capabilities.setMinimumNumberInstances(1);

        return capabilities;
        
    }
    
    /**
     * Classifies an instances.
     * @param object The object that is to be classified.
     * @return The class index of the result.
     */
    @Override
    public double classifyInstance(Instance object){
        
        double closestMatch, eDistance = 0.0;
        Instance[] closestInstances = new Instance[k];
        Instances clonedData = new Instances(dataModel);
        int closestInstanceIndex = 0, classIndex = 0;
        double numOfVotes = 0.0;
        resetVotes();   // Reset votes from previous classification

        // Standardise Object
        if (this.standardise){   
            standardiseObject(object);
        }
        
        // Go through all training data K times and choose smallest distance
        for (int i = 0; i < closestInstances.length; i++){
            
            closestMatch = Double.MAX_VALUE;
            
            for (int j = 0; j < clonedData.numInstances(); j++){
                
                eDistance = distance(clonedData.instance(j), object);
                // If distances are same choose randomly
                if (eDistance == closestMatch){
                    if (Math.random() < 0.5){
                        closestInstances[i] = clonedData.instance(j);
                        closestInstanceIndex = j;
                    }
                } else if (eDistance < closestMatch){
                    closestMatch = eDistance;
                    closestInstances[i] = clonedData.instance(j);
                    closestInstanceIndex = j;
                }

            }
            clonedData.delete(closestInstanceIndex);
            classIndex = (int)(closestInstances[i].classValue());
            if(weightedScheme){
                votes[classIndex] += 1 / (1 + closestMatch);
                votesDiscrete[classIndex]++;
//                System.out.println("Weighted: " + 1 / (1 + closestMatch));
            } else {
                votes[classIndex]++;
                votesDiscrete[classIndex]++;
            }
        }
        
        // Count the votes
        classIndex = 0;
        for (int i = 0; i < votes.length; i++){
//            System.out.println("Class " + (i+1) + ": " + votes[i] + "\n");
            // If distances are same choose randomly
            if (numOfVotes == votes[i]){
                if (Math.random() < 0.5){
                    numOfVotes = votes[i];
                    classIndex = i;
                }
            } else if (numOfVotes < votes[i]){
                numOfVotes = votes[i];
                classIndex = i;
            }
        }
        
        return (double)classIndex;
    }
    
    /**
     * Calculates the distribution to which each class was voted for (0.0 - 1)
     * @param object The object that is to be classified.
     * @return An array of distributions for each class.
     */
    @Override
    public double[] distributionForInstance(Instance object){
        
        double[] results = new double[dataModel.numClasses()];
        
        classifyInstance(object);
        
        for (int i = 0; i < results.length; i++){
            
//            System.out.println(votes[i] + " " + k);
            results[i] = (double)votesDiscrete[i]/(double)k;
//            System.out.println("Vote " + i + ": " + results[i]);
        }
        
        return results;
    }
    
    /**
     * This function calculates the Euclidean distance between the attributes of
     * a classified and an unclassified object and returns that value.
     * @param data The classified training instance.
     * @param object The unclassified instance.
     * @return The Euclidean distance in a double format.
     */
    private double distance(Instance data, Instance object){
        
        int numOfAttr = data.numAttributes() - 1;
        double difference = 0.0, diffSqr = 0.0, total = 0.0;
        
        for (int i = 0; i < numOfAttr; i++){
            
//            System.out.println("Classified: " + data.value(i) + "\nObject: "
//             + object.value(i));
            difference = object.value(i) - data.value(i);
            diffSqr = Math.pow(difference, 2);
            total += diffSqr;
            
        }
//        System.out.println("Distance: " + total + "\n");
        
        return total;
        
    }
    
    /**
     * Resets votes from previous classifications to avoid conflicts.
     */
    private void resetVotes(){
        
        for (int i = 0; i < votes.length; i++){
            votes[i] = 0.0;
        }
        
    }
    
    /**
     * Checks whether K is larger than the number of data model instances
     * and if it is, K is set to the highest value possible which is the 
     * number of data model instances.
     */
    private void testKLimit(){
        if (this.k > dataModel.numInstances()){
            System.out.println("K was: " + this.k);
            this.k = dataModel.numInstances();
        }
//        System.out.println("K is: " + this.k);
    }
    
    /**
     * This functions standardises the attributes of the data model which makes
     * the mean of data to be 0 and standard deviation to 1.
     */
    private void standardiseDataModelAttr(){
        
        // Ignoring the class attribute
        int numberOfAttributes = dataModel.numAttributes() - 1;
        calculateDataModelMean();
        calculateDataModelSD(this.means);
        double standardisedAttr;
        
        for (int i = 0; i < dataModel.numInstances(); i++){
//            System.out.println(dataModel.get(i));
            for (int j = 0; j < numberOfAttributes; j++){
                standardisedAttr = (dataModel.get(i).value(j) - this.means[j])
                        / this.standardDeviations[j];
                dataModel.get(i).setValue(j, standardisedAttr);
            }
//            System.out.println(dataModel.get(i));
        }
        
    }
    
    /**
     * Standardises the object that has been passed to be classified.
     * @param object Object to be classified.
     */
    private void standardiseObject(Instance object){
        
        // Ignoring the class attribute
        int numberOfAttributes = dataModel.numAttributes() - 1;
        double standardisedAttr;
        
//        System.out.println(object);
        for (int j = 0; j < numberOfAttributes; j++){
            standardisedAttr = (object.value(j) - this.means[j])
                    / this.standardDeviations[j];
            object.setValue(j, standardisedAttr);
        }
//        System.out.println(object);
        
    }
    
    /**
     * Calculates the mean of attributes for the data model.
     */
    private void calculateDataModelMean(){
        
        // Ignoring the class attribute
        int numberOfAttributes = dataModel.numAttributes() - 1;
        this.means = new double[numberOfAttributes];
        
        for (int i = 0; i < dataModel.numInstances(); i++){
            for (int j = 0; j < numberOfAttributes; j++){    
                this.means[j] += dataModel.get(i).value(j);        
            }
        }
        for (int j = 0; j < numberOfAttributes; j++){    
            this.means[j] /= dataModel.numInstances();
        }
        
    }
    
    /**
     * Calculates the standard deviation of attributes for the data model.
     * @param means An array of means for each attribute.
     */
    private void calculateDataModelSD(double[] means){
        
        // Ignoring the class attribute
        int numberOfAttributes = dataModel.numAttributes() - 1;
        double distanceFromMean = 0;
        this.standardDeviations = new double[numberOfAttributes];
        
        for (int i = 0; i < dataModel.numInstances(); i++){
            for (int j = 0; j < numberOfAttributes; j++){
                distanceFromMean = dataModel.get(i).value(j) - means[j];
                this.standardDeviations[j] += Math.pow(distanceFromMean, 2);        
            }
        }
        for (int j = 0; j < numberOfAttributes; j++){    
            this.standardDeviations[j] /= dataModel.numInstances();
            this.standardDeviations[j] = Math.sqrt(this.standardDeviations[j]);
        }
        
    }
 
    /**
     * Set K automatically through Leave-One Out Cross Validation. This is done
     * by storing the original dataset in a temporary variable and manipulating
     * the current instance of the classifier, and finally setting back the 
     * current instance to the original dataset.
     */
    private void setKWithLOOCV(){
        
        Instance test;
        Instances train, originalDataModel = new Instances(dataModel);
        int accuracy = 0, highestAccuracy = 0, highestKIndex = 0;
        // The size of training data is always 1 fewer than original
        int[] kRange = setKRange(dataModel.numInstances() - 1);
        int[] kAccuracies = new int[kRange.length];
        // For each instance set it to test and the rest to train
        for (int i = 0; i<originalDataModel.numInstances(); i++){
            dataModel = new Instances(originalDataModel);
            test = dataModel.get(i);
            dataModel.delete(i);
            // Test the accuracy for every value of K (1-Kmax)
            for (int j = 0; j < kRange.length; j++){
                this.k = kRange[j];
                if (test.classValue() == classifyInstance(test)){
                    kAccuracies[j]++;
                }
            }
        }
        
        // Find the highest accuracy of K and settle ties randomly
        for (int i = 0; i < kAccuracies.length; i++){
            if (kAccuracies[i] == highestAccuracy){
                if (Math.random() < 0.5){
                    highestAccuracy = kAccuracies[i];
                    highestKIndex = i;
                }
            } else if (kAccuracies[i] > highestAccuracy){
                highestAccuracy = kAccuracies[i];
                highestKIndex = i;
            }
        }
        
        // Set data model back to the original dataset
        dataModel = new Instances(originalDataModel);
        
        // Set K to the highest accuracy
        this.k = (highestKIndex + 1);

    }
    
    /**
     * Sets the range of Ks when automatically setting K through LOOCV.
     * @param trainSize The size of the training data.
     * @return An arrays of Ks from 1 to Max K
     */
    private int[] setKRange(int trainSize){
        
        int[] kRange;
        // Calculate 20% of tarin data (increase 1 if it's even)
        int train20Perc = (int)(trainSize * 0.2);
        train20Perc = (train20Perc % 2 == 0) ? train20Perc : train20Perc++;
        
        if (train20Perc > 100){
            kRange = new int[100];
        } else {
            kRange = new int[train20Perc];
        }
        
        for (int i = 0; i < kRange.length; i++){
            kRange[i] = i + 1;
        }
        
        return kRange;
        
    }
    
}
