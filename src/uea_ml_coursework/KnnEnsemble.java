/*
 * This class is an implementation of a KNN Ensemble. There some general design
 * decisions made that are justified based on proposed methods for sampling 
 * which are referenced through out this class implementation. The Boosting
 * sampling method has been used to build this ensemble. Inspired by the following:
 * (http://www.sciencedirect.com/science/article/pii/S0957417409002140)
 * According to the aforementioned paper, Boosting method when constructing
 * ensemble of KNNs does not improve the results significantly. Hence, a 
 * modification of this approach is more favourable.
 */
package uea_ml_coursework;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 18/04/2019
 * @author Bijan Ghasemi Afshar (100125463)
 */
public class KnnEnsemble implements Classifier{
    
    // Class properties
    private Instances dataModel;
    private double[] instanceWeights;
    private double[] classWeight;
    private KNN[] knnEnsemble;
    private int bestK;
    private int attrSelectionCycle;
    
    /**
     * Constructor for creating the KNN Ensemble.
     */
    public KnnEnsemble(){
        this.knnEnsemble = new KNN[50];
        this.classWeight = new double[50];
        bestK = 0;
        attrSelectionCycle = 1;
    }
    
    /**
     * Constructor for creating the KNN Ensemble.
     * @param numOfEnsembles Number of ensembles.
     */
    public KnnEnsemble(int numOfEnsembles){
        this.knnEnsemble = new KNN[numOfEnsembles];
        this.classWeight = new double[numOfEnsembles];
        bestK = 0;
        attrSelectionCycle = 2;
    }
    
    /**
     * Accessor for best K.
     * @param bestK The value of the best K.
     */
    public void setBestK(int bestK){
        this.bestK = bestK;
    }
    
    /**
     * Builds the ensemble classifier and sets each member of the ensemble.
     * @param data The training dataset.
     */
    public void buildClassifier(Instances data){
        
        // Initializing variables
        dataModel = new Instances(data);
        Instances clonedDataModel = new Instances(dataModel);
        this.instanceWeights = new double[dataModel.numInstances()];
        double weightedError = 0.0;
        
        // Initialize all instance weights as 1
        resetInstanceWeight();
        
        // Populate the ensemble
        try {
        
            for (int i = 0; i < knnEnsemble.length; i++){
                
                // if first run, get the original dataset
                if (i == 0){
                    
                    // if best K value is not set find it automatically
                    if (this.bestK == 0){
                        knnEnsemble[i] = new KNN(true, true, true);
                    } else {
                        knnEnsemble[i] = new KNN(true, false, true);
                    }
                    
                    knnEnsemble[i].buildClassifier(clonedDataModel);
                    bestK = knnEnsemble[i].getK();
                    
                } 
                // If in the first half run, resample attributes
                else if (i < (knnEnsemble.length/2)){
                    
                    knnEnsemble[i] = new KNN(true, false, true);
                    knnEnsemble[i].setK(bestK);
                    clonedDataModel = resampleAttribute();
                    knnEnsemble[i].buildClassifier(clonedDataModel);
                    
                } 
                // If in the second half run, resample instances
                else {
                    
                    // If second half run starts resample from original data
                    // and move on
                    if(i == (knnEnsemble.length/2)){
                        clonedDataModel = new Instances(dataModel);
                    }
                    
                    knnEnsemble[i] = new KNN(true, false, true);
                    knnEnsemble[i].setK(bestK);
                    knnEnsemble[i].buildClassifier(clonedDataModel);
                }
                // Get the wrong classifications of this classifier through
                // 10-fold cross validation
                int[] wrongClassifications = knnEnsemble[i].crossValidateTest();
                
                // Calculate the weighted error
                weightedError = calculateWeightedError(wrongClassifications);
                
                // if the weighted error is zero give a high weight to the 
                // classifier and reset the instance weights
                if (weightedError == 0.0){
                
                    /** Give high weight to this classifier, reset instance 
                    * weights and use the original data again.
                    * This hopes to minimize the classifier fluctuation of KNN
                    * due to random selection if there are ties in each run.
                    */
                    classWeight[i] = 2;
                    resetInstanceWeight();
                    clonedDataModel = new Instances(dataModel);
                    
                } else {
                    
                    // Calculating the class weight
                    classWeight[i] = Math.log((1-weightedError)/weightedError) 
                            * (0.5);
                    
                    // If in second half run, recalculate instance weight and
                    // resample instances based on weight
                    if (i > (knnEnsemble.length/2)){
                        
                        // Re-calculating the instance weights
                        instanceWeights = calculateInstanceWeight(wrongClassifications, 
                            classWeight[i]);
                        
                        // Resample data with more focus on instances that where
                        // wrongly classified
                        clonedDataModel = clonedDataModel.resampleWithWeights(
                            new Random(100), instanceWeights);
                    }
                }
            }
            
        } catch (Exception e){
            System.out.println("There was an issue creating ensemble\n" + e);
        }
        
    }
    
    /**
     * Classifies an instances.
     * @param object The object that is to be classified.
     * @return The class index of the result.
     */
    public double classifyInstance(Instance object){
        int classIndex = 0;
        double[] classifyResults = new double[dataModel.numClasses()];
        double highestVote = 0;
        
//        System.out.println("\n");
        for (int i = 0; i < knnEnsemble.length; i++){
            classIndex = (int)knnEnsemble[i].classifyInstance(object);
//            System.out.println(classIndex);
            classifyResults[classIndex] += classWeight[i];
        }
//        System.out.println("\n");
        
        // Find the class with highest vote
        for (int i = 0; i < classifyResults.length; i++){
            
//            System.out.println(i + " classify Vote: " + classifyResults[i]);
            
            if (classifyResults[i] == highestVote){
                if (Math.random() < 0.5){
                    highestVote = classifyResults[i];
                    classIndex = i;
                }
            } else if(classifyResults[i] > highestVote){
                highestVote = classifyResults[i];
                classIndex = i;
            } else {}
        }
        
        return (double)classIndex;
    }
    
    /**
     * Calculates the distribution to which each class was voted for (0.0 - 1.0)
     * @param object The object that is to be classified.
     * @return An array of distributions for each class.
     */
    public double[] distributionForInstance(Instance object){
        double[] results = new double[dataModel.numClasses()];
        
        classifyInstance(object);
        
        return results;
    }
 
    /**
     * Function to reset the instance weights if the classifier got all 
     * instances right.
     */
    private void resetInstanceWeight(){
        
        // Initialize all instance weights as 1
        for (int i = 0; i < instanceWeights.length; i++){
            instanceWeights[i] = 1;
        }
        instanceWeights = normalizeInstanceWeight(instanceWeights);
    }
    
    /**
     * Function for normalizing instance weights.
     * @param weights The initial weights that need to be normalized
     * @return The normalized instance weights.
     */
    private double[] normalizeInstanceWeight(double[] weights){
        
        double sumOfWeights = 0.0;
        
        for (double value:weights){
            sumOfWeights += value;
        }
        
//        System.out.println("Sum of instance Weights: " + sumOfWeights);
        
        for (int i = 0; i < dataModel.numInstances(); i++){
            
            weights[i] = weights[i]/sumOfWeights;
//            System.out.println(i + " weight: " + weights[i]);
        }
        
        return weights;
    }
 
    /**
     * Function for calculating instance weights. This is used for giving more
     * weight to wrongly classified instances for creating the next classifier.
     * @param wrongClassifications index of wrongly classified instances.
     * @param classW The classifier weight of the current classifier.
     * @return The new instance weights.
     */
    private double[] calculateInstanceWeight(int[] wrongClassifications, 
            double classW){
        
        for (int i = 0; i < instanceWeights.length; i++){
            
            // If instance was wrongly classified give higher weight
            for (int j = 0; j < wrongClassifications.length; j++){
                if (i == wrongClassifications[j]){
                    instanceWeights[i] = instanceWeights[i] * Math.exp(classW);
                }
            }
            
            // if instance was correctly classified give less weight
            instanceWeights[i] = instanceWeights[i] * Math.exp(-classW);
        }
        
        // normalize weights
        instanceWeights = normalizeInstanceWeight(instanceWeights);
        
        return instanceWeights;
    }
    
    /**
     * Function for calculating the weighted error of a classifier. Calculated 
     * by adding the instance weight of wrongly classified instances.
     * @param index The index of wrongly classified instances.
     * @return The calculated weighted error.
     */
    private double calculateWeightedError(int[] index){
        double sum = 0;
        
        for (int i = 0; i < index.length; i++){
            sum += instanceWeights[index[i]];
        }
        
        return sum;
    }
    
    private Instances resampleAttribute(){
        
        Instances resampledData = new Instances(dataModel);
        Random rand = new Random();
        int attrIndexToRemove = 0;
        // Class attribute and total number of attribute need to be ignored
        // since they don't contribute to attribute combination selection
        int numOfAttr = dataModel.numAttributes() - 2;
        int numOfSelectedAttr = this.attrSelectionCycle % numOfAttr;
        
        // if cycle of attribute sampling matches number of attributes - 1
        // select random (attributes - 1) attributes
        if(numOfSelectedAttr == 0){
            numOfSelectedAttr = numOfAttr;
        }
        
        // Randomly remove attributes 
        for (int i = 0; i < numOfSelectedAttr; i++){
            attrIndexToRemove = rand.nextInt((numOfAttr + 1));
            resampledData.deleteAttributeAt(attrIndexToRemove);
            numOfAttr--;
        }
        this.attrSelectionCycle++;
        
        return resampledData;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
