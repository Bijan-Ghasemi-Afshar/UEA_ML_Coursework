/*
 * This class is an implementation of a KNN Ensemble. There some general design
 * decisions made that are justified based on proposed methods for sampling 
 * which are referenced through out this class implementation. The Boosting
 * sampling method has been used to build this ensemble. Based on the following:
 * (http://www.sciencedirect.com/science/article/pii/S0957417409002140)
 * According to the aforementioned paper, Boosting method when constructing
 * ensemble of KNNs does not improve the results significantly. Hence, a 
 * modification of this approach is more favourable.
 */
package uea_ml_coursework;

import weka.core.Instance;
import weka.core.Instances;

/**
 * 18/04/2019
 * @author Bijan Ghasemi Afshar (100125463)
 */
public class KnnEnsemble {
    
    // Class properties
    private Instances dataModel;
    private double[] instanceWeights;
    
    private KNN[] knnEnsemble;
    
    /**
     * Constructor for creating the KNN Ensemble.
     */
    public KnnEnsemble(){
        this.knnEnsemble = new KNN[50];
    }
    
    /**
     * Builds the ensemble classifier and sets each member of the ensemble.
     * @param data The training dataset.
     */
    public void buildClassifier(Instances data){
        
        // Initializing variables
        dataModel = new Instances(data);
        this.instanceWeights = new double[dataModel.numInstances()];
        double weightedError;
        
        
        // Initialize all instance weights as 1
        for (int i = 0; i < instanceWeights.length; i++){
            instanceWeights[i] = 1;
        }
        instanceWeights = calculateInstanceWeight(instanceWeights);
        
        // Populate the ensemble
        try {
        
            for (int i = 0; i < knnEnsemble.length; i++){
                knnEnsemble[i] = new KNN(true, true, true);
                knnEnsemble[i].buildClassifier(dataModel);
                Instances wrongClassifications = knnEnsemble[i].crossValidateTest();
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
        double[] classifyResults = new double[50];
        
        for (int i = 0; i < knnEnsemble.length; i++){
            classifyResults[i] = knnEnsemble[i].classifyInstance(object);
            classIndex += classifyResults[i];
        }
        classIndex /= 50;
        
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
    
    private double[] calculateInstanceWeight(double[] weights){
        
        double sumOfWeights = 0.0;
        
        for (double value:weights){
            sumOfWeights += value;
        }
        
        System.out.println("Sum of instance Weights: " + sumOfWeights);
        
        for (int i = 0; i < dataModel.numInstances(); i++){
            
            weights[i] = weights[i]/sumOfWeights;
            System.out.println(i + " weight: " + weights[i]);
        }
        
        return weights;
    }
    
}
