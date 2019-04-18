/*
 * This class is an implementation of a KNN Ensemble. There some general design
 * decisions made that are justified based on proposed methods for sampling 
 * which are referenced through out this class implementation.
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
        dataModel = data;
    }
    
    /**
     * Classifies an instances.
     * @param object The object that is to be classified.
     * @return The class index of the result.
     */
    public double classifyInstance(Instance object){
        int classIndex = 0;
        
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
    
}
