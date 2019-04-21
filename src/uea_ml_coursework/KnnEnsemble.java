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
    private double[] classWeight;
    private KNN[] knnEnsemble;
    
    /**
     * Constructor for creating the KNN Ensemble.
     */
    public KnnEnsemble(){
        this.knnEnsemble = new KNN[4];
        this.classWeight = new double[4];
    }
    
    /**
     * Builds the ensemble classifier and sets each member of the ensemble.
     * @param data The training dataset.
     */
    public void buildClassifier(Instances data){
        
        // Initializing variables
        dataModel = new Instances(data);
        this.instanceWeights = new double[dataModel.numInstances()];
        double weightedError = 0.0;
        
        
        
        // Initialize all instance weights as 1
        for (int i = 0; i < instanceWeights.length; i++){
            instanceWeights[i] = 1;
        }
        instanceWeights = normalizeInstanceWeight(instanceWeights);
        
        // Populate the ensemble
        try {
        
            for (int i = 0; i < knnEnsemble.length; i++){
                knnEnsemble[i] = new KNN(true, true, true);
                knnEnsemble[i].buildClassifier(dataModel);
                int[] wrongClassifications = knnEnsemble[i].crossValidateTest();
//                System.out.println("size: " + wrongClassifications.length);
//                for (int j = 0; j < wrongClassifications.length; j++){
//                    System.out.println(wrongClassifications[j]);
//                }
                weightedError = calculateWeightedError(wrongClassifications);
//                System.out.println("The weighted Error: " + weightedError);
                
                // Calculating the class weight
                classWeight[i] = Math.log((1-weightedError)/weightedError) / 2;
                
                System.out.println("Class weight: " + classWeight[i]);
                
                instanceWeights = calculateInstanceWeight(wrongClassifications, 
                        classWeight[i]);
                
                
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
    
}
