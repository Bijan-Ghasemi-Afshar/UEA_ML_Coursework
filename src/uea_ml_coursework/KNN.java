/*
 * This is the implementation of the K-Nearest Neighbour classifier with 
 * additional functionalities. It extends the AbstractClassifier class from
 * the Weka machine learning package. This implementation assumes all attributes
 * are real values.
 */
package uea_ml_coursework;

import java.util.Arrays;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 10/04/2019
 * @author Bijan Ghasemi Afshar (100125463)
 */
public class KNN extends AbstractClassifier{

    // Class properties
    private Instances trainingData;
    private int k;
    private int[] votes;
    
    /**
     * Constructor for initialising the KNN object.
     */
    public KNN(){
        this.k = 1;
    }
    
    /**
     * Sets the K for KNN classifier.
     * @param k The number of closest neighbours considered for classification.
     */
    public void setK(int k){
        this.k = k;
    }
    
    /**
     * Builds the classifier by storing the training data.
     * @param data The classified training data
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainingData = data;
        votes = new int[trainingData.numClasses()];
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
            
            votes[i] = 0;
            
        }
        
    }
    
    @Override
    public double classifyInstance(Instance object){
        
        double closestMatch, eDistance = 0.0;
        double[] newThing = new double[3];
        Instance[] closestInstances = new Instance[k];
        Instances clonedData = new Instances(trainingData);
        int closestInstanceIndex = 0, classIndex = 0, numOfVotes = 0;
        resetVotes();   // Reset votes from previous classification

//        System.out.println("Number of K: " + closestInstances.length);
        // Go through all training data K times and choose smallest distance
        for (int i = 0; i < closestInstances.length; i++){
            
            closestMatch = Double.MAX_VALUE;
            
            for (int j = 0; j < clonedData.numInstances(); j++){
                
                eDistance = distance(clonedData.instance(j), object);
//                System.out.println(eDistance);
                // If distances are same choose randomly
                if (eDistance == closestMatch){
//                    System.out.println("Have to random between : " 
//                            + closestInstanceIndex + " and " + j);
                    if (Math.random() < 0.5){
//                        System.out.println("Chose " + j);
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
//            System.out.println("Closest Match: " + closestMatch);
            classIndex = (int)(closestInstances[i].classValue());
            votes[classIndex]++;
        }
        
        // Count the votes
        classIndex = 0;
        for (int i = 0; i < votes.length; i++){
//            System.out.println(i + " " + votes[i]);
            // If distances are same choose randomly
                if (numOfVotes == votes[i]){
//                    System.out.println("class value random  : " 
//                            + classIndex + " and " + i);
                    if (Math.random() < 0.5){
//                        System.out.println("Chose " + i);
                        numOfVotes = votes[i];
                        classIndex = i;
                    }
                    
                } else if (numOfVotes < votes[i]){
                numOfVotes = votes[i];
                classIndex = i;
            }
        }
//        System.out.println(classIndex);
        
        return (double)classIndex;
    }
    
    @Override
    public double[] distributionForInstance(Instance object){
        
        double[] results = new double[trainingData.numClasses()];
        
        classifyInstance(object);
        
        for (int i = 0; i < results.length; i++){
            
//            System.out.println(votes[i] + " " + k);
            results[i] = (double)votes[i]/(double)k;
//            System.out.println("Vote " + i + ": " + results[i]);
        }
        
        return results;
    }
    
}
