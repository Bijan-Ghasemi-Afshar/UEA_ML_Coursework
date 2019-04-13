/*
 * This is the implementation of the K-Nearest Neighbour classifier with 
 * additional functionalities. It extends the AbstractClassifier class from
 * the Weka machine learning package. This implementation assumes all attributes
 * are real values.
 */
package uea_ml_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * 10/04/2019
 * @author Bijan Ghasemi Afshar (100125463)
 */
public class KNN extends AbstractClassifier{

    // Class properties
    private Instances dataModel;
    private boolean standardise;
    private double[] means;
    private double[] standardDeviations;
    private int k;
    private int[] votes;
    
    /**
     * Constructor for initialising the KNN object.
     */
    public KNN(){
        this.k = 1;
        this.standardise = true;
    }
    
    /**
     * Constructor for initialising the KNN object.
     * @param standardise Flag to whether standardise values of not.
     */
    public KNN(boolean standardise){
        this.k = 1;
        this.standardise = standardise;
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
        votes = new int[dataModel.numClasses()];
        
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
        
        if (this.standardise){   
            standardiseDataModelAttr();
        } else {}
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
        double[] newThing = new double[3];
        Instance[] closestInstances = new Instance[k];
        Instances clonedData = new Instances(dataModel);
        int closestInstanceIndex = 0, classIndex = 0, numOfVotes = 0;
        resetVotes();   // Reset votes from previous classification

        // Standardise Object
        if (this.standardise){   
            standardiseObject(object);
        } else {}
        
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
            results[i] = (double)votes[i]/(double)k;
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
            
            votes[i] = 0;
            
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
    
}
