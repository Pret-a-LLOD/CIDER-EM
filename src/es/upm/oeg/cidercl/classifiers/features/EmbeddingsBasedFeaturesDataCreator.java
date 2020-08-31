package es.upm.oeg.cidercl.classifiers.features;

import java.util.HashSet;

import org.apache.log4j.Logger;
import org.deeplearning4j.models.word2vec.Word2Vec;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import com.hp.hpl.jena.ontology.OntModel;
import com.hp.hpl.jena.ontology.OntResource;

import es.upm.oeg.semanticmeasures.impl.monolingual.CosineSimilarityBetweenOntologyEntities;

/**
 * Computes SoftTFIDF relatedness between the different elements (features) 
 * of two ontology entities and stores the results in a structure that will be used for
 * training the classifier. 
 * 
 * This class is not intended to be used in isolation (it is called from "AlignerFromFeaturesDataGeneration")
 * 
 * @author Jorge Gracia
 *
 */
public class EmbeddingsBasedFeaturesDataCreator extends CosineSimilarityBetweenOntologyEntities{

	private static Logger log = Logger.getLogger(EmbeddingsBasedFeaturesDataCreator.class);
	Instances class_features, property_features;
	HashSet<String> alignedURIs;
	Word2Vec vec;
	
	public EmbeddingsBasedFeaturesDataCreator(Instances classFeaturesDataset, Instances propFeaturesDataset, HashSet<String> alignedURIsSet, String lang) {
		super(lang);
		this.class_features = classFeaturesDataset;
		this.property_features = propFeaturesDataset;
		this.alignedURIs = alignedURIsSet;
	}
	
	/**
	 * Computes features between two ontology entities and updates the corresponding dataset for classes or properties (depending on the type of the entities)
	 *
	 * @param object1 source ontology
	 * @param element1 source entity
	 * @param object2 target ontology
	 * @param element2 target element
	 * @return The returned value is irrelevant (always -1) 
	 */
	public double getValue(Object object1, Object element1, Object object2,	Object element2) {
	
		OntResource r1 = (OntResource) ((OntModel) object1).getOntResource((String) element1);
		OntResource r2 = (OntResource) ((OntModel) object2).getOntResource((String) element2);
		
		if (r1.isClass() && r2.isClass()){
		
			double[] values = new double[class_features.numAttributes()];
			
			values[0] = class_features.attribute(0).addStringValue((String) element1);
			values[1] = class_features.attribute(1).addStringValue((String) element2);
			values[2] = getScoreBetweenLabels((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);  
//			values[3] = getScoreBetweenComments((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
			values[3] = getScoreBetweenEquivalentTerms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
			values[4] = getScoreBetweenSubterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
			values[5] = getScoreBetweenSuperterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
			values[6] = getScoreBetweenDirectSuperterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
			values[7] = getScoreBetweenDirectSubterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
			values[8] = getScoreBetweenPropertiesOfClasses((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
			values[9] = getScoreBetweenDirectPropertiesOfClasses((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
			values[10] = getScoreBetweenRelatedClasses((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);			
		
			//substitute missing values (sim -1.0) by proper symbols ("?" in weka).
			for (int i=2; i<=10; i++){
				if (values[i] == -1.0d) 
					values[i] = Utils.missingValue();
			}
			
			if (alignedURIs.contains((String) element1 + (String) element2) || (alignedURIs.contains((String) element2 + (String) element1)))
				values[12] = 1.0;
			else	
				values[12] = 0.0;
			
			
			Instance inst = new DenseInstance(1.0, values);
			
			this.class_features.add(inst);
		
		} else if (r1.isProperty() && r2.isProperty()){
			
			double[] values = new double[property_features.numAttributes()];
			
			values[0] = property_features.attribute(0).addStringValue((String) element1);
			values[1] = property_features.attribute(1).addStringValue((String) element2);
			values[2] = getScoreBetweenLabels((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);  
//			values[3] = getScoreBetweenComments((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
			values[3] = getScoreBetweenEquivalentTerms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
			values[4] = getScoreBetweenSubterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
			values[5] = getScoreBetweenSuperterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
			values[6] = getScoreBetweenDirectSuperterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
			values[7] = getScoreBetweenDirectSubterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
			values[8] = getScoreBetweenDomainsOfProperties((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
			values[9] = getScoreBetweenDirectDomainsOfProperties((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
			values[10] = getScoreBetweenRangesOfProperties((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		

			//substitute missing values (sim -1.0) by proper symbols ("?" in weka).
			for (int i=2; i<=10; i++){
				if (values[i] == -1.0d) 
					values[i] = Utils.missingValue();
			}
			
			if (alignedURIs.contains((String) element1 + (String) element2) || (alignedURIs.contains((String) element2 + (String) element1)))
				values[12] = 1.0;
			else	
				values[12] = 0.0;
			
			Instance inst = new DenseInstance(1.0, values);
			
			this.property_features.add(inst);
				
		} 
			
		return -1;	//during features computation the total score is meaningless
		}
	
	
	
}
