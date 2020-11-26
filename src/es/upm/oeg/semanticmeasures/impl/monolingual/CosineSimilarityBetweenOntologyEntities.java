package es.upm.oeg.semanticmeasures.impl.monolingual;

import java.util.ArrayList;
import java.util.Arrays;
//import java.util.Collections;
import java.util.Collections;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;

import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Utils;

import com.hp.hpl.jena.ontology.OntModel;
import com.hp.hpl.jena.ontology.OntResource;

import es.upm.oeg.cidercl.extraction.OntologyExtractor;
import es.upm.oeg.cidercl.util.ANNUtil;
import es.upm.oeg.cidercl.util.OntologyUtil;
import es.upm.oeg.semanticmeasures.RelatednessBetweenOntologyEntities;
import eu.monnetproject.wsd.utils.Language;
import es.upm.oeg.cidercl.util.StopWords;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Computes the semantic similarity between two ontology entities based on the Cosine Similarity metric
 */
public class CosineSimilarityBetweenOntologyEntities extends WordVectors implements RelatednessBetweenOntologyEntities{

	//attributes
	private static double DEFAULT_SCORE = 0.0; //If the value is set to -1 it will be interpreted as "missing value"
	private static Logger log = Logger.getLogger(CosineSimilarityBetweenOntologyEntities.class);
	private static final String CLASSIFIER_CLASS_FILE = "ANN/ANN_class.model";
	private static final String CLASSIFIER_PROP_FILE = "ANN/ANN_prop.model";
	private static Classifier classifier_class = ANNUtil.loadClassifier(CLASSIFIER_CLASS_FILE);
	private static Classifier classifier_prop = ANNUtil.loadClassifier(CLASSIFIER_PROP_FILE);   
	
	public CosineSimilarityBetweenOntologyEntities(String lang) {
		super(lang);
	}
	
	public CosineSimilarityBetweenOntologyEntities() {
		super();
	}
	
	public double getScoreBetweenLabels(OntModel model1, String uri1, OntModel model2, String uri2){	
		// Retrieve all the labels. If there is no labels, the URI fragment is used instead
		ArrayList<String> labelsSource = OntologyExtractor.getLabels(model1, uri1, null); //'language' is null so it retrieves labels in all the languages
		if (labelsSource.isEmpty()) 
			labelsSource.add(OntologyExtractor.getUriFragment(model1, uri1));
		ArrayList<String> labelsTarget = OntologyExtractor.getLabels(model2, uri2, null);
		if (labelsTarget.isEmpty()) 
			labelsTarget.add(OntologyExtractor.getUriFragment(model2, uri2));				
//		Collections.sort(labelsSource);
//      Collections.sort(labelsTarget);        
		if (labelsSource.equals(labelsTarget)) {
			return 1;
		}else {
			INDArray meanVectorLabelsSource = getMeanVectorFromWords(labelsSource);
			INDArray meanVectorLabelsTarget = getMeanVectorFromWords(labelsTarget);
			if(meanVectorLabelsSource==null || meanVectorLabelsTarget==null)
				return DEFAULT_SCORE;
			else 
				return Transforms.cosineSim(meanVectorLabelsSource, meanVectorLabelsTarget);	
		}	
	}
	
	public double getScoreBetweenComments(OntModel model1, String uri1, OntModel model2, String uri2){		
		// Retrieve all the comments
		ArrayList<String> commentsSource = OntologyExtractor.getComments(model1, uri1, null); //language is null so it retrieves comments in all the languages
		ArrayList<String> commentsTarget = OntologyExtractor.getComments(model2, uri2, null);		
		if (commentsSource.equals("") || commentsTarget.equals("") || commentsSource.isEmpty() || commentsTarget.isEmpty()) 
			return DEFAULT_SCORE;
		else {			
			String allCommentsSource = "";
			for(String cs : commentsSource) 
				allCommentsSource += " " + cs;						
			String allCommentsTarget = "";
			for(String ct : commentsTarget) 
				allCommentsTarget += " " + ct;			
			StopWords stopWords = new StopWords();
			allCommentsSource = stopWords.removeStopWords(new Language(getLang()), allCommentsSource);
			allCommentsTarget = stopWords.removeStopWords(new Language(getLang()), allCommentsTarget);			
			commentsSource = new ArrayList<String>(Arrays.asList(allCommentsSource.split("\\s")));
			commentsTarget = new ArrayList<String>(Arrays.asList(allCommentsTarget.split("\\s")));			
//			Collections.sort(commentsSource);
//	        Collections.sort(commentsTarget);	         	
			if (commentsSource.equals(commentsTarget))
				return 1;
			else {				
				INDArray meanVectorLabelsSource = getMeanVectorFromWords(commentsSource);
				INDArray meanVectorLabelsTarget = getMeanVectorFromWords(commentsTarget);				
				if(meanVectorLabelsSource==null || meanVectorLabelsTarget==null)
					return DEFAULT_SCORE;
				else
					return Transforms.cosineSim(meanVectorLabelsSource, meanVectorLabelsTarget);			
			}
		}
	}
	
	public double getScoreBetweenSuperterms(OntModel model1, String uri1, OntModel model2, String uri2) {
		ArrayList<String> sourceURIs = OntologyExtractor.getSuperterms(model1, uri1);
		ArrayList<String> targetURIs = OntologyExtractor.getSuperterms(model2, uri2);	
		return scoreBetweenSetsOfURIs(model1, sourceURIs, model2, targetURIs);		
	}

	public double getScoreBetweenDirectSuperterms(OntModel model1, String uri1, OntModel model2, String uri2) {
		ArrayList<String> sourceURIs = OntologyExtractor.getDirectSuperterms(model1, uri1);
		ArrayList<String> targetURIs = OntologyExtractor.getDirectSuperterms(model2, uri2);	
		return scoreBetweenSetsOfURIs(model1, sourceURIs, model2, targetURIs);		
	}

	public double getScoreBetweenSubterms(OntModel model1, String uri1,	OntModel model2, String uri2) {
		ArrayList<String> sourceURIs = OntologyExtractor.getSubterms(model1, uri1);
		ArrayList<String> targetURIs = OntologyExtractor.getSubterms(model2, uri2);
		return scoreBetweenSetsOfURIs(model1, sourceURIs, model2, targetURIs);
	}

	public double getScoreBetweenDirectSubterms(OntModel model1, String uri1, OntModel model2, String uri2) {
		ArrayList<String> sourceURIs = OntologyExtractor.getDirectSubterms(model1, uri1);
		ArrayList<String> targetURIs = OntologyExtractor.getDirectSubterms(model2, uri2);	
		return scoreBetweenSetsOfURIs(model1, sourceURIs, model2, targetURIs);
	}

	public double getScoreBetweenPropertiesOfClasses(OntModel model1, String uri1, OntModel model2, String uri2) {
		ArrayList<String> sourceURIs = OntologyExtractor.getPropertiesOfClass(model1, uri1);
		ArrayList<String> targetURIs = OntologyExtractor.getPropertiesOfClass(model2, uri2);	
		return scoreBetweenSetsOfURIs(model1, sourceURIs, model2, targetURIs);
	}

	public double getScoreBetweenDirectPropertiesOfClasses(OntModel model1, String uri1, OntModel model2, String uri2) {
		ArrayList<String> sourceURIs = OntologyExtractor.getDirectPropertiesOfClass(model1, uri1);
		ArrayList<String> targetURIs = OntologyExtractor.getDirectPropertiesOfClass(model2, uri2);	
		return scoreBetweenSetsOfURIs(model1, sourceURIs, model2, targetURIs);
	}	
	
	public double getScoreBetweenDomainsOfProperties(OntModel model1, String uri1, OntModel model2, String uri2) {
		ArrayList<String> sourceURIs = OntologyExtractor.getDomainsOfProperty(model1, uri1);
		ArrayList<String> targetURIs = OntologyExtractor.getDomainsOfProperty(model2, uri2);
		return scoreBetweenSetsOfURIs(model1, sourceURIs, model2, targetURIs);
	}

	public double getScoreBetweenDirectDomainsOfProperties(OntModel model1, String uri1, OntModel model2, String uri2) {
		ArrayList<String> sourceURIs = OntologyExtractor.getDirectDomainsOfProperty(model1, uri1);
		ArrayList<String> targetURIs = OntologyExtractor.getDirectDomainsOfProperty(model2, uri2);	
		return scoreBetweenSetsOfURIs(model1, sourceURIs, model2, targetURIs);
	}
	
	private double getScoreBetweenRangesOfObjectProperties(OntModel model1, String uri1, OntModel model2, String uri2){
		OntResource r1 = (OntResource) ((OntModel) model1).getOntResource((String) uri1);
		OntResource r2 = (OntResource) ((OntModel) model2).getOntResource((String) uri2);		
		if (r1.isObjectProperty() && r2.isObjectProperty()){
			ArrayList<String> sourceURIs = OntologyExtractor.getRangesOfObjectProperty(model1, uri1);
			ArrayList<String> targetURIs = OntologyExtractor.getRangesOfObjectProperty(model2, uri2);	
			return scoreBetweenSetsOfURIs(model1, sourceURIs, model2, targetURIs);			
		} else return 0.0;
		
	}

	// Comparison is based on equality of URIs purely
	private double getSimilarityBetweenRangesOfDatatypeProperties(OntModel model1, String uri1, OntModel model2, String uri2){
		OntResource r1 = (OntResource) ((OntModel) model1).getOntResource((String) uri1);
		OntResource r2 = (OntResource) ((OntModel) model2).getOntResource((String) uri2);
		double score = DEFAULT_SCORE; //default value
		
		if (r1.isDatatypeProperty() && r2.isDatatypeProperty()){		
			ArrayList<String> rangeURIsSource = OntologyExtractor.getRangesOfDatatypeProperty(model1, uri1);
			ArrayList<String> rangeURIsTarget = OntologyExtractor.getRangesOfDatatypeProperty(model2, uri2);		
			if ((rangeURIsSource != null) && (!rangeURIsSource.isEmpty()) && (rangeURIsTarget != null) && (!rangeURIsTarget.isEmpty())){					
				if (rangeURIsSource.containsAll(rangeURIsTarget) && rangeURIsTarget.containsAll(rangeURIsSource))
					score = 1.0;
				else score = 0.0;
			}
		} else score = 0.0;					
		return score;
	}
	
	public double getScoreBetweenRangesOfProperties(OntModel model1, String uri1, OntModel model2, String uri2) {
		OntResource r1 = (OntResource) ((OntModel) model1).getOntResource((String) uri1);
		OntResource r2 = (OntResource) ((OntModel) model2).getOntResource((String) uri2);		
		if (r1.isObjectProperty() && r2.isObjectProperty()){
			return getScoreBetweenRangesOfObjectProperties(model1, uri1, model2, uri2);
		} else if (r1.isDatatypeProperty() && r2.isDatatypeProperty()){
			return getSimilarityBetweenRangesOfDatatypeProperties(model1, uri1, model2, uri2);
		}
		else return 0.0;	
	}

	public double getScoreBetweenRelatedClasses(OntModel model1, String uri1, OntModel model2, String uri2) {
		int depth = 1;		
		ArrayList<String> relatedClassesURIsSource = OntologyExtractor.getRelatedTermsOfClass(model1, uri1, depth);
		ArrayList<String> relatedClassesURIsTarget = OntologyExtractor.getRelatedTermsOfClass(model2, uri2, depth);		
		return scoreBetweenSetsOfURIs(model1, relatedClassesURIsSource, model2, relatedClassesURIsTarget);
	}
	
	public double getScoreBetweenEquivalentTerms(OntModel model1, String uri1, OntModel model2, String uri2) {
		ArrayList<String> sourceURIs = OntologyExtractor.getEquivalentTerms(model1, uri1);
		ArrayList<String> targetURIs = OntologyExtractor.getEquivalentTerms(model2, uri2);	
		return scoreBetweenSetsOfURIs(model1, sourceURIs, model2, targetURIs);
	}

	private double scoreBetweenSetsOfURIs(OntModel model1, ArrayList<String> urisSource, OntModel model2, ArrayList<String> urisTarget){
		
		double score = DEFAULT_SCORE; //default value		
		if ((urisSource != null) && (!urisSource.isEmpty()) && (urisTarget != null) && (!urisTarget.isEmpty())){		
			// if sets of URIs are the same then result is maximum, otherwise compute similarity (do not use "equals" method as the order in the vector of URIs does not matter in this case, and "equals" considers the order) 
			if (urisSource.containsAll(urisTarget) && urisTarget.containsAll(urisSource)) {
				score = 1.0;
			} else{
				// Retrieve labels of terms. If there are no labels, URI fragments are used instead
				ArrayList<String> termsLabelsSource = new ArrayList<String>();
				for (int i=0; i < urisSource.size(); i++){
					ArrayList<String> termLabels = OntologyExtractor.getLabels(model1, urisSource.get(i), null);
					if (termLabels.isEmpty()) termLabels.add(OntologyExtractor.getUriFragment(model1, urisSource.get(i))); 
					termsLabelsSource.addAll(termLabels);
				}					
				ArrayList<String> termsLabelsTarget = new ArrayList<String>();
				for (int i=0; i < urisTarget.size(); i++){
					ArrayList<String> termLabels = OntologyExtractor.getLabels(model2, urisTarget.get(i), null);
					if (termLabels.isEmpty()) termLabels.add(OntologyExtractor.getUriFragment(model2, urisTarget.get(i))); 
					termsLabelsTarget.addAll(termLabels);
				}				
//				Collections.sort(termsLabelsSource);
//		        Collections.sort(termsLabelsTarget);		         	
				if (termsLabelsSource.equals(termsLabelsTarget))
					return 1;
				else {	
					INDArray meanVectorLabelsSource = getMeanVectorFromWords(termsLabelsSource);
					INDArray meanVectorLabelsTarget = getMeanVectorFromWords(termsLabelsTarget);
					
					// Compute Cosine Similarity				
					if(meanVectorLabelsSource==null || meanVectorLabelsTarget==null)
						score = DEFAULT_SCORE;
					else
						score = Transforms.cosineSim(meanVectorLabelsSource, meanVectorLabelsTarget);	
				}
			}
		}
		return score;		
	}
	
	
	/**
	 * Computes the embeddings-based value of the relatedness between two entities from two ontologies.
	 * It performs elementary computations of "cosine similarity" to compare several features of the ontological 
	 * description of the compared entities. Such features are combined by means of multilayer perceptrons (one
	 * for classes and another one for properties) to produce a final value. 
	 * 
	 *  @param object1 URI of the source ontology
	 *  @param element1 URI of the source entity
	 *  @param object2 URI of the target ontology
	 *  @param element2 URI of the target element
	 *  @return relatedness value 
	 */
	public double getValue(Object object1, Object element1, Object object2,	Object element2) {

		double score = 0.0;							
		OntResource r1 = (OntResource) ((OntModel) object1).getOntResource((String) element1);
		OntResource r2 = (OntResource) ((OntModel) object2).getOntResource((String) element2);
		
		//if ontology + uris are the same, or uris correspond to equivalent entities, then returns 1.0
		if (OntologyUtil.areEquivalentEntities (object1, element1, object2, element2)){
			log.debug("Found equivalence between " + element1 + " and " + element2);
			return 1.0;
		}		
		if (r1.isClass() && r2.isClass()){			
			Instance instance;	
			
//			double[] sim = new double[10];
//			sim[0] = getScoreBetweenLabels((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);  
//			sim[1] = getScoreBetweenComments((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
//			sim[2] = getScoreBetweenEquivalentTerms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
//			sim[3] = getScoreBetweenSubterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
//			sim[4] = getScoreBetweenSuperterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
//			sim[5] = getScoreBetweenDirectSuperterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
//			sim[6] = getScoreBetweenDirectSubterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
//			sim[7] = getScoreBetweenPropertiesOfClasses((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
//			sim[8] = getScoreBetweenDirectPropertiesOfClasses((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
//			sim[9] = getScoreBetweenRelatedClasses((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
//			//substitute missing values (sim -1.0) by proper symbols ("?" in weka).
//			for (int i=0; i<=9; i++){
//				if (sim[i] == -1.0d) {
//					sim[i] = Utils.missingValue();
//				}
//			}
			
			double[] sim = new double[5];
			sim[0] = getScoreBetweenLabels((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);  
			sim[1] = getScoreBetweenSubterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
			sim[2] = getScoreBetweenDirectSuperterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
			sim[3] = getScoreBetweenDirectSubterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
			sim[4] = getScoreBetweenDirectPropertiesOfClasses((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
			//substitute missing values (sim -1.0) by proper symbols ("?" in weka).
			for (int i=0; i<=4; i++){
				if (sim[i] == -1.0d) {
					sim[i] = Utils.missingValue();
				}
			}
			
			instance = new DenseInstance(1.0, sim);
			try {
				score = classifier_class.classifyInstance(instance);
			} catch (Exception e) {
				log.error("Classifier error when computing relatedness between " + element1 + " and " + element2 + "\n" + e.toString());
			}
			
		} else if (r1.isProperty() && r2.isProperty()){
			
			Instance instance;
			
//			double[] sim = new double[10];
//			sim[0] = getScoreBetweenLabels((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);  
//			sim[1] = getScoreBetweenComments((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
//			sim[2] = getScoreBetweenEquivalentTerms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
//			sim[3] = getScoreBetweenSubterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
//			sim[4] = getScoreBetweenSuperterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
//			sim[5] = getScoreBetweenDirectSuperterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
//			sim[6] = getScoreBetweenDirectSubterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
//			sim[7] = getScoreBetweenDomainsOfProperties((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
//			sim[8] = getScoreBetweenDirectDomainsOfProperties((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);
//			sim[9] = getScoreBetweenRangesOfProperties((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
//			//substitute missing values (sim -1.0) by proper symbols ("?" in weka).
//			for (int i=0; i<=9; i++){
//				if (sim[i] == -1.0d) {
//					sim[i] = Utils.missingValue();
//				}				
//			}	
						
			double[] sim = new double[3];			
			sim[0] = getScoreBetweenLabels((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);  
			sim[1] = getScoreBetweenSubterms((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
			sim[2] = getScoreBetweenRangesOfProperties((OntModel) object1, (String) element1, (OntModel) object2, (String) element2);		
			//substitute missing values (sim -1.0) by proper symbols ("?" in weka).
			for (int i=0; i<=2; i++){
				if (sim[i] == -1.0d) {
					sim[i] = Utils.missingValue();
				}
			}
			
			instance = new DenseInstance(1.0, sim);
			try {
				score = classifier_prop.classifyInstance(instance);
			} catch (Exception e) {
				log.error("Classifier error when computing relatedness between " + element1 + " and " + element2 + "\n" + e.toString());
			}	
			
		} else 
			score = 0;		
		score = Math.min(score, 1.0);
		score = Math.max(0,  score); 		
		log.debug("Word2Vec-based relatedness between " + element1 + " and " + element2 + ": " + score);

		//THRESHOLD
		if (score < 0.2)
			score = 0;
		
		return score;	
		
	}

	//uncomment for testing
//	public static void main(String[] args) {
//			
//			String uri1 = "http://oaei.ontologymatching.org/2011/benchmarks/101/onto.rdf#Article";
//			String uri2 = "http://oaei.ontologymatching.org/2011/benchmarks/201/onto.rdf#hazdn";		
//			String ontology1 = "http://oaei.ontologymatching.org/2011/benchmarks/101/onto.rdf";
//			String ontology2 = "http://oaei.ontologymatching.org/2011/benchmarks/201/onto.rdf";			
//			String lang = "en";
//			
//			CosineSimilarityBetweenOntologyEntities measure = new CosineSimilarityBetweenOntologyEntities(lang);
//			
//			System.out.println(measure.getValue(OntologyExtractor.modelObtaining(ontology1), uri1 , OntologyExtractor.modelObtaining(ontology2), uri2));
//			log.info("finished");
//	}

}