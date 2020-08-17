package es.upm.oeg.cidercl.classifiers.features; 

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.Properties;

import org.apache.log4j.Logger;
import org.semanticweb.owl.align.Alignment;
import org.semanticweb.owl.align.AlignmentProcess;
import org.semanticweb.owl.align.AlignmentException;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import es.upm.oeg.cidercl.extraction.OntologyExtractor;
import es.upm.oeg.semanticmeasures.impl.crosslingual.CosineSimilarityCLBetweenOntologyEntities;
import fr.inrialpes.exmo.align.impl.DistanceAlignment;
import fr.inrialpes.exmo.align.impl.MatrixMeasure;
import fr.inrialpes.exmo.ontowrap.OntowrapException;

/**
 * Alignment process for calculating the features that will be used 
 * for training the classifier (cross-lingual case).
 *  * Command synopsis is as follows:
 * <pre>
 * java -jar cidercl.jar -i es.upm.oeg.cidercl.classifiers.features.AlignerForFeaturesDataGenerationCL onto1 onto2 -o output_file
 * </pre>
 * 
 * @author Jorge Gracia 
 *
 */
 public class AlignerForFeaturesDataGenerationCL extends DistanceAlignment implements AlignmentProcess {
  
	private static Logger log = Logger.getLogger(AlignerForFeaturesDataGenerationCL.class);

	static final String CLASS_FEATURES_FILE = "./ANN/ClassFeaturesFileCL.arff";
	static final String PROP_FEATURES_FILE = "./ANN/PropFeaturesFileCL.arff";
	static final String REF_ALIGNMENTS_FILE = "./ANN/refalignCL.csv";
	
	Instances dataset_class;
	Instances dataset_prop;
	HashSet<String> alignedURIs;
	
	static CosineSimilarityCLBetweenOntologyEntities sim; 
		
	static final String langS = "en";
	static final String langT = "es";
   // Method dissimilarity = null;
   
    /** 
     * Creates the aligner, set type of alignment to "11" (one to one), load initial feature files if they 
     * already exist (otherwise it creates their skeleton), and creates a set containing the pairs of aligned 
     * URIs from the reference alignment 
     * 
     */
    public AlignerForFeaturesDataGenerationCL() {
    	
		
    	setSimilarity( new SimMatrixMeasure() );
		setType("11");  
		// Load initial features files if the files already exist
		File f_class = new File(CLASS_FEATURES_FILE);
		File f_prop = new File(PROP_FEATURES_FILE);		
		//If features files do not exist their skeletons are created
		if (!f_class.exists()) 	
			SkeletonFeaturesFileCreator.defineDatasetOfClasses(CLASS_FEATURES_FILE);
		try {
			dataset_class = DataSource.read(CLASS_FEATURES_FILE);
		} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
		}
		if (!f_prop.exists()) SkeletonFeaturesFileCreator.defineDatasetOfProperties(PROP_FEATURES_FILE);
		
		try {
			dataset_prop = DataSource.read(PROP_FEATURES_FILE);
		} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
		}
		// Create list of aligned URIs
		this.alignedURIs = createSetFromRefAlignmentFile(REF_ALIGNMENTS_FILE);
		sim = new CLEmbeddingsBasedFeaturesDataCreator(dataset_class, dataset_prop, alignedURIs, langS, langT);
		System.out.println("7");
    }
    
    
    protected class SimMatrixMeasure extends MatrixMeasure {
		public SimMatrixMeasure() {
		    similarity = false; // This is a distance matrix
		}
		
		// for the moment the assumption is it done that there is one language per ontology
		public double measure( Object o1, Object o2 ) throws Exception {

			
		 		try {
		 				String strOntA, strOntB, strTermA, strTermB;
		 					
		 				strOntA = ontology1().getFile().toString();
		 				strTermA = ontology1().getEntityURI(o1).toString();

		 				strOntB = ontology2().getFile().toString();
		 				strTermB = ontology2().getEntityURI(o2).toString();
		 				
		 				
		 			if ((strOntA == null) || (strOntB == null))
		 				return 1.; //ES DECIR, NO HACE LA COMPARACION
		 			else {
		 				return (1. - sim.getValue(OntologyExtractor.modelObtaining(strOntA), strTermA, OntologyExtractor.modelObtaining(strOntB), strTermB));		
		 			}
		 			
		 			
		 		} catch ( OntowrapException owex ) {
		 		    throw new AlignmentException( "Error getting entity name", owex );
		 		}

		
		}
		public double classMeasure( Object cl1, Object cl2 ) throws Exception {
		    return measure( cl1, cl2 );
		}
		public double propertyMeasure( Object pr1, Object pr2 ) throws Exception{
		    return measure( pr1, pr2 );
		}
		public double individualMeasure( Object id1, Object id2 ) throws Exception{
		    //return measure( id1, id2 );
			return 1.; //ignoro instancias
		}
    }


    
   
    // Creates a hashset with the uris that appear in the reference alignment 
    private HashSet<String> createSetFromRefAlignmentFile(String csvRefFile){
    	
    	HashSet<String> pairsOfURIsSet = new HashSet<String>();
    	BufferedReader br = null;
    	String line = "";
    	String cvsSplitBy = ",";
    		 
    	try {
    		 
			br = new BufferedReader(new FileReader(csvRefFile));
			
			while ((line = br.readLine()) != null) {
	 
			    // use comma as separator
				String[] fields = line.split(cvsSplitBy);
	 
				// add a concatenation of URI1 and URI2 into the hashSet
				pairsOfURIsSet.add(fields[0]+fields[1]);
				
			}
	 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
    	return pairsOfURIsSet;
    }
    
    
    
    /* Processing */
    //I MADE IT SYNCHRONIZED to avoid potential concurrency errors during writing
    synchronized public void align( Alignment alignment, Properties params ) throws AlignmentException {

		// Initialize matrix
		getSimilarity().initialize( ontology1(), ontology2());
	
		// Compute similarity/dissimilarity
		getSimilarity().compute(params);
	
		// Print matrix if asked
		// Extract alignment
		extract( type, params );
		
		// save data as ARFF
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(CLASS_FEATURES_FILE));
			writer.write(this.dataset_class.toString());
			writer.flush();
			writer.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(PROP_FEATURES_FILE));
			writer.write(this.dataset_prop.toString());
			writer.flush();
			writer.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
			
		
		log.info("Features files created for " + ontology1().getURI() + " " + ontology2().getURI() ); 
		
    }

}
