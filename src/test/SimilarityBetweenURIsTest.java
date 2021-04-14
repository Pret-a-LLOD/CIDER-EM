package test;

//please include cidercl.jar in classpath
import es.upm.oeg.semanticmeasures.impl.monolingual.CosineSimilarityBetweenOntologyEntities;
import es.upm.oeg.semanticmeasures.impl.crosslingual.CosineSimilarityCLBetweenOntologyEntities;
import es.upm.oeg.cidercl.extraction.OntologyExtractor;

public class SimilarityBetweenURIsTest {

	 public static void main(String args[]) {
		  
		double monoSim, clSim;
		 
		String uriA = "http://cmt_en#c-4268400-1612321";
		String ontologyA = "file:./test/cmt-en.owl";
		String uriB = "http://cmt_es#c-7348985-5560772";
		String ontologyB = "file:./test/cmt-es.owl";
		String uriC = "http://sigkdd_en#c-0407849-6361536";
		String ontologyC = "file:./test/sigkdd-en.owl";
		   
	//	CosineSimilarityCLBetweenOntologyEntities clesa = new CosineSimilarityCLBetweenOntologyEntities();

	//	clSim = clesa.getValue(OntologyExtractor.modelObtaining(ontologyA),  uriA,  null, OntologyExtractor.modelObtaining(ontologyB));
	
		CosineSimilarityBetweenOntologyEntities measure = new CosineSimilarityBetweenOntologyEntities();
		monoSim = measure.getValue(OntologyExtractor.modelObtaining(ontologyA), uriA , OntologyExtractor.modelObtaining(ontologyC), uriC);

	//	System.out.println( "---CROSS-LINGUAL SIMILARITY between " + uriA + " and " + uriB + ": " + clSim + " ---");		  			  	
		System.out.println( "---MONO-LINGUAL SIMILARITY between " + uriA + " and " + uriC + ": " + monoSim + " ---");
		
	 }// end-main	
}
