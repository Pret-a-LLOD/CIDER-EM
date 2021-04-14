package es.upm.oeg.semanticmeasures.impl.monolingual;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import es.upm.oeg.cidercl.util.StopWords;
import es.upm.oeg.cidercl.util.StringTools;
import es.upm.oeg.semanticmeasures.Relatedness;
import eu.monnetproject.wsd.utils.Language;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Defines some basic methods to operate with the computation of the Cosine Similarity metric (see https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-word2vec)
 *
 * @author Marta Lanau
 */

public abstract class WordVectors implements Relatedness{
	
    /** Location (local file system) of the vectors.
     * TODO this should be parameterized  */	
	private static final String WORD_VECTORS_PATH = "./embeddings/monolingual/GoogleNews-vectors-negative300.bin.gz";
	//private static final String WORD_VECTORS_PATH = "G:\\Unidades compartidas\\LanguageResources_LOD\\CIDER-EM_embeddings\\monolingual\\GoogleNews-vectors-negative300.bin.gz";
	private static final Word2Vec vector = WordVectorSerializer.readWord2VecModel(WORD_VECTORS_PATH);
    
	private Word2Vec vec;
    private String lang;
	
	public WordVectors(String lang){
		this.vec = vector;
		this.lang = lang;
	}
	
	public WordVectors() {
		this.vec = vector;
	}
    
    public Word2Vec getModel() {return this.vec;}
    public String getLang() {return this.lang;}
    
    /*
     * Compute the cosine similarity between two words
     */
    public double score(String s, String t) {		
	    return vec.similarity(StringTools.splitCamelCase(s),StringTools.splitCamelCase(t));
	}	
	
	public INDArray getMeanVectorFromWords(ArrayList<String> words_rep) {		
		Set<String> words = new HashSet<>(words_rep);
		INDArray meanVector = null;
		ArrayList<String> wordsInModel = new ArrayList<>();
		StopWords stopWords = new StopWords();
		for(String w : words) {
			if(vec.hasWord(w)) 
				wordsInModel.add(w);
			else {
				String [] array = w.split("(?=\\p{Upper})");
				if (array.length >1) {					
					for (String a : array){
						a  = stopWords.removeStopWords(new Language(getLang()), a);
						if(!a.equals("") && vec.hasWord(a)) 
							wordsInModel.add(a);
					}
				}					
			}
		}
		
		if(!wordsInModel.isEmpty())
			meanVector = vec.getWordVectorsMean(wordsInModel);								
		return meanVector;
	}

}