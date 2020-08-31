package es.upm.oeg.semanticmeasures.impl.monolingual;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import es.upm.oeg.cidercl.util.StringTools;
import es.upm.oeg.semanticmeasures.Relatedness;


import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JException;

/**
 * Defines some basic methods to operate with the computation of the Cosine Similarity metric (see https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-word2vec)
 * 
 * @author Jorge Gracia, Marta Lanau
 *
 */

public abstract class WordVectors implements Relatedness{

	/** Location (local file system) of the vectors. */	
	//private static final String WORD_VECTORS_PATH = "./embeddings/monolingual/GoogleNews-vectors-negative300.bin.gz";
	//private String WORD_VECTORS_PATH = "GoogleNews-vectors-negative300.bin.gz";
	//private Word2Vec vector = WordVectorSerializer.readWord2VecModel(WORD_VECTORS_PATH);
    private Word2Vec vec;
    private String lang;
	
	public WordVectors(String lang) {
   	
		try{
			String WORD_VECTORS_PATH = "./embeddings/monolingual/cc." + lang + ".300.vec";
			this.vec = WordVectorSerializer.readWord2VecModel(new File(WORD_VECTORS_PATH),false);
	    } catch(ND4JException e){
	        e.printStackTrace();
	    }
		//this.vec = vector;
		this.lang = lang;
	}
	
	public WordVectors() {
		String WORD_VECTORS_PATH = "./embeddings/monolingual/GoogleNews-vectors-negative300.bin.gz";
		this.vec = WordVectorSerializer.readWord2VecModel(new File(WORD_VECTORS_PATH),false);
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
		for(String w : words) {
			if(vec.hasWord(w)) {
				wordsInModel.add(w);
			}
		}			
		if(!wordsInModel.isEmpty())
			meanVector = vec.getWordVectorsMean(wordsInModel);								
		return meanVector;
	}

}