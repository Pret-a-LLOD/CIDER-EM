package es.upm.oeg.semanticmeasures.impl.crosslingual;


import java.util.ArrayList;
import java.util.Arrays;

import org.apache.log4j.Logger;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import org.nd4j.linalg.api.ndarray.INDArray;

import es.upm.oeg.semanticmeasures.Relatedness;

/** 
 * Defines the basic methods to operate with the CL-ESA metric. 
 * CL-ESA implementation (wsd.jar) by Kartik Asooja for the Monnet project 
 * 
 * @author Jorge Gracia
 * @author Kartik Asooja
 *
 */
public abstract class WordVectorsCL implements Relatedness{
	
	private static Logger log = Logger.getLogger(WordVectorsCL.class);	
	
	private static final String SRC_VECTORS_PATH = "././embeddings/mapped/SRC_MAPPED_unsupervised_en.EMB";
	private static final Word2Vec src_vector = WordVectorSerializer.readWord2VecModel(SRC_VECTORS_PATH);
	private static final String TRG_VECTORS_PATH = "././embeddings/mapped/TRG_MAPPED_unsupervised_es.EMB";
	private static final Word2Vec trg_vector = WordVectorSerializer.readWord2VecModel(TRG_VECTORS_PATH);
	
    private Word2Vec src_vec;
    private String langS;
    private Word2Vec trg_vec;
    private String langT;
	
	public WordVectorsCL(String langS, String langT){		
		this.src_vec = src_vector;
		this.trg_vec = trg_vector;
		this.langS = langS;
		this.langT = langT;
	}
	
	public WordVectorsCL(){
	}
	
	public Word2Vec getSRC_map() {return this.src_vec;}
	public Word2Vec getTRG_map() {return this.trg_vec;}
	public String getLangS() {return this.langS;}
	public String getLangT() {return this.langT;}
    
    /*
     * Compute the cosine similarity between two words
     */
//    public double score(String s, String t) {		
//	    return vec.similarity(StringTools.splitCamelCase(s),StringTools.splitCamelCase(t));
//	}

	public INDArray getMeanVectorFromWords(ArrayList<String> words, String lang) {
		
		Word2Vec vec = null;
		if(lang.equals(this.langS)) 		
			vec = this.src_vec;
		else {
			if(lang.equals(this.langT))
				vec = this.trg_vec;
		}		
		INDArray meanVector = null;
		ArrayList<String> wordsInModel = new ArrayList<>();			
		if(vec != null) {
			for(String w : words) {				
				if(vec.hasWord(w)) {
					System.out.println("vector de " + w + ": " + Arrays.toString(vec.getWordVector(w)));
					wordsInModel.add(w);
				}					
			}	
			System.out.println(Arrays.toString(wordsInModel.toArray()));		
			if(!wordsInModel.isEmpty()) 
					meanVector = vec.getWordVectorsMean(wordsInModel);	
		}		
		return meanVector;
	}

	
}