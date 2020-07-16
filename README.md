
# CIDER-EM

Context and Inference baseD ontology alignER. 

**CIDER-EM** is a word-embedding-based system for monolingual and cross-lingual ontology alignment. Evolves the CIDER-CL tool by including the use of word embeddings.

----------
### Features
- It operates as **aligner** between two ontologies (using *Aligner.java* class) 
-  It aligns **classes** and **properties**, but not instances.
- It can operate in two modes: 
	 * **Monolingual.** 
	     * It gets the word embeding from the ontology's language.
	     * It computes the word-embedding-based value of the relatedness between two entities from two ontologies in the same language.
		 * 	It performs elementary computations of *"cosine similarity"* to compare several features of the ontological description of the compared entities. 
		 * 	Such features are combined by means of multilayer perceptrons (one for classes and another one for properties) to produce a final value. 
	 * **Crosslingual.**
		 * It gets the word-embedings from both ontologies' languages.
		 * *Preprocessing:* The word embeddings must have the same nature and must be rotated in order to be in the same vector space, which allow compute the cosine distance between entities from different languages. Those crosslingual mappings are done with [Vecmap](https://github.com/artetxem/vecmap).
		 *  It computes the word-embedding-based value of the relatedness between two entities from two ontologies in different languages.
		 * 	It performs elementary computations of *"cosine similarity"* to compare several features of the ontological description of the compared entities. 
		 * 	Such features are combined by means of multilayer perceptrons (one for classes and another one for properties) to produce a final value. 

- This aligner is not intended to operate with large ontologies.



