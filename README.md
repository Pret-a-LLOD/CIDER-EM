
# CIDER-EM

Context and Inference baseD ontology alignER. 

**CIDER-EM** is a word-embedding-based system for monolingual and cross-lingual ontology alignment deveoped by the SID group (University of Zaragoza). It evolves the [CIDER-CL](https://github.com/jogracia/cider-cl) tool, deveoped at OEG (Universidad Politécnica de Madrid) and SID (University of Zaragoza) by including the use of word embeddings.
   	

>    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

>    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

>    You should have received a copy of the GNU General Public License
    along with this program.  If not, see [http://www.gnu.org/licenses/].

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

### Word Embeddings
You can download some word embeddings [here](https://drive.google.com/drive/folders/188jUDHGBYrLYKLbVTVY0mmsT8OgvojW5?usp=sharing).

----------
> **Note:** This version is *under development*. You can find the last working version of the aligner [here](https://github.com/jogracia/cider-cl).

