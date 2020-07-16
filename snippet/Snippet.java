package snippet;

public class Snippet {
	cd %CIDER_HOME%
	java -cp %ALIGN%/procalign.jar;%CIDER_HOME%/cidercl.jar fr.inrialpes.exmo.align.cli.GroupAlign -w %BENCH% -o CIDERCL_dummy  -n file:%BENCH%/101/onto.rdf -t onto.rdf -i es.upm.oeg.cidercl.classifiers.features.AlignerForFeaturesDataGeneration -Dnoinst=1
	
}

