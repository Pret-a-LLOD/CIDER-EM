package es.upm.oeg.cidercl.util;

import java.util.Properties;

import org.apache.jcs.JCS;
import org.apache.jcs.engine.control.CompositeCacheManager;
import org.apache.log4j.Logger;

/** 
 * Class to manipulate cache (essentially initialize it) 
 * 
 * @author OEG group (Universidad Politécnica de Madrid) and SID group (Universidad de Zaragoza)
 *
 */
public class CacheHandler {
	
	private static Logger log = Logger.getLogger(CacheHandler.class);

	
	//Initialize the cache
	public static JCS initializeCache(String typeCache) {		
		
		
		
		CompositeCacheManager ccm = CompositeCacheManager.getUnconfiguredInstance();
		Properties props = new Properties();

		props.put("jcs.default","DC");
		props.put("jcs.default.cacheattributes",
		          "org.apache.jcs.engine.CompositeCacheAttributes");
		// lots more props.put - this is basically the contents of cache.ccf

		ccm.configure(props);		
		
		
		JCS cache = null;
			try {
				cache = JCS.getInstance(typeCache);	           
			}catch (Exception e) {	       
				log.error("Problem initializing cache for region name [" + typeCache + "]. " + e.toString() );
			}		 
		return cache;
	}
		

	

}
