package trolls;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import com.yahoo.yrlhaifa.haifa_utils.ds.IntegerHistogram;
import com.yahoo.yrlhaifa.haifa_utils.ds.MutableInt;

public class KLPost {

    enum Ngram {
        UNIGRAM, BIGRAM, TRIGRAM;
    }

    private static final char ngramSep = ' ';
    private static final String ngramBeg = "^";
    private static final String ngramEnd = "&";

    private static final double SMOOTHING_FACTOR = 2.0;
//    private int cnt=0;

    IntegerHistogram<String> base = new IntegerHistogram<String>();
    IntegerHistogram<String> target = new IntegerHistogram<String>();
    
    int basePosts=0, targetPosts=0;

    public List<NgramKL> calculateKL(String baseFileName, String targetFileName, Ngram type, int topK)
                    throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(baseFileName));
        String line;
        while ((line = reader.readLine()) != null) {  
        	if (line.trim().isEmpty()) {
        		continue;
        	}
        	basePosts++;
            collectNgrams(line, type, base);
//            cnt++;
//            if (cnt % 100000 ==0) {
//            	System.out.println("cnt: " + cnt);
//            }
        }
        reader.close();
        System.out.println("Finished based");
        reader = new BufferedReader(new FileReader(targetFileName));
        while ((line = reader.readLine()) != null) {
        	if (line.trim().isEmpty()) {
        		continue;
        	}
        	targetPosts++;
            collectNgrams(line, type, target);
//            cnt++;
//            if (cnt % 100000 ==0) {
//            	System.out.println("cnt: " + cnt);
//            }
        }
        System.out.println("Finished target");
        double kl, p, pbase;
        // use Java util's topNHeap in case it is too heavy to hold everything in memory
        ArrayList<NgramKL> nklist = new ArrayList<NgramKL>();
        for (Map.Entry<String, MutableInt> entry : target) {
            p = entry.getValue().doubleValue() / targetPosts;
            pbase = base.get(entry.getKey()).doubleValue() / basePosts;
            if (pbase == 0) {
                pbase = 1.0 / (basePosts * SMOOTHING_FACTOR);
            }
            kl = p * Math.log(p / pbase);
            nklist.add(new NgramKL(entry.getKey(), kl));
        }
        System.out.println("Sorting...");
        Collections.sort(nklist);
        return nklist.subList(0, topK);
    }

    public void collectNgrams(String line, Ngram type, IntegerHistogram<String> histo) {
        line = line.replaceAll("[^a-zA-Z0-9\\'\\- ]", "").toLowerCase();
        // add later: ():;-
        String[] parts = line.split("\\s+");
        HashSet<String> uniques = new HashSet<String>();
        if (parts.length == 0) {
        	return;
        }
        if (type == Ngram.UNIGRAM) {
            for (String word : parts) {              
                uniques.add(word);
            }
        }
        else if (type == Ngram.BIGRAM) {
            int cnt = 0;           
            uniques.add(ngramBeg + parts[0]);
            while (cnt < parts.length - 1) {                
                uniques.add(parts[cnt] + ngramSep + parts[cnt + 1]);
                cnt++;
            }           
            uniques.add(parts[parts.length - 1] + ngramEnd);
        }
        else if (type == Ngram.TRIGRAM) {
            if (parts.length < 2) {
                return;
            }
            int cnt = 0;            
            uniques.add(ngramBeg + parts[0] + ngramSep + parts[1]);
            while (cnt < parts.length - 2) {               
                uniques.add(parts[cnt] + ngramSep + parts[cnt + 1] + ngramSep + parts[cnt + 2]);
                cnt++;
            }            
            uniques.add(parts[parts.length - 2] + ngramSep + parts[parts.length - 1] + ngramEnd);
        }
        for (String s : uniques) {
        	histo.inc(s);
        }
        return;
    }

    public void testCollectNgrams() {
    	KLPost klpost = new KLPost();
        IntegerHistogram<String> his = new IntegerHistogram<String>();
        String line = "RB   CMA VB  RB  IN  DT  NN  TO  VB  NNS CC  VB  IN  DT  NN  IN  PRP MD  PRD NN NN NN";
        klpost.collectNgrams(line, Ngram.UNIGRAM, his);
        System.out.println(his.toOrderedString());
    }

    public int getBase(String ngram) {
        return base.get(ngram).intValue();
    }

    public int getTarget(String ngram) {
        return target.get(ngram).intValue();
    }
    
    public int getBasePosts() {
        return basePosts;
    }

    public int getTargetPosts() {
        return targetPosts;
    }
    

    public static void main(String[] args) {
    	KLPost klpost = new KLPost();
//        klpost.testCollectNgrams();
        try {

            // List<NgramKL> klist = klcalc.calculateKL("/Users/idoguy/Pro/Papers/Tips/useful15-30.txt",
            // "/Users/idoguy/Pro/Papers/Tips/NU15-30.txt", Ngram.TRIGRAM, 1000);
            List<NgramKL> klist = klpost.calculateKL("/Users/iguy/Pro/Papers/Trolls/Data/Trolltext/Experiment1/Categories/Answers/Bea/abuse.txt", "/Users/iguy/Pro/Papers/Trolls/Data/Trolltext/Experiment1/Categories/Answers/Bea/troll.txt",
                            Ngram.TRIGRAM	, 1000);
//        	List<NgramKL> klist = klpost.calculateKL("/Users/iguy/Pro/Papers/Trolls/Data/Trolltext/Experiment1/KLSimple/Content/abuse.txt", "/Users/iguy/Pro/Papers/Trolls/Data/Trolltext/Experiment1/KLSimple/Content/troll.txt",
//                  Ngram.UNIGRAM		, 1000);
            for (NgramKL nkl : klist) {
                System.out.println(
                                nkl + "\t" + (double)klpost.getBase(nkl.getNgram())/klpost.getBasePosts() + "\t" + (double)klpost.getTarget(nkl.getNgram())/klpost.getTargetPosts());
                               
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}


