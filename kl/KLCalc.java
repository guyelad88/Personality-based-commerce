//package trolls;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

// import com.yahoo.yrlhaifa.haifa_utils.ds.IntegerHistogram;
// import com.yahoo.yrlhaifa.haifa_utils.ds.MutableInt;

public class KLCalc {

    enum Ngram {
        UNIGRAM, BIGRAM, TRIGRAM;
    }

    private static final char ngramSep = ' ';
    private static final String ngramBeg = "^";
    private static final String ngramEnd = "&";

    private static final double SMOOTHING_FACTOR = 2.0;
    private int cnt=0;

    IntegerHistogram<String> base = new IntegerHistogram<String>();
    IntegerHistogram<String> target = new IntegerHistogram<String>();
    IntegerHistogram<String> baseUnique = new IntegerHistogram<String>();
    IntegerHistogram<String> targetUnique = new IntegerHistogram<String>();

    public List<NgramKL> calculateKL(String baseFileName, String targetFileName, Ngram type, int topK)
                    throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(baseFileName));
        String line;
        while ((line = reader.readLine()) != null) {        	
            collectNgrams(line, type, base, baseUnique);
            cnt++;
            if (cnt % 100000 ==0) {
            	System.out.println("cnt: " + cnt);
            }
        }
        reader.close();
        System.out.println("Finished based");
        reader = new BufferedReader(new FileReader(targetFileName));
        while ((line = reader.readLine()) != null) {
            collectNgrams(line, type, target, targetUnique);
            cnt++;
            if (cnt % 100000 ==0) {
            	System.out.println("cnt: " + cnt);
            }
        }
        double kl, p, pbase;
        // use Java util's topNHeap in case it is too heavy to hold everything in memory
        ArrayList<NgramKL> nklist = new ArrayList<NgramKL>();
        for (Map.Entry<String, MutableInt> entry : target) {
            p = entry.getValue().doubleValue() / target.total().doubleValue();              // freq tf target - p
            pbase = base.get(entry.getKey()).doubleValue() / base.total().doubleValue();    // freq tf base - q

            if (pbase == 0) {                                                               // smoothing
                pbase = 1.0 / (base.total().doubleValue() * SMOOTHING_FACTOR);
            }

            kl = p * Math.log(p / pbase);                                                   // kl term
            nklist.add(new NgramKL(entry.getKey(), kl));
        }
        Collections.sort(nklist);
        return nklist.subList(0, topK);
    }

    public void collectNgrams(String line, Ngram type, IntegerHistogram<String> histo, IntegerHistogram<String> uniqueHisto) {
        line = line.replaceAll("[^a-zA-Z0-9\\'\\- ]", "").toLowerCase();
        // add later: ():;-
        String[] parts = line.split("\\s+");
        HashSet<String> uniques = new HashSet<String>();
        if (parts.length == 0) {
        	return;
        }
        if (type == Ngram.UNIGRAM) {
            for (String word : parts) {
                histo.inc(word);
                uniques.add(word);
            }
        }
        else if (type == Ngram.BIGRAM) {
            int cnt = 0;
            histo.inc(ngramBeg + parts[0]);
            uniques.add(ngramBeg + parts[0]);
            while (cnt < parts.length - 1) {
                histo.inc(parts[cnt] + ngramSep + parts[cnt + 1]);
                uniques.add(parts[cnt] + ngramSep + parts[cnt + 1]);
                cnt++;
            }
            histo.inc(parts[parts.length - 1] + ngramEnd);
            uniques.add(parts[parts.length - 1] + ngramEnd);
        }
        else if (type == Ngram.TRIGRAM) {
            if (parts.length < 2) {
                return;
            }
            int cnt = 0;
            histo.inc(ngramBeg + parts[0] + ngramSep + parts[1]);
            uniques.add(ngramBeg + parts[0] + ngramSep + parts[1]);
            while (cnt < parts.length - 2) {
                histo.inc(parts[cnt] + ngramSep + parts[cnt + 1] + ngramSep + parts[cnt + 2]);
                uniques.add(parts[cnt] + ngramSep + parts[cnt + 1] + ngramSep + parts[cnt + 2]);
                cnt++;
            }
            histo.inc(parts[parts.length - 1] + ngramSep + parts[parts.length - 2] + ngramEnd);
            uniques.add(parts[parts.length - 1] + ngramSep + parts[parts.length - 2] + ngramEnd);
        }
        for (String s : uniques) {
        	uniqueHisto.inc(s);
        }
        return;
    }

    public void testCollectNgrams() {
    	KLCalc klcalc = new KLCalc();
        IntegerHistogram<String> his = new IntegerHistogram<String>();
        String line = "RB   CMA VB  RB  IN  DT  NN  TO  VB  NNS CC  VB  IN  DT  NN  IN  PRP MD  PRD NN NN NN";
        klcalc.collectNgrams(line, Ngram.UNIGRAM, his, null);
        System.out.println(his.toOrderedString());
    }

    public int getBase(String ngram) {
        return base.get(ngram).intValue();
    }

    public int getTarget(String ngram) {
        return target.get(ngram).intValue();
    }
    
    public int getBaseU(String ngram) {
        return baseUnique.get(ngram).intValue();
    }

    public int getTargetU(String ngram) {
        return targetUnique.get(ngram).intValue();
    }

    public static void main(String[] args) {
    	KLCalc klcalc = new KLCalc();
        // klcalc.testCollectNgrams();
        try {

            // List<NgramKL> klist = klcalc.calculateKL("/Users/idoguy/Pro/Papers/Tips/useful15-30.txt",
            // "/Users/idoguy/Pro/Papers/Tips/NU15-30.txt", Ngram.TRIGRAM, 1000);
            List<NgramKL> klist = klcalc.calculateKL(
                    "/Users/iguy/Pro/Papers/Trolls/Data/Trolltext/Experiment1/KLSimple/Content/all.txt",
                    "/Users/iguy/Pro/Papers/Trolls/Data/Trolltext/Experiment1/KLSimple/Content/troll.txt",
                     Ngram.BIGRAM, 1000);
            for (NgramKL nkl : klist) {
                System.out.println(
                                nkl + "\t" + klcalc.getBase(nkl.getNgram()) + "\t" + klcalc.getTarget(nkl.getNgram())
                                + "\t" + klcalc.getBaseU(nkl.getNgram()) + "\t" + klcalc.getTargetU(nkl.getNgram()));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}


class IntegerHistogram <T>{

    list <T> words;
    list <integer> histogram;

    public void inc(T word) {

        return ngram;
    }

    public int get(T word) {
        return kl;
    }

    public int total() {
        return;
    }

}

class NgramKL implements Comparable<NgramKL> {

    String ngram;
    double kl;

    NgramKL(String ngram, double kl) {
        this.ngram = ngram;
        this.kl = kl;
    }

    public String getNgram() {
        return ngram;
    }

    public double getKL() {
        return kl;
    }

    @Override
    public int compareTo(NgramKL o) {
        if (this.kl > o.getKL()) {
            return -1;
        }
        if (this.kl < o.getKL()) {
            return 1;
        }
        return 0;
    }

    public String toString() {
        return ngram + "\t" + kl;
    }
}


