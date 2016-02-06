import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;

import org.apache.commons.io.FileUtils;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class Preprocess {
    
    public static String encoding = "UTF-8";
    
    public static void main(String[] args) {
        
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma");
        props.put("tokenize.options", "americanize=true,strictTreebank3=true");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        
        String prefix = "../data/";
        String outdir = "../data/";
        
        //SICK data - similarity
        ArrayList<String> data = getSICKdata(prefix+"SICK_test_annotated.txt", pipeline);
        writeFile(outdir+"sicktest",data);
        data = getSICKdata(prefix+"SICK_train.txt", pipeline);
        writeFile(outdir+"sicktrain",data);
        data = getSICKdata(prefix+"SICK_trial.txt", pipeline);
        writeFile(outdir+"sickdev",data);
        
        //SICK data - entailment
        data = getSICKdataEnt(prefix+"SICK_test_annotated.txt", pipeline);
        writeFile(outdir+"sicktest-ent",data);
        data = getSICKdataEnt(prefix+"SICK_train.txt", pipeline);
        writeFile(outdir+"sicktrain-ent",data);
        data = getSICKdataEnt(prefix+"SICK_trial.txt", pipeline);
        writeFile(outdir+"sickdev-ent",data);
    }
    
    private static void writeFile(String fout, ArrayList<String> data) {
        PrintWriter writer;
        try {
            writer = new PrintWriter(fout, encoding);
            for(String d: data)
                writer.println(d);
            writer.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            e.printStackTrace();
        }
    }
    
    private static ArrayList<String> getSICKdata(String fname,
                                                 StanfordCoreNLP pipeline) {
        ArrayList<String> data = new ArrayList<String>();
        
        try {
            List<String> lines = FileUtils.readLines(new File(fname), encoding);
            
            for(int i=1; i < lines.size(); i++) {
                String[] arr = lines.get(i).split("\\t");
                String t1 = getTokenString(arr[1], pipeline);
                String t2 = getTokenString(arr[2], pipeline);
                String score = arr[3];
                String out = t1+"\t"+t2+"\t"+score;
                data.add(out);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        return data;
    }
    
    private static ArrayList<String> getSICKdataEnt(String fname, StanfordCoreNLP pipeline) {
        ArrayList<String> data = new ArrayList<String>();
        
        try {
            List<String> lines = FileUtils.readLines(new File(fname), encoding);
            
            for(int i=1; i < lines.size(); i++) {
                String[] arr = lines.get(i).split("\\t");
                String t1 = getTokenString(arr[1], pipeline);
                String t2 = getTokenString(arr[2], pipeline);
                String score = arr[4];
                String out = t1+"\t"+t2+"\t"+score;
                data.add(out);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        return data;
    }
    
    public static String getTokenString(String s, StanfordCoreNLP pipeline) {
        ArrayList<String> tok1 = getTokens(s,pipeline);
        
        String st = "";
        for(String t: tok1) {
            s = t.toLowerCase();
            if(s.equals("-lsb-"))
                s = "[";
            if(s.equals("-rsb-"))
                s = "]";
            if(s.equals("-lrb-"))
                s = "(";
            if(s.equals("-rrb-"))
                s = ")";
            
            st += s+" ";
        }
        
        return st;
    }
    
    public static ArrayList<String> getTokens(String s, StanfordCoreNLP pipeline) {
        Annotation document = new Annotation(s);
        pipeline.annotate(document);
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        ArrayList<String> tokens = new ArrayList<String>();
        
        for(CoreMap sentence: sentences) {
            
            List<CoreLabel> lis = sentence.get(TokensAnnotation.class);
            int t=0;
            for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
                String word = token.get(TextAnnotation.class);
                tokens.add(word);
            }
        }
        
        return tokens;
    }
}
