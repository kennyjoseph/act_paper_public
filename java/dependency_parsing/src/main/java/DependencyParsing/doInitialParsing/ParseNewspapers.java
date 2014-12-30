package DependencyParsing.doInitialParsing;

import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.parser.shiftreduce.ShiftReduceParser;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;
import org.apache.commons.io.FilenameUtils;

import java.io.*;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Created by kjoseph on 10/1/14.
 */
public class ParseNewspapers implements Runnable {

    Path mFile;
    String mDate;
    String mArticleNumber;
    MaxentTagger mTagger;
    String mOutputFileName;
    ShiftReduceParser mParser;
    StanfordCoreNLP mCoreNLP;

    protected Thread mRunThread;

    public ParseNewspapers(Path file,
                           MaxentTagger tagger,
                           ShiftReduceParser parser,
                           StanfordCoreNLP coreNLP,
                           String outputDirectory){
        mFile = file;
        mCoreNLP = coreNLP;
        String[] fileSplit = FilenameUtils.separatorsToUnix(file.toString()).split("/");
        mDate = fileSplit[fileSplit.length-2].split("_")[1];
        mArticleNumber = fileSplit[fileSplit.length-1].replace(".txt", "");
        mRunThread = new Thread(this, file.toString());
        mTagger = tagger;
        mParser = parser;
        mOutputFileName = outputDirectory + "/" + mDate + "_" + mArticleNumber;
    }

    @Override
    public void run() {
        System.out.println(Thread.currentThread().getName()+" " + mFile + " Start.");

        //newTry();
        oldTry();

        System.out.println(Thread.currentThread().getName() + " " + mFile + " End.");
    }


    public void oldTry(){
        try {
            DocumentPreprocessor tokenizer = new DocumentPreprocessor(new FileReader(mFile.toFile()));
            BufferedWriter writer = new BufferedWriter(new FileWriter(mOutputFileName));

            for (List<HasWord> sentence : tokenizer) {
                List<TaggedWord> tagged = mTagger.tagSentence(sentence);
                Tree tree = mParser.apply(tagged);

                Tree newTree = new TreeLemmatizer().transformTree(tree);
                if(!tree.equals(newTree)){
                    System.out.println(newTree);
                }

                writer.write(tree.toString()+"\n");
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();

        }
    }
    public void newTry(){
        // create an empty Annotation just with the given text
        Annotation document = new Annotation(IOUtils.slurpFileNoExceptions(mFile.toString()));

        // run all Annotators on this text
        mCoreNLP.annotate(document);

        // these are all the sentences in this document
        // a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

        for(CoreMap sentence: sentences) {
            // traversing the words in the current sentence
            // a CoreLabel is a CoreMap with additional token-specific methods
            for (CoreLabel token: sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                // this is the text of the token
                String word = token.get(CoreAnnotations.TextAnnotation.class);
                // this is the POS tag of the token
                String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                // this is the NER label of the token
                String ne = token.get(CoreAnnotations.NamedEntityTagAnnotation.class);
                String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
            }
            // this is the parse tree of the current sentence
            //Tree tree = sentence.get(TreeAnnotation.class);
            // this is the Stanford dependency graph of the current sentence
            //SemanticGraph dependencies = sentence.get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);
        }

        // This is the coreference link graph
        // Each chain stores a set of mentions that link to each other,
        // along with a method for getting the most representative mention
        // Both sentence and token offsets start at 1!
        //Map<Integer, CorefChain> graph =
        //        document.get(CorefCoreAnnotations.CorefChainAnnotation.class);


    }

    public void start(){
        mRunThread.start();
    }
}
