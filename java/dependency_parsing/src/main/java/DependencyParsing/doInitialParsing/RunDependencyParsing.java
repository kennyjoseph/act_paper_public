package DependencyParsing.doInitialParsing;

/**
 * Created by kjoseph on 10/1/14.
 */

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import edu.stanford.nlp.parser.shiftreduce.ShiftReduceParser;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;

/**
 * Demonstrates how to first use the tagger, then use the
 * ShiftReduceParser.  Note that ShiftReduceParser will not work
 * on untagged text.
 *
 * @author John Bauer
 */
public class RunDependencyParsing {
    public static final String modelPath =
            "edu/stanford/nlp/models/srparser/englishSR.ser.gz";
    public static final String taggerPath =
            "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger";

    public static void runParsing(String inputDirectory,
                           final String outputDirectory,
                           int numThreads){
        final MaxentTagger tagger = new MaxentTagger(taggerPath);
        final ShiftReduceParser model = ShiftReduceParser.loadModel(modelPath);
        final ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        try {
            Files.walkFileTree(Paths.get(inputDirectory), new SimpleFileVisitor<Path>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                    System.out.println(file);
                    if(file.toString().contains(".DS_Store")){
                        return FileVisitResult.CONTINUE;
                    }
                    executor.execute(new ParseNewspapers(file, tagger, model,null, outputDirectory));
                    return FileVisitResult.CONTINUE;
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        executor.shutdown();
    }

    public static void main(String[] args) {
        if(args.length != 3){
            System.out.println("Usage: [input_directory] [output_directory] [n_threads]");
            System.exit(-1);
        }
        String inputDirectory = args[0];
        if(!inputDirectory.endsWith("/")){
            inputDirectory+="/";
        }
        String outputDirectory = args[1];
        if(!outputDirectory.endsWith("/")){
            outputDirectory+="/";
        }
        File outputDirFile = new File(outputDirectory);
        if (!outputDirFile.exists()) {
            boolean madeDir = outputDirFile.mkdirs();
            if (madeDir) {
                System.out.println("Output Directory successfully created");
            }
            else {
                System.out.println("Failed to create output directory: " + outputDirectory + ", exiting");
                System.exit(-1);
            }
        } else {
            System.out.println("Overwriting output directory");
        }

        int numThreads = Integer.parseInt(args[2]);
        runParsing(inputDirectory,outputDirectory,numThreads);

    }
}