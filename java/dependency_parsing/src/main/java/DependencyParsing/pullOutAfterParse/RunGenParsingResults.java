package DependencyParsing.pullOutAfterParse;

import edu.stanford.nlp.io.IOUtils;

import java.io.*;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by kjoseph on 10/1/14.
 */



public class RunGenParsingResults {

    public static Set<String> constructSetFromFile(String filename){
        //Read in the (tab separated) files that map from the actual ACT terms
        //to the reduced forms we will search for in the text
        Set<String> toReturn = new HashSet<>();

        String[] linesInFile = IOUtils.slurpFileNoExceptions(filename).split("\n");
        for(String line: linesInFile){
            toReturn.add(line.split("\t")[1]);
        }
        return toReturn;
    }

    public static void getParsingResults(String inputDirectory,
                                  final String outputDirectory,
                                  String behaviorNamesFile,
                                  String identityNamesFile,
                                  int numThreads){



        final Set<String> behaviors = constructSetFromFile(behaviorNamesFile);
        final Set<String> identities = constructSetFromFile(identityNamesFile);

        final ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        try {
            Files.walkFileTree(Paths.get(inputDirectory), new SimpleFileVisitor<Path>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                    System.out.println(file);
                    executor.execute(
                            new PullOutDependencyResults(file.toString(), outputDirectory,behaviors,identities));
                    return FileVisitResult.CONTINUE;
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        executor.shutdown();
    }

    public static void main(String[] args) {
        if(args.length != 5){
            System.out.println("Usage: [input_directory] "+
                                "[output_directory] " +
                                "[behavior_file_name] [identities_file_name]"+
                                "[n_threads]");
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

        String behaviorFileName = args[2];
        String identityFileName = args[3];

        int numThreads = Integer.parseInt(args[4]);
        getParsingResults(inputDirectory, outputDirectory, behaviorFileName, identityFileName, numThreads);

    }

}
