package DependencyParsing.pullOutAfterParse;

import DependencyParsing.pullOutAfterParse.struct.Sentence;
import DependencyParsing.pullOutAfterParse.struct.SimpleDependency;
import edu.stanford.nlp.util.StringUtils;
import org.apache.commons.io.FilenameUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Created by kjoseph on 10/2/14.
 */
public class PullOutDependencyResults implements Runnable {

    String mFileName;
    String mOutputDirectory;
    Set<String> mBehaviors;
    Set<String> mIdentities;

    @Override
    public void run(){
        System.out.println(Thread.currentThread().getName()+" " + mFileName + " Start, output to: "+mOutputDirectory);

        try {
            BufferedReader reader = new BufferedReader(new FileReader(mFileName));
            String line ="";
            int sentenceId = 0;
            String docId = FilenameUtils.getBaseName(mFileName);

            List<SimpleDependency> simpleDependencies = new ArrayList<>();
            while((line = reader.readLine()) != null) {
                Sentence s = new Sentence(docId, sentenceId, line, true);
                sentenceId++;
                for(SimpleDependency sd : s.getSimpleDependencies()){
                    if(sd.foundIn(mIdentities,mBehaviors)) {
                        simpleDependencies.add(sd);
                    }
                }
            }

            if(!simpleDependencies.isEmpty()){
                BufferedWriter writer = new BufferedWriter(new FileWriter(mOutputDirectory+"/"+docId));
                for(SimpleDependency sd : simpleDependencies){
                    writer.write(sd.getString()+ "\n");
                }
                writer.close();
            }


        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(Thread.currentThread().getName() + " " + mFileName + " End.");
    }
    public PullOutDependencyResults(String filename,
                                    String outputDirectory,
                                    Set<String> behaviors,
                                    Set<String> identities){
        mFileName = filename;
        mOutputDirectory = outputDirectory;
        mBehaviors = behaviors;
        mIdentities = identities;
    }
}
