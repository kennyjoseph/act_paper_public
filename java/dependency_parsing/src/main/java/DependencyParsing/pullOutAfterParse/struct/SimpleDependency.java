package DependencyParsing.pullOutAfterParse.struct;

import edu.stanford.nlp.util.StringUtils;

import java.util.Arrays;
import java.util.Set;

/**
 * Created by kjoseph on 10/2/14.
 */
public class SimpleDependency {

    public String docId;
    public String sentenceId;
    public String fullGoverningTerm = "";
    public String initialGoverningTerm = "";
    public String fullRelationshipTerm = "";
    public String initialRelationshipTerm = "";
    public String fullChildTerm = "";
    public String initialChildTerm = "";

    public SimpleDependency(String docString, String sentenceId,
                            String fullFromToken, String fullVerb, String fullToToken,
                            String partialFromToken, String partialVerb, String partialToToken) {
        docId = docString;
        this.sentenceId = sentenceId;
        fullGoverningTerm = fullFromToken;
        fullRelationshipTerm = fullVerb;
        fullChildTerm = fullToToken;
        initialGoverningTerm = partialFromToken;
        initialRelationshipTerm = partialVerb;
        initialChildTerm = partialToToken;

    }

    public boolean foundIn(Set<String> identities, Set<String> behaviors) {
        return true; //(identities.contains(fullGoverningTerm) | identities.contains(initialGoverningTerm) |
                //identities.contains(fullChildTerm) | identities.contains(initialChildTerm)) |
                 //   (behaviors.contains(fullRelationshipTerm) | behaviors.contains(initialRelationshipTerm));
    }

    public String getString() {
        return
            StringUtils.join(Arrays.asList(
                    docId,
                    sentenceId,
                    fullGoverningTerm,
                    fullRelationshipTerm,
                    fullChildTerm,
                    initialGoverningTerm,
                    initialRelationshipTerm,
                    initialChildTerm), "\t");
    }
}
