package DependencyParsing.pullOutAfterParse.struct;

import com.google.common.collect.*;
import com.google.common.primitives.Ints;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import util.U;

import java.io.IOException;
import java.io.StringReader;
import java.util.*;

/**
 * Created by kjoseph on 10/1/14.
 */
public class Sentence {

    public static GrammaticalStructureFactory gsf;
    static {
        gsf = new PennTreebankLanguagePack().grammaticalStructureFactory();
    }
    public static TreeFactory tree_factory;

    static { tree_factory = new LabeledScoredTreeFactory();

    }

    private Tree tree;
    public Tree getTree(){
        return tree;
    }

    public String docidStr;
    public String sentidStr;
    public int sentid;
    public String jsentStr;
    public boolean hasParse = false;


    public List<Dep> dependencies = new ArrayList<>();
    public List<Token> tokens = new ArrayList<>();
    public List<Mention> mentions = new ArrayList<>();

    public List<Dep> nsubjDependencies = new ArrayList<>();
    // dependency indexes
    public Multimap<Integer,Dep> child2govs;
    public Multimap<Integer,Dep> gov2childs;
    private Map<Pair<Integer,Integer>, String> childgov2rel;

    public boolean isSilent;


    public int T() { return tokens.size(); }

    public List<String> stringTokens() {
        List<String> toks = Lists.newArrayList();
        for (int t=0; t < T(); t++)
            toks.add(tokens.get(t).word);
        return toks;
    }

    public String spacesepText() {
        return StringUtils.join(stringTokens());
    }

    public String pos(int t) { return tokens.get(t).posTag; }
    public String lemma(int t) {return tokens.get(t).lemma; }
    public String word(int t) {return tokens.get(t).word; }

    public void indexDependencies() {
        child2govs = HashMultimap.create();
        gov2childs = HashMultimap.create();
        childgov2rel = Maps.newHashMap();


        for (Dep dep : dependencies) {
            child2govs.put(dep.child, dep);
            gov2childs.put(dep.gov, dep);
            childgov2rel.put(U.pair(dep.child, dep.gov), dep.rel);
        }
        child2govs = ImmutableMultimap.copyOf(child2govs);
        gov2childs = ImmutableMultimap.copyOf(gov2childs);
        childgov2rel = ImmutableMap.copyOf(childgov2rel);

    }

    public String deprel(int child, int gov) {
        String r = childgov2rel.get(U.pair(child,gov));
        return r;
    }

    public Sentence(String docId, int sentenceId, String line, boolean silent){

        isSilent = silent;
        TreeReader treeReader = new PennTreeReader(new StringReader(line), tree_factory);
        Tree inputTree = null;
        try {
            inputTree = treeReader.readTree();
        } catch (IOException e) {
            e.printStackTrace();
        }

        tree = inputTree;
        docidStr =docId;
        sentid = sentenceId;
        sentidStr = "" + sentenceId;
        hasParse = true;

        GrammaticalStructure grammaticalStructure = gsf.newGrammaticalStructure(inputTree);

        Morphology morph = new Morphology();
        ArrayList<TaggedWord> words = inputTree.taggedYield();
        for(TaggedWord tw: words){

            Token tok = new Token();
            tok.word = tw.word();
            tok.posTag = tw.tag();
            tok.nerTag = "NONE";
            tok.lemma = morph.lemma(tw.word(),tw.tag(),true);
            tokens.add(tok);
        }
        if(!isSilent) {
            System.out.println("\n\n" + line + "\n");
        }
        for(TypedDependency td : grammaticalStructure.typedDependenciesCCprocessed()){
            if(!isSilent) {
                System.out.println("\t" + td + "   " + td.gov().index() + "    " + td.dep().index());
            }
            Dep dep = new Dep();
            dep.rel   = td.reln().getShortName();
            dep.child = td.dep().index();
            dep.gov   = td.gov().index();
            dependencies.add(dep);
            if(dep.rel.equals("nsubj")){
                nsubjDependencies.add(dep);
            }
        }

        indexDependencies();

    }

    public boolean isNominal(int t) {
        Token tok = tokens.get(t);
        String rough = roughPOS(tok.posTag);
        return rough.equals("NOUN")
                || rough.equals("PRON");
    }
    public String roughPOS(String posTag) {
        String c = PTB_TO_COARSE.get(posTag);
        if (c==null) return posTag;
        return c;
    }
    public static String roughPOSclean(String posTag) {
        String c = PTB_TO_COARSE.get(posTag);
        if (c==null) c = posTag;
        if (c.equals(".")) c = "PUNCT";
        return c.toLowerCase();
    }

    private int getHeadToken(int startToken){
        //Map<Integer,Integer> tokensMaximalGov = Maps.newHashMap();
        int head = startToken;
        // BF traversal to find upward paths
        // the queue will only have things within the span.
        Queue<Integer> queue = Queues.newArrayDeque();
        queue.add(startToken);
        Set<Integer> allSeen = Sets.newHashSet();
        while (!queue.isEmpty()) {
            int s = queue.remove();
            for (Dep dep : gov2childs.get(s)) {
                if ((dep.rel.equals("amod") || dep.rel.equals("nn")) &&
                        isNominal(dep.child)) {
                    int r = dep.child;
                    if(r < head){
                        head = r;
                    }
                    queue.add(r);
                    allSeen.add(r);
                }
            }
        }
        return head;
    }

    private Pair<Integer,Integer> getHeadVerbToken(int startToken){
        int head = startToken;
        int tail = startToken;
        // BF traversal to find upward paths
        // the queue will only have things within the span.
        Queue<Integer> queue = Queues.newArrayDeque();
        queue.add(startToken);
        Set<Integer> allSeen = Sets.newHashSet();
        while (!queue.isEmpty()) {
            int s = queue.remove();
            for (Dep dep : gov2childs.get(s)) {
                if (/*dep.rel.equals("xcomp") ||*/ dep.rel.equals("aux")) {
                    int r = dep.child;
                    if(r < head){
                        head = r;
                    }
                    if(r > tail){
                        tail = r;
                    }
                    queue.add(r);
                    allSeen.add(r);
                }
            }
        }
        return new Pair<Integer,Integer>(head,tail);
    }


    private boolean isNoiseToken(String token){
        return token.equals("i") ||
               token.equals("we") ||
               token.equals("they") ||
               token.equals("you") ||
                token.equals("it") ||
                token.equals("he") ||
                token.equals("she") ||
                token.equals("them") ||
               token.equals("us") ;
    }

    public List<SimpleDependency> getSimpleDependencies(){
        List<SimpleDependency> simpleDependencies = Lists.newArrayList();

        List<String> lemmaStrings = new ArrayList<>();
        for(Token t : tokens){
            lemmaStrings.add(t.lemma);
        }

        for( Dep nsubjDependency : nsubjDependencies){
            if(!isNominal(nsubjDependency.child-1)){
                continue;

            }
            String fromToken = StringUtils.join(lemmaStrings.subList(
                                                        getHeadToken(nsubjDependency.child)-1,
                                                        nsubjDependency.child)).toLowerCase();

            if(isNoiseToken(fromToken)){
                continue;
            }
            Pair<Integer,Integer> headTailVerb = getHeadVerbToken(nsubjDependency.gov);

            String fullVerb = StringUtils.join(lemmaStrings.subList(
                    headTailVerb.first - 1,
                    headTailVerb.second)).toLowerCase();

            for(Dep childDep : gov2childs.get(nsubjDependency.gov)){
                if(childDep.rel.equals("dobj")) {
                    String toToken = StringUtils.join(lemmaStrings.subList(
                            getHeadToken(childDep.child) - 1,
                            childDep.child)).toLowerCase();
                    if(isNoiseToken(toToken) || !isNominal(childDep.child-1)){
                        continue;
                    }
                    if(!isSilent) {
                        System.out.println(docidStr + " " + sentidStr + " " +
                                fromToken + " " + fullVerb + " " + toToken);
                    }

                    simpleDependencies.add(
                            new SimpleDependency(docidStr, sentidStr,
                                    fromToken,fullVerb,toToken,
                                    tokens.get(nsubjDependency.child-1).lemma,
                                    tokens.get(nsubjDependency.gov-1).lemma,
                                    tokens.get(childDep.child-1).lemma));
                }
            }

        }

        return simpleDependencies;
    }

    public String cleanNER(String tag) {
        return tag.replace("PERSON", "PER").replace("ORGANIZATION","ORG").replace("LOCATION","LOC");
    }

    public Collection<Dep> sorted_gov2childs(int gov) {
        Ordering<Dep> o = new Ordering<Dep>() {
            @Override
            public int compare(Dep arg0, Dep arg1) {
                return Ints.compare(arg0.child, arg1.child);
            }
        };
        return o.sortedCopy(gov2childs.get(gov));
    }

    public String govCentricView() {
        StringBuilder sb = new StringBuilder();
        for (int gov=0; gov < T(); gov++) {
//			String s = String.format("%3d %-15s  ", gov, tokens.get(gov).word);
            Token tok = tokens.get(gov);
            String s = String.format("%3d %-15s %-4s %-4s ", gov, tok.word, tok.posTag, cleanNER(tok.nerTag));
            if (tok.ssTag != null) {
                s += U.sf("%-18s ", tok.ssTag);
            }
            sb.append(s);

            List<String> strs = new ArrayList<String>();
            for (Dep dep : sorted_gov2childs(gov)) {
                strs.add(String.format("<-%s %s", dep.rel, wt(dep.child)));
            }
            if (strs.size() > 0) {
                sb.append(strs.get(0));
            }
            sb.append("\n");
            for (int c=1; c < strs.size(); c++) {
                sb.append(StringUtils.pad("", s.length()));
                sb.append(strs.get(c));
                sb.append("\n");
            }
        }
        return sb.toString();
    }
    private String wt(int t) {
        return String.format("%s:%d", tokens.get(t).word, t);
    }

    public String toString() {
        String s = "";
        s += govCentricView();
        for (Mention m : mentions) {
            s += U.sf("MENT\t%s\n", m);
        }
        return s;
    }

    /** http://universal-pos-tags.googlecode.com/svn/trunk/en-ptb.map **/
    public static ImmutableMap<String,String> PTB_TO_COARSE;
    /** This one I just made up, vaguely like the Twitter tags **/
    public static ImmutableMap<String,String> COARSE_TO_SHORT;

    static {

        COARSE_TO_SHORT = new ImmutableMap.Builder<String,String>()
                .put("ADJ","J")
                .put("ADP","P")
                .put("ADV","R")
                .put("CONJ","C")
                .put("DET","D")
                .put("NOUN","N")
                .put("NUM","NUM")
                .put("PRON","O")
                .put("PRT","T")
                .put("VERB","V")
                .put(".",".").put("X","X")
                .build();
        PTB_TO_COARSE = new ImmutableMap.Builder<String,String>()
                .put("!",".")
                .put("#",".")
                .put("$",".")
                .put("''",".")
                .put("(",".")
                .put(")",".")
                .put(",",".")
                .put("-LRB-",".")
                .put("-RRB-",".")
                .put(".",".")
                .put(":",".")
                .put("?",".")
                .put("``",".")
                .put("JJ","ADJ")
                .put("JJR","ADJ")
                .put("JJRJR","ADJ")
                .put("JJS","ADJ")
                .put("JJ|RB","ADJ")
                .put("JJ|VBG","ADJ")
                .put("IN","ADP")
                .put("IN|RP","ADP")
                .put("RB","ADV")
                .put("RBR","ADV")
                .put("RBS","ADV")
                .put("RB|RP","ADV")
                .put("RB|VBG","ADV")
                .put("WRB","ADV")
                .put("CC","CONJ")
                .put("DT","DET")
                .put("EX","DET")
                .put("PDT","DET")
                .put("WDT","DET")
                .put("NN","NOUN")
                .put("NNP","NOUN")
                .put("NNPS","NOUN")
                .put("NNS","NOUN")
                .put("NN|NNS","NOUN")
                .put("NN|SYM","NOUN")
                .put("NN|VBG","NOUN")
                .put("NP","NOUN")
                .put("CD","NUM")
                .put("PRP","PRON")
                .put("PRP$","PRON")
                .put("PRP|VBP","PRON")
                .put("WP","PRON")
                .put("WP$","PRON")
                .put("POS","PRT")
                .put("PRT","PRT")
                .put("RP","PRT")
                .put("TO","PRT")
                .put("MD","VERB")
                .put("VB","VERB")
                .put("VBD","VERB")
                .put("VBD|VBN","VERB")
                .put("VBG","VERB")
                .put("VBG|NN","VERB")
                .put("VBN","VERB")
                .put("VBP","VERB")
                .put("VBP|TO","VERB")
                .put("VBZ","VERB")
                .put("VP","VERB")
                .put("CD|RB","X")
                .put("FW","X")
                .put("LS","X")
                .put("RN","X")
                .put("SYM","X")
                .put("UH","X")
                .put("WH","X")
                .build();
    }

}
