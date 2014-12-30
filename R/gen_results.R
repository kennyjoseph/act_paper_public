library(reshape2)
library(dplyr)

full_res <- fread("R/results/full_final_out.tsv")
setnames(full_res, c("fold","name","ntopic","random","p","alpsi","initsd","q","medrank","meanrank","top10","sumll","ndoc"))

##get rid of development fold
full_res <- full_res[fold != 1,]
##get rid of values we didn't end up using
full_res <- full_res[p == 3 | is.na(p)]
full_res <- full_res[alpsi  == 1 | is.na(alpsi)]
full_res <- full_res[q ==1 | is.na(q)]
full_res <- full_res[!(name=='Simple' & random==T)]

full_res[name=="Simple"]$name <- "ACT only"
full_res[name=='Full' & random==T]$name <- "No ACT"

res <- full_res[,as.list(smean.cl.boot(-sumll/ndoc)), by=c("name","ntopic")]

fig1 <- ggplot() 
fig1 <- fig1 + geom_pointrange(data=res[ntopic >0], size=1.2, 
                               aes(x=ntopic, ymin=Lower,ymax=Upper,y=Mean,color=name)) 
fig1 <- fig1 + geom_line(data=res[ntopic >0],size=1.2, aes(x=ntopic,y=Mean,color=name)) 
fig1 <- fig1 + geom_hline(data=res[ntopic==-1],size=1.2, aes(yintercept=Upper,color=name))
fig1 <- fig1 + geom_hline(data=res[ntopic==-1], size=1.2,aes(yintercept=Lower,color=name))
fig1 <- fig1 + ylab("Average Perplexity") + xlab("Number of Topics")
fig1 <- fig1 + scale_color_discrete(guide=guide_legend("Model",title.position='top',title.hjust=.5,label.position='bottom',label.hjust=.5)) 
ggsave("paper/results_1.png",fig1,h=6,w=10,dpi=400)


full <- fread("R/results/best_full_runs.tsv")
bigram <- fread("R/results/bigram_runs.tsv")

full$ind <- 1:nrow(full)
bigram$ind <- 1:nrow(bigram)

setnames(full,"V10","Full")
setnames(full,"V11","Full_Rank")
setnames(bigram,"V10","Bigram")
setnames(bigram,"V11","Bigram_Rank")
merged <- merge(full[,c("ind","V9","Full","Full_Rank"),with=F],
           bigram[,c("ind","V9","Bigram","Bigram_Rank"),with=F],by=c("ind","V9"))

fig2 <- ggplot(merged, aes(Bigram,Full)) + geom_point(alpha=.3) + geom_abline(slope=1,color='red',size=1.2)
fig2 <- fig2 + xlab("Likelihood of Test Point - Bigram Model") + ylab("Likelihood of Test Point- Best Full Model")
ggsave("paper/results_2.png",fig2,h=6,w=6,dpi=400)


cutoffs <- c(.001, .002, .01,.02,.1,.2)
df <- data.frame(cutoff=cutoffs , Full=rep(0,length(cutoffs)), Bigram=rep(0,length(cutoffs)))

for( i in c(1:length(cutoffs))){
  dat <- merged[Full > cutoffs[i] & Bigram > cutoffs[i],]
  df[i,"Full"] <- -sum(log(dat$Full))/nrow(dat)
  df[i,"Bigram"] <- -sum(log(dat$Bigram))/nrow(dat)
}

fig3 <- ggplot(melt(df,id.vars="cutoff"), aes(cutoff,value,color=variable)) + geom_line(size=1.2) + geom_point(size=5)  
fig3 <- fig3 + ylab("Perplexity") + xlab("Cutoff for Model Certainty") 

fig3 <- fig3 + theme(legend.position='bottom') + scale_color_discrete(guide=guide_legend("Model",title.position='top',title.hjust=.5,label.position='bottom',label.hjust=.5,keywidth=4)) 
ggsave("paper/results_3.png",fig3,h=6,w=6,dpi=400)


test_items <- fread("R/results/full_test_set.tsv",header=F)
test_items$ind <- 1:nrow(test_items)
merged <- merge(test_items,merged, by="ind")

arrange(merged[Bigram > Full],-(log(Bigram) - log(Full)))[1:25]
arrange(merged[Full > Bigram],-(log(Full) - log(Bigram)))[1:25]


col_names <- c("iteration",
               "datatype",
               "from_data",
               "entity",
               "epa",
               "topic",
               "n_samples",
               "new_values_mean",
               "new_mu_0",
               "new_sd_0"
)
epa_labeller <- function(var, value){
  value <- as.character(value)
  if (var=="epa") { 
    value[value==0] <- "Evaluative"
    value[value==1]   <- "Potency"
    value[value==2]   <- "Activity"
  }
  return(value)
}


train <- fread("python/data/full_dataset/trained_models/Full_5_False_3_1_0.1_1/output_mu.tsv",header=F)
setnames(train, col_names)
train <- train[iteration == 199]

##things in the ACT dictionaries but not in the data
not_of_interest <- train[,sum(n_samples),by="entity"]
not_of_interest <- not_of_interest[V1 == 0,]$entity

train <- train[!(entity %in% not_of_interest),]
train <- train[n_samples >= 10]
train$entity <- factor(train$entity, 
                       levels=rev(arrange(train[epa==0,(new_mu_0*n_samples)/sum(n_samples),by="entity"],V1)$entity))
train$low <- with(train,ifelse(new_mu_0-1.96*new_sd_0 < -4.3, -4.3, new_mu_0-1.96*new_sd_0))
train$high <- with(train,ifelse(new_mu_0+1.96*new_sd_0 > 4.3, 4.3, new_mu_0+1.96*new_sd_0))


yscale <- scale_y_continuous(limits=c(-4.3,4.3),breaks=c(-4.3,-2.1,0,2.1, 4.3))

fig4 <-  ggplot(train[datatype==1 & from_data==0], 
                aes(entity, new_mu_0,ymin=low,ymax=high, group=factor(topic)))
fig4 <- fig4 + geom_pointrange(position=position_dodge(width=.8)) 
fig4 <- fig4 + xlab("Behaviors") + ylab("") + geom_hline(y=0,color='red')
fig4 <- fig4 + facet_grid(epa~.,labeller=epa_labeller) 
fig4 <- fig4 + theme(axis.text.x=element_text(angle=45,hjust=1,vjust=1),legend.position='bottom') + yscale
ggsave("paper/results_4.png",width=13,h=7.5,dpi=400)

all_identities <- fig4 %+% train[datatype==0 & from_data==0] + xlab("Identities")
ggsave("paper/all_identities.png",width=15,h=7,dpi=400)

pol <- c("democrat","republican")
rel <- c("christian","jews","muslims","islamists","extremist","sunni")
count <- c("palestinian", "israeli","libyans","iraqis","iranians")
of_interest <- rel #c(pol,rel,count)
                    
fig5_data <- train[datatype==0& entity %in% of_interest]

fig5_data$type <- "U.S. Politics"
fig5_data[entity %in% rel]$type <- "Religious"
fig5_data[entity %in% count]$type <- "Nationality"

fig5_data$entity <- factor(fig5_data$entity,levels=of_interest)

fig5 <- fig4 %+% fig5_data + facet_wrap(epa~type,labeller=epa_labeller)
ggsave("paper/results_5.png",w=12,h=6,dpi=400)

d <- fig5_data[epa ==0,list(n=sum(n_samples),t=sum(n_samples > 0)),by='entity']
cor(d$n,d$t)

#color=factor(from_data),
#fig4 <- fig4 + scale_color_manual(values=c("black","#CCCCE0"),labels=c("No","Yes"),
#                                  guide=guide_legend("In ACT Dictionary",
#                                                     label.position='bottom',
#                                                     label.hjust=.5,keywidth=4))