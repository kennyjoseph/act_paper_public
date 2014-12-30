library(doBy)
library(stringr)
library(stringdist)

##CHANGE THIS
top_dir <- "~/git/thesis/act_paper/"
source(paste0(top_dir,"R/dep_parse_funcs.R"))

##Starting to experiment with the full set of dependency parses, but still
##need future work on how best to extract identities
#d1 <- fread("~/Desktop/tmp_arab/parsed_all.tsv",sep="\t",header=F,stringsAsFactors=F)
d1 <- fread(paste0(top_dir,"/dep_parse/dep_parse_all.tsv"),sep="\t",header=F,stringsAsFactors=F)

##get date/time information
S1 <- d1[, list(X = unlist(strsplit(V1, "[_-]"))), by = seq_len(nrow(d1))]
S1[, Time := sequence(.N), by = seq_len]
d2 <- dcast.data.table(S1, seq_len ~ Time, value.var="X")
setnames(d2, c("s","year","month","day","article"))
d1 <- cbind(d1,d2[,c("year","month","day","article"),with=F])
d1$V1 <- NULL
d1$date <- as.Date(paste(d1$month,d1$day,d1$year,sep="-"),"%m-%d-%Y")

setnames(d1,c("V2","V3","V4","V5","V6","V7","V8"), 
         c("SentenceID","fullA","fullB","fullO","partialA","partialB","partialO"))

##for now, lets make sure there is at least one act term in each 
identities <- summarised_dat[type=="identity",]$name
behaviors <- summarised_dat[type=="behavior",]$name

d_clean <- d1[str_length(partialA) > 3 & 
                     str_length(partialB) > 2 & 
                     str_length(partialO) > 3,]  

##for now, get rid of negations ... not sure how theory handles that
d_clean[,negate_value:=ifelse(grepl(" not ",d_clean$fullB),1,0),]
d_clean <- d_clean[negate_value == 0,]

##instead of shingling, lets just make sure that we only consider 
##a particular event worded in exactly the same way once on each day.  
##Its not quite as scientific, but its probably more useful for what we're doing.
##really, its not clear that we should even theoretically be taking these out,
##but thats for future work.
d_clean <- d_clean[!duplicated.data.frame(d_clean[,c("fullA","fullB","fullO","year","month","day"),with=F])]
d_clean <- d_clean[tolower(partialA) != tolower(partialO),]

###get rid of relations where behavior doesnt appear > CUTOFF times
d_clean <- remove_less_than_cutoff(d_clean,25)

d_clean <- d1[(tolower(partialA) %in% identities | tolower(partialO) %in% identities | partialB %in% behaviors),]

##lowercase everything for the final set
d_final <- d_clean[,c("partialA","partialB","partialO"),with=F]
d_final$partialA <- tolower(d_final$partialA)
d_final$partialB <- tolower(d_final$partialB)
d_final$partialO <- tolower(d_final$partialO)

##do a few cleaning steps here
all_identities <- sort(unique(tolower(c(d_final$partialA,d_final$partialO))))
dist_data <- stringdistmatrix(all_identities,all_identities)
dimnames(dist_data) <- list(1:nrow(dist_data),1:ncol(dist_data))
df <- as.data.table(as.table(dist_data))
df <- df[N > 0 & N < 2,]
df$V1 <- as.integer(df$V1)
df$V2 <- as.integer(df$V2)
dt <- data.table(dist_data)
df$V1A <- all_identities[df$V1]
df$V2A <- all_identities[df$V2]
df <- df[substr(df$V1A,0,1) == substr(df$V2A,0,1),]
df <- df[df$V1 < df$V2,]
##a lot of these mappings look like they might be connected, but the only really obvious
##ones are those that are plurals.  At some point we should deal with this and also deal with full vs. partial, but for now I'm just going to get rid of the plural issue and cases where its 'algeria' vs 'algerian' and 'al-quiada' vs 'al-quaidah'
sub <- rbind(df[paste0(df$V1A,"s") == df$V2A, c("V1A","V2A"),with=F],
             df[paste0(df$V1A,"n") == df$V2A, c("V1A","V2A"),with=F],
             df[paste0(df$V1A,"h") == df$V2A, c("V1A","V2A"),with=F])
sub <- sub[! sub$V2A %in% c("satan","demon","booth","saleh") ]
for(row in 1:nrow(sub)){
  d_final[partialA ==sub[row]$V2A,]$partialA  <- sub[row]$V1A
  d_final[partialO ==sub[row]$V2A,]$partialO  <- sub[row]$V1A
}

all_identities_cleaned <- fread("python/data/all_identities_sparse.txt",header=F)
all_behaviors_cleaned <- fread("python/data/all_behaviors_sparse.txt",header=F)

d_cleaned_final <- d_final[partialA %in% all_identities_cleaned$V1 & partialO %in% all_identities_cleaned$V1,]
d_cleaned_final <- d_cleaned_final[partialB %in% all_behaviors_cleaned$V1]
d_cleaned_final$count <- 1
#predict left-out events 
d_cleaned_final <- remove_less_than_cutoff(d_cleaned_final,25)

##don't rewrite these out, for now
#write.table(unique(d_cleaned_final$partialB),"python/data/final_behaviors.txt",row.names=F,col.names=F,quote=F)
#write.table(unique(c(d_cleaned_final$partialA,d_cleaned_final$partialO)),"python/data/final_identities.txt",row.names=F,col.names=F,quote=F)

library(cvTools)

DATA_DIR <- "python/data"
set.seed(0)

K_FOLD_EVENTS_DIR <- file.path(DATA_DIR,"k_fold_test/")
dir.create(K_FOLD_EVENTS_DIR)
folds <- cvFolds(nrow(d_cleaned_final), 10)

for(i in 1:10){
  dir_path <- file.path(K_FOLD_EVENTS_DIR,paste0("fold_",i))
  dir.create(dir_path)
  
  test_data <- d_cleaned_final[folds$which == i]
  train_data <- d_cleaned_final[folds$which != i]
  write.table(test_data,
              file.path(dir_path,"test.tsv"),
              row.names=F,col.names=F,sep="\t",quote=F)
  write.table(train_data,
              file.path(dir_path,"train.tsv"),
              row.names=F,col.names=F,sep="\t",quote=F)
  
}



