wd <- "/Users/kjoseph/git/thesis/act_paper/"
library(stringr)
library(reshape2)

get_epa_scores_from_raw_for_user <- function(r){
  df <- data.frame(uid= as.integer(r["uid"]), 
                   name="",
                   e=rep(-45,n_ids),
                   p=rep(-46,n_ids),
                   a=rep(-47,n_ids),
                   stringsAsFactors=F)
  for(i in seq(1,n_ids)){
    df[i,]$name <- r[10+i*4]
    df[i,]$e <-    as.double(r[11+i*4])
    df[i,]$p <-    as.double(r[12+i*4])
    df[i,]$a <-    as.double(r[13+i*4])
  }
  return(df)
}

##read in the raw data
d <- read.csv(paste0(wd,"All_Data_A&S+Kelley_USA_only.csv"),stringsAsFactors=F,encoding="utf-8")

##pull out codings
id_names <- names(d)[seq(14,length(names(d))-4,by=4)]
d$uid <- c(1:nrow(d))
n_ids <- length(id_names)
full_dat <- rbindlist(apply(d,1,get_epa_scores_from_raw_for_user))

##mash back up with user info
d_tmp <- d[,c("uid","mon_yr","study","stim_set","skipped","minutes","sex","race","region","marital")]
no_na_full <- full_dat[!is.na(full_dat$e),]
final_full <- merge(no_na_full,d_tmp,by="uid")
final_full$type <- "Identity"
final_full[grepl("m_",final_full$name),]$type <- "Modifier"
final_full[grepl("s_",final_full$name),]$type <- "Setting"
final_full[grepl("b_",final_full$name),]$type <- "Behavior"
dt_final_full <- data.table(final_full)

##summarise male and female values
summarised_dat <- dt_final_full[,
                                list(e=mean(e),e_var=var(e),e_sd=sd(e),
                                     p=mean(p),p_var=var(p),p_sd=sd(p),
                                     a=mean(a),a_var=var(a),a_sd=sd(a),
                                     count=length(e)),by=c("name","type")]
##one or two terms w/ bad text, just ignore
summarised_dat <- summarised_dat[summarised_dat$count > 1,]

summarised_dat$name <- str_split_fixed(summarised_dat$name,"_",2)[,2]

###EDA
##high correlations, except between e and a
ggplot(summarised_dat,aes(e,a,size=mean(e_sd+p_sd)/sqrt(count),color=p)) + geom_point(alpha=.8)  + geom_vline(x=0) + geom_hline(y=0) + scale_color_continuous(low="blue",high='orange')

df <- melt(summarised_dat,id.vars=c(),measure.vars=c("e","p","a"))
ggplot(df, aes(value,color=variable)) + geom_density()
ggplot(melt(summarised_dat,id.vars=c(),measure.vars=c("e_var","p_var","a_var")), aes(value,color=variable))+ geom_density()

##write it out for analyses
summarised_dat$type <- tolower(summarised_dat$type)
#summarised_dat[,e_se:=e_sd/sqrt(count)]
#summarised_dat[,p_se:=p_sd/sqrt(count)]
#summarised_dat[,a_se:=a_sd/sqrt(count)]
write.table(summarised_dat[type=="identity"|type=="behavior",c("name","type","e","e_var","p","p_var","a","a_var"),with=F],paste0(wd,"python/data/act_init_scores_next.tsv"),sep="\t",row.names=F,col.names=F,quote=F)


