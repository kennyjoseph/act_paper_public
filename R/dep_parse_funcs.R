remove_less_than_cutoff <- function(data,cutoff){
  while(T){
    prev_nrow <- nrow(data)
    print(paste("prev: ", prev_nrow))
    
    verb_freq <- data.frame(table(data$partialB))
    verb_freq <- verb_freq[verb_freq$Freq > cutoff,]
    verb_freq$Var1 <- as.character(verb_freq$Var1)
    
    data <- data[data$partialB %in% verb_freq$Var1,]
    
    noun_freq <- data.frame(table(c(tolower(data$partialO),tolower(data$partialA))))
    noun_freq <- noun_freq[noun_freq$Freq > cutoff,]
    noun_freq$Var1 <- as.character(noun_freq$Var1)
    
    data <- data[tolower(data$partialA) %in% noun_freq$Var1 &
                         tolower(data$partialO) %in% noun_freq$Var1,]
    
    post_nrow <- nrow(data)
    print(paste("post: ", post_nrow))
    
    if(prev_nrow == post_nrow){
      print("here")
      break
    }
  }
  return(data)
}


tmp <- function(){
  ##okay, now that we've corrected those, I'm going to look over the results. 
  ##There's definitely some errors, but really the only bad ones are the things that are verbs
  ##Given that I eventually want to extend the theory to model things beyond identities and 
  ##behaviors, I think I'm going to leave this stuff in for now.
  all_identities <- sort(unique(tolower(c(d_final$partialA,d_final$partialO))))
  write.table(all_identities, "python/data/all_identities.txt", row.names=F,col.names=F, quote=F)
  
  ##These look pretty good. not going to make any changes here
  all_behaviors <- sort(unique(d_final$partialB))
  write.table(all_behaviors, "python/data/all_behaviors.txt", row.names=F,col.names=F, quote=F)
  
  ##lets check out duplicates in the data
  d_final$full <- paste(d_final$partialA,d_final$partialB,d_final$partialO)
  duplicate_analysis <- d_final[,nrow(.SD),by="full"]
  duplicate_analysis <- orderBy(~-V1, duplicate_analysis)
  
  d_final_with_counts <- merge(d_final, duplicate_analysis,by="full")
  d_final_with_counts <- unique(d_final_with_counts)
  d_final_with_counts$full <- NULL
  
  
  
  
  d1[( (tolower(partialA) %in% identities) &  (tolower(partialO) %in% identities) & !(tolower(partialB) %in% behaviors)) |
       (!(tolower(partialA) %in% identities) &  (tolower(partialO) %in% identities) & (tolower(partialB) %in% behaviors)) |
       ( (tolower(partialA) %in% identities) & !(tolower(partialO) %in% identities) & (tolower(partialB) %in% behaviors)) ,]
  
  
  all_vals <- d1[tolower(partialA) %in% identities &  tolower(partialO) %in% identities & tolower(partialB) %in% behaviors,]
  all_vals_partials <- all_vals[,c("partialA","partialB","partialO"),with=F]
  all_vals_partials$partialA <- tolower(all_vals_partials$partialA)
  all_vals_partials$partialB <- tolower(all_vals_partials$partialB)
  all_vals_partials$partialO <- tolower(all_vals_partials$partialO)
  all_vals_partials$id <- c(1:nrow(all_vals_partials))
  deflection_of_events <- all_vals_partials[,compute_deflection_single(coeff_matrix,coeff_list,summarised_dat,.SD,get_epa_for_abo_terms_from_original_data),by=c("id")]
  all_vals_partials <- merge(all_vals_partials,deflection_of_events,by="id")
  ggplot(all_vals_partials,aes(V1)) + geom_histogram()
  
  
  net <- rbind(d_final_with_counts[,c("partialA","partialB"),with=F],d_final_with_counts[,c("partialO","partialB"),with=F],d_final_with_counts[,c("partialA","partialO"),with=F], use.names=F )
  
  all_identities <- sort(unique(tolower(c(d_final$partialA,d_final$partialO))))
  all_identities_old <- fread("python/data/all_identities.txt",header=F)
  all_identities_cleaned <- fread("python/data/all_identities_cleaned.txt",header=F)
  diff_set <- setdiff(all_identities, all_identities_old$V1)
  write.table(diff_set,"python/data/diff_identities.txt",row.names=F,col.names=F,quote=F)
  
  all_behaviors <- sort(unique(tolower(d_final$partialB))) 
  all_behaviors_old <- fread("python/data/all_behaviors.txt",header=F)
  all_behaviors_cleaned <- fread("python/data/all_behaviors_cleaned.txt",header=F)
  diff_set <- setdiff(all_behaviors, all_behaviors_old$V1)
  write.table(diff_set,"python/data/diff_behaviors.txt",row.names=F,col.names=F,quote=F)
  
}