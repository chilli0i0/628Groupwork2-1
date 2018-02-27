rm(list = ls())

# Read the lines from the SentiWordNet Lexicon
senti = readLines("/Users/yilixia/Downloads/SentiWordNet_3.0.0.txt")

# colnames(senti) = c("POS",	"ID",	"PosScore",	"NegScore",	"SynsetTerms")

# select words with sentiment scores not equal to 0
new_senti = data.frame()
for(i in 1:length(senti)){
  tmp_senti = strsplit(senti[i],"\t")
  if((tmp_senti[[1]][3] != 0) | (tmp_senti[[1]][4] != 0)){
    new_senti = rbind(new_senti, data.frame(t(tmp_senti[[1]][5:3])))
  }
}

colnames(new_senti) =  c("Word", "Neg", "Pos")

# Expand the dataframe so that each row only contains one word
final_senti = data.frame()
for(i in 1:dim(new_senti)[1]){
  tmp_senti = regmatches(new_senti[i,1], 
                         gregexpr(pattern = "([A-z]+|([A-z]+-[A-z]+))", new_senti[i,1]))
  for(j in 1:length(tmp_senti[[1]])){
    final_senti = rbind(final_senti, data.frame(c(Word = tmp_senti[[1]][j], new_senti[i,2:3])))
  }
  print(i)
}

grep(pattern = "([A-z]+)#\\d", new_senti[5,1])
tmp_senti = regmatches(new_senti[5,1], gregexpr(pattern = "([A-z]+|([A-z]+-[A-z]+))", new_senti[5,1]))




