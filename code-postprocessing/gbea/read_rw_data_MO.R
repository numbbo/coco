#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
if(length(args)!=2){
  stop("looking for 2 arguments, path to input folder and name of output file")
}

## or else just load stringr
require(stringr) 

dir = args[1]
name = args[2]
##dir = "/home/volz/svn/gbea/code-experiments/build/c/exdata/rw-gan-mario"
##name = "rw-gan-mario.RData"

## get file names (required to be in main path of the experiments data folder)
files <- list.files(path=dir, pattern = "\\.txt$",recursive=T) #switch to "\\.tdat" if required

df <- data.frame(evaluation=integer(), f1=double(), f2=double(),loc=list(), dim=integer(), fun=integer(), inst=integer())
for(f in files){
  dim <- as.numeric(str_extract(str_extract(f,"d[0-9]+"),"\\d+")) #extract dimension
  fun <- as.numeric(str_extract(str_extract(f,"f[0-9]+"),"\\d+")) #extract function number
  inst <- as.numeric(str_extract(str_extract(f,"i[0-9]+"),"\\d+")) #extract instance number
  f = paste(dir,f, sep="/")
  res <- readLines(f) #read file
  readdf <- read.table(textConnection(res[4:length(res)]),header=F) #convert lines to table
  loc = readdf[,4:(3+dim)] #x-values
  loc = apply(loc, 1, list)
  readdf <- readdf[,1:3]
  colnames(readdf) = c("evaluation", "f1", "f2")
  readdf$loc = loc
  readdf$dim = dim
  readdf$fun = fun
  readdf$inst = inst
  df <- rbind(df,readdf) #append result
}
save(df,file=name)
