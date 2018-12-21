#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
if(length(args)!=2){
  stop("looking for 2 argument, path to input file and name of output file")
}

file = args[1]
name = args[2]

require("flacco")
require("RANN")
require("e1071")
##file = "/home/volz/svn/gbea/code-experiments/build/c/exdata/mario.RData"
load(file)

#names = c("basic.dim", "basic.observations", "basic.lower_min", "basic.lower_max", "basic.upper_min", , "basic.upper_max", "basic.objective_min", "basic.objective_max", "basic.blocks_min", "basic.blocks_max", , "basic.cells_total", "basic.cells_filled", "basic.minimize_fun", "basic.costs_fun_evals", "basic.costs_runtime", , "nbc.nn_nb.sd_ratio", "nbc.nn_nb.mean_ratio", "nbc.nn_nb.cor", "nbc.dist_ratio.coeff_var", "nbc.nb_fitness.cor", , "nbc.costs_fun_evals", "nbc.costs_runtime", "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10", , "disp.ratio_mean_25", "disp.ratio_median_02", "disp.ratio_median_05", "disp.ratio_median_10", "disp.ratio_median_25", , "disp.diff_mean_02", "disp.diff_mean_05", "disp.diff_mean_10", "disp.diff_mean_25", "disp.diff_median_02", , "disp.diff_median_05", "disp.diff_median_10", "disp.diff_median_25", "disp.costs_fun_evals", "disp.costs_runtime", , "ic.h.max", "ic.eps.s", "ic.eps.max", "ic.eps.ratio", "ic.m0", , "ic.costs_fun_evals", "ic.costs_runtime", "pca.expl_var.cov_x", "pca.expl_var.cor_x", "pca.expl_var.cov_init", , "pca.expl_var.cor_init", "pca.expl_var_PC1.cov_x", "pca.expl_var_PC1.cor_x", "pca.expl_var_PC1.cov_init", "pca.expl_var_PC1.cor_init", , "pca.costs_fun_evals", "pca.costs_runtime", "ela_distr.skewness", "ela_distr.kurtosis", "ela_distr.number_of_peaks", , "ela_distr.costs_fun_evals", "ela_distr.costs_runtime", "ela_meta.lin_simple.adj_r2", "ela_meta.lin_simple.intercept", "ela_meta.lin_simple.coef.min", , "ela_meta.lin_simple.coef.max", "ela_meta.lin_simple.coef.max_by_min", "ela_meta.lin_w_interact.adj_r2", "ela_meta.quad_simple.adj_r2", "ela_meta.quad_simple.cond", , "ela_meta.quad_w_interact.adj_r2", "ela_meta.costs_fun_evals", "ela_meta.costs_runtime")
tries = unique(df[,c("dim", "fun", "inst")])
result=NULL
for(i in 1:nrow(tries)){
  data = df[df$dim==tries$dim[i] & df$fun==tries$fun[i] & df$inst==tries$inst[i],]
  X = matrix(unlist(data$loc), ncol=tries$dim[i], byrow = TRUE)
  y = data$fitness
  feat.object = createFeatureObject(X=X, y=y)
  ctrl = list(subset=c("basic", "nbc", "disp", "ic", "pca", "ela_distr"))
  #compute ela:  basic, nbc, disp, ic, pca, ela_distr,  ela_meta,
  features=NULL
  try(expr=(features=calculateFeatures(feat.object, control=ctrl)), silent=T)
  if(is.null(features)){
    next
  }
  features$dim = tries$dim[i]
  features$fun = tries$fun[i]
  features$inst = tries$inst[i]
  print(as.data.frame(features))
  result = rbind(result, as.data.frame(features))
}

save(result,file=name)

