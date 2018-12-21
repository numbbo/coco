#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
if(length(args)!=2){
  stop("looking for 2 argument, path to input file and name of output file")
}


desat <- function(cols, sat=0.5) {
  X <- diag(c(1, sat, 1)) %*% rgb2hsv(col2rgb(cols))
  hsv(X[1,], X[2,], X[3,])
}

file = args[1]
name = args[2]

load(file)
#names = c("basic.dim", "basic.observations", "basic.lower_min", "basic.lower_max", "basic.upper_min", , "basic.upper_max", "basic.objective_min", "basic.objective_max", "basic.blocks_min", "basic.blocks_max", , "basic.cells_total", "basic.cells_filled", "basic.minimize_fun", "basic.costs_fun_evals", "basic.costs_runtime", , "nbc.nn_nb.sd_ratio", "nbc.nn_nb.mean_ratio", "nbc.nn_nb.cor", "nbc.dist_ratio.coeff_var", "nbc.nb_fitness.cor", , "nbc.costs_fun_evals", "nbc.costs_runtime", "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10", , "disp.ratio_mean_25", "disp.ratio_median_02", "disp.ratio_median_05", "disp.ratio_median_10", "disp.ratio_median_25", , "disp.diff_mean_02", "disp.diff_mean_05", "disp.diff_mean_10", "disp.diff_mean_25", "disp.diff_median_02", , "disp.diff_median_05", "disp.diff_median_10", "disp.diff_median_25", "disp.costs_fun_evals", "disp.costs_runtime", , "ic.h.max", "ic.eps.s", "ic.eps.max", "ic.eps.ratio", "ic.m0", , "ic.costs_fun_evals", "ic.costs_runtime", "pca.expl_var.cov_x", "pca.expl_var.cor_x", "pca.expl_var.cov_init", , "pca.expl_var.cor_init", "pca.expl_var_PC1.cov_x", "pca.expl_var_PC1.cor_x", "pca.expl_var_PC1.cov_init", "pca.expl_var_PC1.cor_init", , "pca.costs_fun_evals", "pca.costs_runtime", "ela_distr.skewness", "ela_distr.kurtosis", "ela_distr.number_of_peaks", , "ela_distr.costs_fun_evals", "ela_distr.costs_runtime", "ela_meta.lin_simple.adj_r2", "ela_meta.lin_simple.intercept", "ela_meta.lin_simple.coef.min", , "ela_meta.lin_simple.coef.max", "ela_meta.lin_simple.coef.max_by_min", "ela_meta.lin_w_interact.adj_r2", "ela_meta.quad_simple.adj_r2", "ela_meta.quad_simple.cond", , "ela_meta.quad_w_interact.adj_r2", "ela_meta.costs_fun_evals", "ela_meta.costs_runtime")
pdf(name)
tries = unique(df[,c("dim", "fun")])
for(i in 1:nrow(tries)){
  data = df[df$dim==tries$dim[i] & df$fun==tries$fun[i],]
  data= data[data$f1 <1000 & data$f2 <1000,]
  insts = unique(data$inst)
  cols = rainbow(length(insts))
  plot(0,type="n", main=paste("dim", tries$dim[i], "fun", tries$fun[i]),xlab="f1", ylab="f2",
       xlim=c(min(data$f1, na.rm=TRUE), max(data$f1, na.rm=TRUE)), ylim = c(min(data$f2, na.rm=TRUE), max(data$f2, na.rm = TRUE)))
  for(j in 1:length(insts)){
    dt = data[data$inst == insts[j],]
    dt$alpha = (1-dt$evaluation/max(dt$evaluation))*0.9
    for(r in 1:nrow(dt)){
      points(dt$f1[r], dt$f2[r], col=adjustcolor(cols[j],offset = c(dt$alpha[r], dt$alpha[r], dt$alpha[r], 0)), pch=16)
    }
    #points(dt$f1, dt$f2, col=adjustcolor(cols[j],offset = c(0.9, 0.9, 0.9, 0), alpha.f = 0.7), pch=16)
  }
}

dev.off()
