#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
if(length(args)!=2){
  stop("looking for 2 argument, path to input file and name of output file")
}

file = args[1]
name = args[2]

load(file)
#names = c("basic.dim", "basic.observations", "basic.lower_min", "basic.lower_max", "basic.upper_min", , "basic.upper_max", "basic.objective_min", "basic.objective_max", "basic.blocks_min", "basic.blocks_max", , "basic.cells_total", "basic.cells_filled", "basic.minimize_fun", "basic.costs_fun_evals", "basic.costs_runtime", , "nbc.nn_nb.sd_ratio", "nbc.nn_nb.mean_ratio", "nbc.nn_nb.cor", "nbc.dist_ratio.coeff_var", "nbc.nb_fitness.cor", , "nbc.costs_fun_evals", "nbc.costs_runtime", "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10", , "disp.ratio_mean_25", "disp.ratio_median_02", "disp.ratio_median_05", "disp.ratio_median_10", "disp.ratio_median_25", , "disp.diff_mean_02", "disp.diff_mean_05", "disp.diff_mean_10", "disp.diff_mean_25", "disp.diff_median_02", , "disp.diff_median_05", "disp.diff_median_10", "disp.diff_median_25", "disp.costs_fun_evals", "disp.costs_runtime", , "ic.h.max", "ic.eps.s", "ic.eps.max", "ic.eps.ratio", "ic.m0", , "ic.costs_fun_evals", "ic.costs_runtime", "pca.expl_var.cov_x", "pca.expl_var.cor_x", "pca.expl_var.cov_init", , "pca.expl_var.cor_init", "pca.expl_var_PC1.cov_x", "pca.expl_var_PC1.cor_x", "pca.expl_var_PC1.cov_init", "pca.expl_var_PC1.cor_init", , "pca.costs_fun_evals", "pca.costs_runtime", "ela_distr.skewness", "ela_distr.kurtosis", "ela_distr.number_of_peaks", , "ela_distr.costs_fun_evals", "ela_distr.costs_runtime", "ela_meta.lin_simple.adj_r2", "ela_meta.lin_simple.intercept", "ela_meta.lin_simple.coef.min", , "ela_meta.lin_simple.coef.max", "ela_meta.lin_simple.coef.max_by_min", "ela_meta.lin_w_interact.adj_r2", "ela_meta.quad_simple.adj_r2", "ela_meta.quad_simple.cond", , "ela_meta.quad_w_interact.adj_r2", "ela_meta.costs_fun_evals", "ela_meta.costs_runtime")
pdf(name)
tries = unique(df[,c("dim", "fun", "run")])
for(i in 1:nrow(tries)){
  data = df[df$dim==tries$dim[i] & df$fun==tries$fun[i] &df$run==tries$run[i],]
  data= data[data$f1 <1000 & data$f2 <1000,]
  insts = unique(data$inst)
  cols = rep(rainbow(3),4)
  plot(0,type="n", main=paste("dim", tries$dim[i], "fun", tries$fun[i]),xlab="evaluation", ylab="fitness", xlim=c(0, max(data$evaluation)), ylim = c(min(data$fitness, na.rm=TRUE), max(data$fitness, na.rm=TRUE)))
  for(j in 1:length(insts)){
    dt = data[data$inst == insts[j],]
    lines(dt$evaluation, dt$fitness, col=cols[j])
  }
  
  plot(0,type="n", main=paste("decreasing dim", tries$dim[i], "fun", tries$fun[i]),xlab="evaluation", ylab="fitness", xlim=c(0, max(data$evaluation)), ylim = c(min(data$fitness, na.rm=TRUE), max(data$fitness, na.rm=TRUE)))
  for(j in 1:length(insts)){
    dt = data[data$inst == insts[j],]
    min = 2000
    idx = logical(nrow(dt))
    idx[1] = TRUE
    for(k in 2:nrow(dt)){
      if(!is.na(dt$fitness[k]) & dt$fitness[k]<min){
        idx[k]=TRUE
        min=dt$fitness[k]
      }else{
        idx[k]=FALSE
      }
    }
    dt = dt[idx,]
    lines(c(dt$evaluation, max(data$evaluation)), c(dt$fitness, min(dt$fitness, na.rm=TRUE)), col=cols[j])
  }
  legend("topright", c("inst 1", "inst 2", "inst 3"), col=cols, pch=16)
}

dev.off()

