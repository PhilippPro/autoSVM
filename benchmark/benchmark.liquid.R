library(devtools)
library(OpenML)
library(mlr)
library(mlrHyperopt)
load_all("../autoSVM")
source("benchmark/mlrHyperoptWrapper.R")
source("benchmark/RLearner_classif_liquidSVM.R")

lrn.liquid = list(
  makeLearner("classif.svm", predict.type = "prob"),
  makeLearner("classif.ksvm", predict.type = "prob"),
  # Defaults of the tunability paper
  makeLearner("classif.svm", id = "def.svm", predict.type = "prob", kernel = "radial", gamma = 0.005, cost = 680),
  makeLearner("classif.ksvm", id = "def.ksvm", predict.type = "prob", kernel = "rbfdot", sigma = 0.005, C = 680),
  makeLearner("classif.liquidSVM", predict.type = "response")
)

rdesc = makeResampleDesc("Holdout")
configureMlr(on.learner.error = "warn")
measures = list(mmce, timetrain) #list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)
configureMlr(on.learner.error = "warn")
set.seed(126)
bmr1 = benchmark(lrn.liquid, iris.task, rdesc, measures)
bmr1

library(OpenML)
#task.ids = listOMLTasks(number.of.classes = 2L, number.of.missing.values = 0, tag = "OpenML100", estimation.procedure = "10-fold Crossvalidation")$task.id
load(file = "./benchmark/task_ids.RData")

# time estimation
load("./benchmark/time.estimate.RData")

rdesc = makeResampleDesc("RepCV", reps = 10, folds = 5)
# benchmark
bmr_liquid = list()

# Choose small and big datasets
# select datasets where RF do not take longer than ...

# take only small ones first; afterwards some bigger datasets
task.ids.bmr = task.ids[which((unlist(time.estimate)-100)<60)]
cbind(time.estimate, (unlist(time.estimate)-100)<60)
unlist(time.estimate)[which((unlist(time.estimate)-100)<60)]

for(i in seq_along(task.ids.bmr)) { # 13 datasets
  print(i)
  set.seed(200 + i)
  task = getOMLTask(task.ids.bmr[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_liquid[[i]] = benchmark(lrn.liquid, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_liquid, file = "./benchmark/bmr_liquid.RData")
}
load("./benchmark/bmr_liquid.RData")
# Which datasets are not super easy (AUC < 0.99) and discriminate between the algorithms?

# medium datasets (between 160 seconds and 10 minutes)
task.ids.bmr_liquid2 = task.ids[which((unlist(time.estimate)-100)>60 & (unlist(time.estimate))<600)]
unlist(time.estimate)[which((unlist(time.estimate)-100)>60 & (unlist(time.estimate))<600 )]
rdesc = makeResampleDesc("CV", iters = 5)

# 13 datasets
for(i in seq_along(task.ids.bmr_liquid2)) {
  print(i)
  set.seed(300 + i)
  task = getOMLTask(task.ids.bmr_liquid2[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_liquid[[length(bmr_liquid) + 1]] = benchmark(lrn.liquid, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_liquid, file = "./benchmark/bmr_liquid.RData")
}
load("./benchmark/bmr_liquid.RData")

# big datasets (between 10 minutes and 1 hour)
task.ids.bmr_liquid3 = task.ids[which((unlist(time.estimate))>600 & (unlist(time.estimate))<3600)]
unlist(time.estimate)[which((unlist(time.estimate))>600 & (unlist(time.estimate))<3600)]
# 9 datasets

rdesc = makeResampleDesc("CV", iters = 5)
bmr_liquid_big = list()
# Hier evtl. doch ein paar Wiederholungen einbauen, da die Streuung sonst zu groß ist. 
# Zunächst einfach mal durchlaufen lassen (kann dannach hinzugefügt werden).
for(i in seq_along(task.ids.bmr_liquid3)) {
  print(i)
  set.seed(400 + i) 
  task = getOMLTask(task.ids.bmr_liquid3[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_liquid_big[[length(bmr_liquid_big) + 1]] = benchmark(lrn.liquid, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_liquid_big, file = "./benchmark/bmr_liquid_big.RData")
}
load("./benchmark/bmr_liquid_big.RData")

# Very big datasets, 4 datasets
task.ids.bmr_liquid4 = task.ids[which((unlist(time.estimate))>=3600)]
unlist(time.estimate)[which((unlist(time.estimate))>=3600)]

for(i in seq_along(task.ids.bmr_liquid4)) {
  print(i)
  set.seed(400 + i) 
  task = getOMLTask(task.ids.bmr_liquid4[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_liquid_big[[length(bmr_liquid_big) + 1]] = benchmark(lrn.liquid, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_liquid_big, file = "./benchmark/bmr_liquid_big.RData")
}
load("./benchmark/bmr_liquid_big.RData")

# Amazon bricht ab bei kernlab, zu viel Speicherverbrauch vermutlich.


load("./benchmark/bmr.RData")
load("./benchmark/bmr_liquid.RData")
load("./benchmark/bmr_liquid_big.RData")
bmr_liquid = c(bmr_liquid, bmr_liquid_big)

library(mlr)
# Data cleaning
names = names(bmr_liquid[[1]]$results[[1]]$classif.svm$aggr)
nr.learners = length(bmr_liquid[[1]]$learners)

# if less than 20 percent NA, impute by the mean of the other iterations
# for(i in seq_along(bmr_liquid)) {
#   for(j in 1:nr.learners) {
#     print(paste(i,j))
#     na.percentage = mean(is.na(bmr_liquid[[i]]$results[[1]][[j]]$measures.test$mmce))
#     if(na.percentage > 0 & na.percentage <= 0.2) {
#       resis = bmr_liquid[[i]]$results[[1]][[j]]$measures.test
#       bmr_liquid[[i]]$results[[1]][[j]]$aggr = colMeans(resis[!is.na(resis$mmce),])[2:6]
#       names(bmr_liquid[[i]]$results[[1]][[j]]$aggr) = names
#     }
#   }
# }

# Analysis of time
resi = list()
resi[[1]] = data.frame(getBMRAggrPerformances(bmr_liquid[[1]]))

for(i in 2:length(bmr_liquid)) {
  resi[[i]] = data.frame(getBMRAggrPerformances(bmr_liquid[[i]]))
  # caret gets no result, if NA
}

lty.vec = c(rep(1,4), c(2,3,4,5))
library(RColorBrewer)
col.vec = brewer.pal(8, "Dark2")
time = matrix(NA, length(bmr_liquid),  ncol(resi[[1]]))
for(i in seq_along(bmr_liquid)) {
  time[i,] = unlist(resi[[i]][nrow(resi[[i]]),])
}
time_order = order(time[,1])
time = time[time_order,]
plot(time[,1], type = "l", ylim = c(0, max(time, na.rm = T)), ylab = "Time in seconds", xlab = "Dataset number", col = col.vec[1])
for(i in 2:ncol(time)){
  points(1:length(bmr_liquid), time[,i], col = col.vec[i], cex = 0.4)
  lines(time[,i], col = col.vec[i], lty = lty.vec[i])
}
leg.names = lrn.names = c("e1071", "kernlab", "def.e1071", "def.kernlab", "liquidSVM")
legend("topleft", legend = leg.names, col = col.vec, lty = lty.vec)


# Descriptive Analysis
resi = list()
resi[[1]] = data.frame(getBMRAggrPerformances(bmr_liquid[[1]]))
res_aggr = resi[[1]]
res_aggr_rank = apply(resi[[1]], 1, rank)

for(i in 2:length(bmr_liquid)) {
  resi[[i]] = data.frame(getBMRAggrPerformances(bmr_liquid[[i]]))
  # models gets the worst result, if NA
  for(j in 1:nr.learners) {
    print(paste(i,j))
    if(is.na(resi[[i]][1,j])) {
      resi[[i]][1,j] = max(resi[[i]][1,], na.rm = T)
      # resi[[i]][2,j] = min(resi[[i]][2,], na.rm = T)
      # resi[[i]][3,j] = max(resi[[i]][3,], na.rm = T)
      # resi[[i]][4,j] = max(resi[[i]][4,], na.rm = T)
      # resi[[i]][5,j] = max(resi[[i]][5,], na.rm = T)
    }
  }
  res_aggr = res_aggr + resi[[i]]
  res_aggr_rank = res_aggr_rank + apply(resi[[i]], 1, rank)
}
res_aggr = res_aggr/length(bmr_liquid)
# average rank matrix
res_aggr_rank = res_aggr_rank/length(bmr_liquid)

library(stringr)
library(xtable)
rownames(res_aggr_rank) = lrn.names
colnames(res_aggr_rank) = str_sub(colnames(res_aggr_rank), start=1, end=-11)
colnames(res_aggr_rank) = c("Error rate",  "Training Runtime")# "(Multiclass) AUC", "Brier Score", "Logarithmic Loss", "Training Runtime")
res_aggr_rank
xtable(res_aggr_rank, digits = 2, caption = "Average rank results of the different algorithms for the small datasets", label = "rank_small")

library(knitr)
rownames(res_aggr) = paste("--", colnames(res_aggr_rank))
kable(t(round(res_aggr,4)))
colnames(res_aggr_rank) = paste("--", colnames(res_aggr_rank))
kable(round(res_aggr_rank,2))

colnames(res_aggr) = lrn.names
kable(round(t(res_aggr), 4))

# Graphical analysis of performance
# Compared to ranger model
perfis = list()
perfi = matrix(NA, length(bmr_liquid),  ncol(resi[[1]]))
for(j in c(1:4)) {
  for(i in 1:length(bmr_liquid)) {
    perfi[i,] = unlist(resi[[i]][j,]) - unlist(resi[[i]][j,1])
  }
  colnames(perfi) = lrn.names
  perfis[[j]] = perfi
}

measure.names = c("Error rate", "AUC", "Brier score", "Logarithmic Loss")
op <- par(mfrow = c(1,2),
  oma = c(0,0,0,0) + 0.1,
  mar = c(2.5,2,1,0) + 0.1)
outline = c(TRUE, FALSE)
outlier_name = c("", "(without outliers)")
for(i in 1:1) {
  for(j in 1:2) {
    boxplot(perfis[[i]], main = paste(measure.names[i], outlier_name[j]), horizontal = F, xaxt = "n", outline = outline[j])
    axis(1, at = c(1,2,3,4,5,6,7,8), labels = FALSE, cex = 0.1, tck = -0.02)
    #axis(1, at = c(6,8), labels = FALSE, cex = 0.1, tck = -0.07)
    mtext(lrn.names, 1, line = 0.1, at = 1:length(lrn.names), cex = 0.6)
    #mtext(lrn.names2[c(6,8)], 1, line = 0.7, at = c(6,8), cex = 0.6)
    abline(0, 0, col = "red")
    #axis(1,at=c(0.5,1,2,3,3.5,4,4.5),col="black",line=1.15,tick=T,labels=rep("",7),lwd=2,lwd.ticks=0)
  }
}



# Visualisierumg der Surrogate Modelle
bmr_liquid
pdf(paste0("benchmark/images/surrogates.pdf"))
for(i in seq_along(bmr_liquid)){
  print(i)
  plotOptPath(bmr_liquid[[i]]$results[[1]]$svm.hyperopt.hyperopt$extract[[1]]$opt.path, title = paste0("dataset_", i))
}
dev.off()

# Anhand der Grafiken analysieren, in welche Richtung der Hyperparameterraum ausgeweitet werden sollte.



# Vergleich mit MBO
for(i in seq_along(bmr)[-17]){
  print(i)
  res = c(bmr[[i]]$results[[1]]$svm.hyperopt.hyperopt$aggr[1],
    bmr[[i]]$results[[1]]$ksvm.hyperopt.hyperopt$aggr[1],
    bmr_liquid[[i]]$results[[1]]$classif.liquidSVM$aggr[1])
  names(res) = c("svm.hyperopt", "ksvm.hyperopt", "liquidSVM")
  print(res)
}

erg = erg_runtime = matrix(NA, 39, 5)
for(i in seq_along(bmr_liquid)){
  erg[i, 1] = bmr_liquid[[i]]$results[[1]]$classif.svm$aggr[1]
  if(i < 35) erg[i, 2] = bmr_liquid[[i]]$results[[1]]$classif.ksvm$aggr[1]
  erg[i, 3] = bmr_liquid[[i]]$results[[1]]$classif.liquidSVM$aggr[1]
  erg_runtime[i, 1] = bmr_liquid[[i]]$results[[1]]$classif.svm$aggr[2]
  if(i < 35) erg_runtime[i, 2] = bmr_liquid[[i]]$results[[1]]$classif.ksvm$aggr[2]
  erg_runtime[i, 3] = bmr_liquid[[i]]$results[[1]]$classif.liquidSVM$aggr[2]
  if(i != 17 & i <=26) {
    erg[i, 4] = bmr[[i]]$results[[1]]$svm.hyperopt.hyperopt$aggr[1]
    erg[i, 5] = bmr[[i]]$results[[1]]$ksvm.hyperopt.hyperopt$aggr[1]
    erg_runtime[i, 4] = bmr[[i]]$results[[1]]$svm.hyperopt.hyperopt$aggr[5]
    erg_runtime[i, 5] = bmr[[i]]$results[[1]]$ksvm.hyperopt.hyperopt$aggr[5]
  }
}

pdf("./benchmark/images/e1071_vs_ksvm_vs_liquidSVM.pdf")
plot(erg[order(erg[,1]),1], type = "l", ylab = "mmce", xlab = "ordered datasets (by mmce)", main = "5foldCV", col = "black")
lines(erg[order(erg[,1]),2], col = "green")
lines(erg[order(erg[,1]),3], col = "blue")
lines(erg[order(erg[,1]),4], col = "red")
points(erg[order(erg[,1]),4], col = "red", pch = 15)
lines(erg[order(erg[,1]),5], col = "violet")
points(erg[order(erg[,1]),5], col = "violet", pch = 15)
legend("topleft", c("e1071-default", "ksvm-default", "liquidSVM-default", "e1071-hyperopt", "ksvm-hyperopt"), col = c("black", "green", "blue", "red", "violet"), lty = 1)

plot(erg_runtime[order(erg_runtime[,1]),1], type = "l", ylab = "runtime", xlab = "ordered datasets (by runtime)", main = "5foldCV", col = "black")
lines(erg_runtime[order(erg_runtime[,1]),2], col = "green")
lines(erg_runtime[order(erg_runtime[,1]),3], col = "blue")
lines(erg_runtime[order(erg_runtime[,1]),4], col = "red")
lines(erg_runtime[order(erg_runtime[,1]),5], col = "violet")
legend("topleft", c("e1071-default", "ksvm-default", "liquidSVM-default", "e1071-hyperopt", "ksvm-hyperopt"), col = c("black", "green", "blue", "red", "violet"), lty = 1)

plot(erg_runtime[order(erg_runtime[,1]),1], type = "l", ylim = range(erg_runtime, na.rm = T), ylab = "runtime (log-scale)", xlab = "ordered datasets (by runtime)", main = "5foldCV", col = "black", log = "y")
lines(erg_runtime[order(erg_runtime[,1]),2], col = "green")
lines(erg_runtime[order(erg_runtime[,1]),3], col = "blue")
lines(erg_runtime[order(erg_runtime[,1]),4], col = "red")
lines(erg_runtime[order(erg_runtime[,1]),5], col = "violet")
legend("topleft", c("e1071-default", "ksvm-default", "liquidSVM-default", "e1071-hyperopt", "ksvm-hyperopt"), col = c("black", "green", "blue", "red", "violet"), lty = 1)
dev.off()
  