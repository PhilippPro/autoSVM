library(devtools)
library(OpenML)
library(mlr)
library(mlrHyperopt)
load_all("../autoSVM")
source("benchmark/mlrHyperoptWrapper.R")

lrn.ranger = list(
  makeLearner("classif.ranger")
)

rdesc = makeResampleDesc("Holdout")
configureMlr(on.learner.error = "warn")
measures = list(mmce, timetrain) #list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)
configureMlr(on.learner.error = "warn")
set.seed(126)
bmr1 = benchmark(lrn.ranger, iris.task, rdesc, measures)
bmr1

library(OpenML)
#task.ids = listOMLTasks(number.of.classes = 2L, number.of.missing.values = 0, tag = "OpenML100", estimation.procedure = "10-fold Crossvalidation")$task.id
load(file = "./benchmark/task_ids.RData")

# time estimation
load("./benchmark/time.estimate.RData")

rdesc = makeResampleDesc("RepCV", reps = 10, folds = 5)
# benchmark
bmr_ranger = list()

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
  bmr_ranger[[i]] = benchmark(lrn.ranger, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_ranger, file = "./benchmark/bmr_ranger.RData")
}
load("./benchmark/bmr_ranger.RData")
# Which datasets are not super easy (AUC < 0.99) and discriminate between the algorithms?

# medium datasets (between 160 seconds and 10 minutes)
task.ids.bmr_ranger2 = task.ids[which((unlist(time.estimate)-100)>60 & (unlist(time.estimate))<600)]
unlist(time.estimate)[which((unlist(time.estimate)-100)>60 & (unlist(time.estimate))<600 )]
rdesc = makeResampleDesc("CV", iters = 5)

# 13 datasets
for(i in seq_along(task.ids.bmr_ranger2)) {
  print(i)
  set.seed(300 + i)
  task = getOMLTask(task.ids.bmr_ranger2[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_ranger[[length(bmr_ranger) + 1]] = benchmark(lrn.ranger, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_ranger, file = "./benchmark/bmr_ranger.RData")
}
load("./benchmark/bmr_ranger.RData")

# big datasets (between 10 minutes and 1 hour)
task.ids.bmr_ranger3 = task.ids[which((unlist(time.estimate))>600 & (unlist(time.estimate))<3600)]
unlist(time.estimate)[which((unlist(time.estimate))>600 & (unlist(time.estimate))<3600)]
# 9 datasets

rdesc = makeResampleDesc("CV", iters = 5)
bmr_ranger_big = list()
# Hier evtl. doch ein paar Wiederholungen einbauen, da die Streuung sonst zu groß ist. 
# Zunächst einfach mal durchlaufen lassen (kann dannach hinzugefügt werden).
for(i in seq_along(task.ids.bmr_ranger3)) {
  print(i)
  set.seed(400 + i) 
  task = getOMLTask(task.ids.bmr_ranger3[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_ranger_big[[length(bmr_ranger_big) + 1]] = benchmark(lrn.ranger, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_ranger_big, file = "./benchmark/bmr_ranger_big.RData")
}
load("./benchmark/bmr_ranger_big.RData")

# Very big datasets, 4 datasets
task.ids.bmr_ranger4 = task.ids[which((unlist(time.estimate))>=3600)]
unlist(time.estimate)[which((unlist(time.estimate))>=3600)]

for(i in seq_along(task.ids.bmr_ranger4)) {
  print(i)
  set.seed(400 + i) 
  task = getOMLTask(task.ids.bmr_ranger4[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_ranger_big[[length(bmr_ranger_big) + 1]] = benchmark(lrn.ranger, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_ranger_big, file = "./benchmark/bmr_ranger_big.RData")
}
load("./benchmark/bmr_ranger_big.RData")


load("./benchmark/bmr_ranger.RData")
load("./benchmark/bmr_ranger_big.RData")
bmr_ranger = c(bmr_ranger, bmr_ranger_big)

library(mlr)
# Data cleaning
names = names(bmr_ranger[[1]]$results[[1]]$classif.svm$aggr)
nr.learners = length(bmr_ranger[[1]]$learners)

# if less than 20 percent NA, impute by the mean of the other iterations
# for(i in seq_along(bmr_ranger)) {
#   for(j in 1:nr.learners) {
#     print(paste(i,j))
#     na.percentage = mean(is.na(bmr_ranger[[i]]$results[[1]][[j]]$measures.test$mmce))
#     if(na.percentage > 0 & na.percentage <= 0.2) {
#       resis = bmr_ranger[[i]]$results[[1]][[j]]$measures.test
#       bmr_ranger[[i]]$results[[1]][[j]]$aggr = colMeans(resis[!is.na(resis$mmce),])[2:6]
#       names(bmr_ranger[[i]]$results[[1]][[j]]$aggr) = names
#     }
#   }
# }

# Analysis of time
resi = list()
resi[[1]] = data.frame(getBMRAggrPerformances(bmr_ranger[[1]]))

for(i in 2:length(bmr_ranger)) {
  resi[[i]] = data.frame(getBMRAggrPerformances(bmr_ranger[[i]]))
  # caret gets no result, if NA
}

lty.vec = c(rep(1,4), c(2,3,4,5))
library(RColorBrewer)
col.vec = brewer.pal(8, "Dark2")
time = matrix(NA, length(bmr_ranger),  ncol(resi[[1]]))
for(i in seq_along(bmr_ranger)) {
  time[i,] = unlist(resi[[i]][nrow(resi[[i]]),])
}
time_order = order(time[,1])
time = time[time_order,]
plot(time[,1], type = "l", ylim = c(0, max(time, na.rm = T)), ylab = "Time in seconds", xlab = "Dataset number", col = col.vec[1])
for(i in 2:ncol(time)){
  points(1:length(bmr_ranger), time[,i], col = col.vec[i], cex = 0.4)
  lines(time[,i], col = col.vec[i], lty = lty.vec[i])
}
leg.names = lrn.names = c("e1071", "kernlab", "def.e1071", "def.kernlab", "liquidSVM")
legend("topleft", legend = leg.names, col = col.vec, lty = lty.vec)


# Descriptive Analysis
resi = list()
resi[[1]] = data.frame(getBMRAggrPerformances(bmr_ranger[[1]]))
res_aggr = resi[[1]]
res_aggr_rank = apply(resi[[1]], 1, rank)

for(i in 2:length(bmr_ranger)) {
  resi[[i]] = data.frame(getBMRAggrPerformances(bmr_ranger[[i]]))
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
res_aggr = res_aggr/length(bmr_ranger)
# average rank matrix
res_aggr_rank = res_aggr_rank/length(bmr_ranger)

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
perfi = matrix(NA, length(bmr_ranger),  ncol(resi[[1]]))
for(j in c(1:4)) {
  for(i in 1:length(bmr_ranger)) {
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
bmr_ranger
pdf(paste0("benchmark/images/surrogates.pdf"))
for(i in seq_along(bmr_ranger)){
  print(i)
  plotOptPath(bmr_ranger[[i]]$results[[1]]$svm.hyperopt.hyperopt$extract[[1]]$opt.path, title = paste0("dataset_", i))
}
dev.off()

# Anhand der Grafiken analysieren, in welche Richtung der Hyperparameterraum ausgeweitet werden sollte.



# Vergleich mit MBO
for(i in seq_along(bmr_ranger)){
  print(i)
  print(bmr_liquid[[i]])
  print(bmr_ranger[[i]])
}
# 15:23 für ranger