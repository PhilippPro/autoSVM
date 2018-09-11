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

load("./benchmark/bmr_liquid.RData")
load("./benchmark/bmr_liquid_big.RData")
bmr_liquid = c(bmr_liquid, bmr_liquid_big)


# Vergleich mit MBO
for(i in seq_along(bmr_ranger)){
  print(i)
  print(bmr_liquid[[i]])
  print(bmr_ranger[[i]])
}
# 15:23 für ranger


for(i in seq_along(bmr_ranger)[-17]){
  print(i)
  res = c(bmr_ranger[[i]]$results[[1]]$classif.ranger$aggr[1],
    bmr_liquid[[i]]$results[[1]]$classif.liquidSVM$aggr[1])
  names(res) = c("ranger", "liquidSVM")
  print(res)
}

erg = erg_runtime = matrix(NA, 39, 2)
for(i in seq_along(bmr_liquid)){
  erg[i, 1] = bmr_ranger[[i]]$results[[1]]$classif.ranger$aggr[1]
  erg[i, 2] = bmr_liquid[[i]]$results[[1]]$classif.liquidSVM$aggr[1]
  erg_runtime[i, 1] = bmr_ranger[[i]]$results[[1]]$classif.ranger$aggr[2]
  erg_runtime[i, 2] = bmr_liquid[[i]]$results[[1]]$classif.liquidSVM$aggr[2]
}

pdf("./benchmark/images/ranger_vs_liquidSVM.pdf")
plot(erg[order(erg[,1]),1], type = "l", ylab = "mmce", xlab = "ordered datasets (by mmce)", main = "5foldCV", col = "green")
lines(erg[order(erg[,1]),2], col = "blue")
legend("topleft", c("ranger", "liquidSVM-default"), col = c("green", "blue"), lty = 1)

plot(erg_runtime[order(erg_runtime[,1]),1], type = "l", ylab = "runtime", xlab = "ordered datasets (by runtime)", main = "5foldCV", col = "green")
lines(erg_runtime[order(erg_runtime[,1]),2], col = "blue")
legend("topleft", c("ranger", "liquidSVM-default"), col = c("green", "blue"), lty = 1)

plot(erg_runtime[order(erg_runtime[,1]),1], type = "l", ylim = range(erg_runtime, na.rm = T), ylab = "runtime (log-scale)", xlab = "ordered datasets (by runtime)", main = "5foldCV", col = "green", log = "y")
lines(erg_runtime[order(erg_runtime[,1]),2], col = "blue")
legend("topleft", c("ranger", "liquidSVM-default"), col = c("green", "blue"), lty = 1)
dev.off()

# Datasets where the difference is big:
task.ids.all = c(task.ids.bmr, task.ids.bmr_ranger2, task.ids.bmr_ranger3, task.ids.bmr_ranger4)

# ranger better than liquidSVM
# Datensatz 27
id = task.ids.all[[27]]
task = getOMLTask(id)
task = convertOMLTaskToMlr(task)$mlr.task
task
head(task$env$data,20)
data = task$env$data
data$class = as.numeric(data$class)
M = cor(x = data.matrix(data))
library('corrplot') #package corrplot
corrplot(M, method = "circle")

# Datensatz 32: madelon: many non-informative features -> high mtry would be even better, svm cannot handle it
id = task.ids.all[[32]]
task = getOMLTask(id)
task = convertOMLTaskToMlr(task)$mlr.task
task
head(task$env$data,20)

data = task$env$data
x = data.matrix(data)
M = cor(x = data.matrix(data))
plot(M[, 501])
# library(qtlcharts)
# iplotCorr(x[,c(481:500,501)], reorder=FALSE)

# Datensatz 33
id = task.ids.all[[33]]
task = getOMLTask(id)
task = convertOMLTaskToMlr(task)$mlr.task
task
head(task$env$data,20)
data = task$env$data
x = data.matrix(data)
M = cor(x = data.matrix(data))
corrplot(M, method = "circle")

# liquidSVM better than ranger
# Datensatz 5: monks-problem2: Wird durch mtry-Tuning behoben, siehe tuneRanger-Paper
id = task.ids.all[[5]]
task = getOMLTask(id)
task = convertOMLTaskToMlr(task)$mlr.task
head(task$env$data,20)
# Datensatz 31
id = task.ids.all[[31]]
task = getOMLTask(id)
task = convertOMLTaskToMlr(task)$mlr.task
task
head(task$env$data,20)
data = task$env$data
x = data.matrix(data)
M = cor(x = data.matrix(data))
plot(M[, 101])
corrplot(M, method = "circle")
# Idee: Versuche mtry tuning
head(data)
library(gplots)
library()
heatmap(as.matrix(x[11:20,11:20]))

# (Datensatz 23, Datensatz13)








