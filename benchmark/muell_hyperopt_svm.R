devtools::install_github("berndbischl/ParamHelpers") # version >= 1.11 needed.
devtools::install_github("jakob-r/mlrHyperopt", dependencies = FALSE)

library(mlrHyperopt)

res = hyperopt(iris.task, learner = "classif.svm")
res
plotOptPath(res$opt.path)

task = makeClassifTask(data = iris, target = "Species")
lrn = makeLearner("classif.svm")
lrn = makeHyperoptWrapper(lrn)
mod = train(lrn, task)
print(getTuneResult(mod))
plotOptPath(mod$learner.model$opt.result$opt.path)

par.set = makeParamSet(
  makeNumericParam("cost", lower = -5, upper = 15, trafo = function(x) 2^x),
  makeNumericParam("gamma", lower = -15, upper = 3, trafo = function(x) 2^x) #, requires = quote(kernel == "radial"))
)
par.config = makeParConfig(
  par.set = par.set,
  par.vals = list(kernel = "radial"),
  learner.name = "svm",
  note = "Based on the practical guide to SVM: https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf"
)
#uploadParConfig(par.config, "philipp_probst@gmx.de")

lrn = makeLearner("classif.svm", id = "svm.hyperopt")
lrn = makeHyperoptWrapper(lrn, par.config)
mod = train(lrn, task)
print(getTuneResult(mod))
plotOptPath(mod$learner.model$opt.result$opt.path)

par.set = makeParamSet(
  makeNumericParam("C", lower = -5, upper = 15, trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -15, upper = 3, trafo = function(x) 2^x)
)
par.config = makeParConfig(
  par.set = par.set,
  par.vals = list(kernel = "rbfdot"),
  learner.name = "svm",
  note = "Based on the practical guide to SVM: https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf"
)
#uploadParConfig(par.config, "philipp_probst@gmx.de")

lrn = makeLearner("classif.ksvm", id = "ksvm.hyperopt")
lrn = makeHyperoptWrapper(lrn, par.config)
mod = train(lrn, task)
print(getTuneResult(mod))
plotOptPath(mod$learner.model$opt.result$opt.path)

# different recommendation on the plot and the numbers??

# with many observations: Use linear kernel.
# Is there always a tradeoff between c and sigma?
# Plot several graphs like above for several datasets
# see practical guide to SVM
# Visualisation for separating the points