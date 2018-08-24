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

lrn1 = makeLearner("classif.svm", id = "svm.hyperopt", predict.type = "prob")
lrn1 = makeHyperoptWrapper(lrn1, par.config)

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

lrn2 = makeLearner("classif.ksvm", id = "ksvm.hyperopt", predict.type = "prob")
lrn2 = makeHyperoptWrapper(lrn2, par.config)

# target measure??

# other kernels

par.set = makeParamSet(
  makeNumericParam("cost", lower = -5, upper = 15, trafo = function(x) 2^x)
)
par.config = makeParConfig(
  par.set = par.set,
  par.vals = list(kernel = "linear"),
  learner.name = "svm"
)
lrn1.linear = makeLearner("classif.svm", id = "svm.hyperopt.linear", predict.type = "prob")
lrn1.linear = makeHyperoptWrapper(lrn1.linear, par.config)

par.set = makeParamSet(
  makeNumericParam("C", lower = -5, upper = 15, trafo = function(x) 2^x)
)
par.config = makeParConfig(
  par.set = par.set,
  par.vals = list(kernel = "vanilladot"),
  learner.name = "svm"
)
lrn2.linear = makeLearner("classif.ksvm", id = "ksvm.hyperopt.linear", predict.type = "prob")
lrn2.linear = makeHyperoptWrapper(lrn2.linear, par.config)

par.set = makeParamSet(
  makeNumericParam("cost", lower = -5, upper = 15, trafo = function(x) 2^x),
  makeIntegerParam(id = "degree", default = 3L, lower = 1L, upper = 5)
)
par.config = makeParConfig(
  par.set = par.set,
  par.vals = list(kernel = "polynomial"),
  learner.name = "svm"
)
lrn1.polynomial = makeLearner("classif.svm", id = "svm.hyperopt.polynomial", predict.type = "prob")
lrn1.polynomial = makeHyperoptWrapper(lrn1.polynomial, par.config)

par.set = makeParamSet(
  makeNumericParam("C", lower = -5, upper = 15, trafo = function(x) 2^x),
  makeIntegerParam(id = "degree", default = 3L, lower = 1L, upper = 5)
)
par.config = makeParConfig(
  par.set = par.set,
  par.vals = list(kernel = "polydot"),
  learner.name = "svm"
)
lrn2.polynomial = makeLearner("classif.ksvm", id = "ksvm.hyperopt.polynomial", predict.type = "prob")
lrn2.polynomial = makeHyperoptWrapper(lrn2.polynomial, par.config)

