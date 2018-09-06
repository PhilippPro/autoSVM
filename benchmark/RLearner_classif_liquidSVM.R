makeRLearner.classif.liquidSVM = function() {
  makeRLearnerClassif(
    cl = "classif.liquidSVM",
    package = "liquidSVM",
    par.set = makeParamSet(
      # makeDiscreteLearnerParam(id = "type", default = "C-classification", values = c("C-classification", "nu-classification")),
      # makeNumericLearnerParam(id = "cost",  default = 1, lower = 0, requires = quote(type == "C-classification")),
      # makeNumericLearnerParam(id = "nu", default = 0.5, requires = quote(type == "nu-classification")),
      makeNumericVectorLearnerParam("class.weights", len = NA_integer_, lower = 0),
      # makeDiscreteLearnerParam(id = "kernel", default = "radial", values = c("linear", "polynomial", "radial", "sigmoid")),
      # makeIntegerLearnerParam(id = "degree", default = 3L, lower = 1L, requires = quote(kernel == "polynomial")),
      # makeNumericLearnerParam(id = "coef0", default = 0, requires = quote(kernel == "polynomial" || kernel == "sigmoid")),
      # makeNumericLearnerParam(id = "gamma", lower = 0, requires = quote(kernel != "linear")),
      # makeNumericLearnerParam(id = "cachesize", default = 40L),
      # makeNumericLearnerParam(id = "tolerance", default = 0.001, lower = 0),
      # makeLogicalLearnerParam(id = "shrinking", default = TRUE),
      # makeIntegerLearnerParam(id = "cross", default = 0L, lower = 0L, tunable = FALSE),
      # makeLogicalLearnerParam(id = "fitted", default = TRUE, tunable = FALSE),
      # makeLogicalVectorLearnerParam(id = "scale", default = TRUE, tunable = TRUE),
      makeIntegerLearnerParam(id = "threads", default = 1L, lower = 0L)
    ),
    properties = c("twoclass", "multiclass", "numerics", "factors", "prob", "class.weights"),
    class.weights.param = "class.weights",
    name = "Support Vector Machines (libsvm)",
    short.name = "svm",
    callees = "svm"
  )
}

trainLearner.classif.liquidSVM = function(.learner, .task, .subset, .weights = NULL,  ...) {
  f = getTaskFormula(.task)
  liquidSVM::svm(f, getTaskData(.task, .subset), predict.prob = .learner$predict.type == "prob",  ...)
}

predictLearner.classif.liquidSVM = function(.learner, .model, .newdata, ...) {
  predict(.model$learner.model, newdata = .newdata, ...)
  # res = as.matrix(predict(.model$learner.model, newdata = .newdata, ...))
  # res = res/rowSums(res)
  # colnames(res) = getTaskClassLevels(.task)
  # rownames(res) = NULL
  # res
}

# Problems with probability
# library(liquidSVM)
# library(OpenML)
# library(mlr)
# task = getOMLTask(31)
# task = convertOMLTaskToMlr(task)$mlr.task
# train = getTaskData(task)
# model <- mcSVM(class~., train, predict.prob = TRUE)
# train$class = as.numeric(train$class)
# model <- mcSVM(class~., train, predict.prob = TRUE)
# res = as.matrix(predict(model, newdata = train))
# 
# library(OpenML)
# task = getOMLTask(37)
# task = convertOMLTaskToMlr(task)$mlr.task
# train = getTaskData(task)
# train$class = as.numeric(train$class)
# model <- mcSVM(class~., train, predict.prob = TRUE)
# res = as.matrix(predict(model, newdata = train))
# 
# model <- mcSVM( Species~. , iris, predict.prob = TRUE)
# res = as.matrix(predict(model, newdata = iris))
# rowSums(res)
# # ungleich 1!