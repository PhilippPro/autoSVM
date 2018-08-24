#autoSVM = function(task, measure = NULL, iters = 70, iters.warmup = 30, num.threads = NULL, num.trees = 1000, 
#  parameters = list(replace = FALSE, respect.unordered.factors = "order"), 
#  tune.parameters = c("mtry", "min.node.size", "sample.fraction"), save.file.path = NULL,
#  build.final.model = TRUE, show.info = getOption("mlrMBO.show.info", TRUE)) 