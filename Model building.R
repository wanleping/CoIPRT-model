# Load the necessary packages.
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(nnet)
library(gbm)
library(pROC)
library(PRROC)
library(boot)
library(ggplot2)
library(ggprism)
library(DescTools)  

# Loading data
data <- read.csv("E:/WLP/WHU/article-idea/article/Cognitive_condition/最新数据Cognitive/16smote_minCognitive4_append.csv")
#names(data)[names(data) == 'Cognitive.condition'] <- 'label'
data$label <- as.factor(data$label)
head(data)

# Partitioning the dataset
set.seed(123)  # Set random seed
inTrain <- createDataPartition(y = data[, "label"], p = 0.8, list = FALSE)
traindata <- data[inTrain, ]  
temp_testdata <- data[-inTrain, ]  
inValidation <- createDataPartition(y = temp_testdata[, "label"], p = 0.5, list = FALSE)
validationdata <- temp_testdata[inValidation, ]  
testdata <- temp_testdata[-inValidation, ]  


# 
calculate_metrics <- function(predictions, true_labels, probabilities = NULL) {
  conf_matrix <- table(predictions, true_labels)
  TP <- conf_matrix[1, 1]
  FP <- conf_matrix[1, 2]
  TN <- conf_matrix[2, 2]
  FN <- conf_matrix[2, 1]
  
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision <- TP / (TP + FP)
  recall <- sensitivity
  f1_score <- 2 * (precision * recall) / (precision + recall)
  npv <- TN / (TN + FN)
  
  # 
  auc <- NA
  if (!is.null(probabilities)) {
    roc_curve <- roc(true_labels, probabilities)
    auc <- auc(roc_curve)
  }
  
  # Calculate the 95% confidence interval.
  metrics_ci <- list(
    accuracy_ci = BinomCI(TP + TN, TP + TN + FP + FN, conf.level = 0.95),
    sensitivity_ci = BinomCI(TP, TP + FN, conf.level = 0.95),
    specificity_ci = BinomCI(TN, TN + FP, conf.level = 0.95),
    precision_ci = BinomCI(TP, TP + FP, conf.level = 0.95),
    npv_ci = BinomCI(TN, TN + FN, conf.level = 0.95)
  )
  
  return(list(
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    npv = npv,
    auc = auc,
    metrics_ci = metrics_ci
  ))
}

# Define the function to calculate AUPRC and its confidence interval
calculate_auprc <- function(probabilities, true_labels, n_boot = 1000) {
  pr_curve <- pr.curve(scores.class0 = probabilities, weights.class0 = as.numeric(true_labels) - 1)
  auprc <- pr_curve$auc.integral
  auprc_boot <- numeric(n_boot)
  for (i in 1:n_boot) {
    boot_idx <- sample(1:length(probabilities), replace = TRUE)
    boot_pr <- pr.curve(scores.class0 = probabilities[boot_idx], weights.class0 = as.numeric(true_labels[boot_idx]) - 1)
    auprc_boot[i] <- boot_pr$auc.integral
  }
  auprc_ci <- quantile(auprc_boot, c(0.025, 0.975))
  
  return(list(auprc = auprc, auprc_ci = auprc_ci))
}

# RF model
rf_model <- randomForest(label ~ ., data = traindata)
rf_probabilities <- predict(rf_model, newdata = validationdata, type = "prob")[, 2]
rf_predictions <- ifelse(rf_probabilities > 0.5, levels(validationdata$label)[2], levels(validationdata$label)[1])
rf_metrics <- calculate_metrics(rf_predictions, validationdata$label, rf_probabilities)
rf_auprc <- calculate_auprc(rf_probabilities, validationdata$label)

# SVM model
svm_model <- svm(label ~ ., data = traindata, type = "C-classification", kernel = "radial", probability = TRUE)
svm_probabilities <- attr(predict(svm_model, newdata = validationdata, probability = TRUE), "probabilities")[, 2]
svm_predictions <- ifelse(svm_probabilities > 0.5, levels(validationdata$label)[2], levels(validationdata$label)[1])
svm_metrics <- calculate_metrics(svm_predictions, validationdata$label, svm_probabilities)
svm_auprc <- calculate_auprc(svm_probabilities, validationdata$label)

#GBDTmodel
print(table(traindata$label))
train_label_numeric <- ifelse(traindata$label == levels(traindata$label)[2], 1, 0)

print(table(train_label_numeric))
validation_label_numeric <- ifelse(validationdata$label == levels(validationdata$label)[2], 1, 0)

gbdt_model <- gbm(formula = train_label_numeric ~ ., 
                  data = traindata[, -which(names(traindata) == "label")], 
                  distribution = "bernoulli", 
                  n.trees = 100, 
                  interaction.depth = 3, 
                  shrinkage = 0.1, 
                  cv.folds = 5)


best_iter <- gbm.perf(gbdt_model, method = "cv")
gbdt_probabilities <- predict(gbdt_model, newdata = validationdata[, -which(names(validationdata) == "label")], n.trees = best_iter, type = "response")
gbdt_predictions <- ifelse(gbdt_probabilities > 0.5, levels(validationdata$label)[2], levels(validationdata$label)[1])
gbdt_metrics <- calculate_metrics(gbdt_predictions, validationdata$label, gbdt_probabilities)
gbdt_auprc <- calculate_auprc(gbdt_probabilities, validationdata$label)


# XGBoost model
print(table(traindata$label))
print(table(validationdata$label))
traindata$label <- as.numeric(traindata$label) - 1
validationdata$label <- as.numeric(validationdata$label) - 1
xgboost_model <- xgboost(
  data = as.matrix(traindata[, -ncol(traindata)]),
  label = traindata$label,
  nrounds = 150,
  max_depth = 4,
  eta = 0.3,
  gamma = 0.25,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 0.5,
  objective = "binary:logistic"
)
xgboost_probabilities <- predict(xgboost_model, as.matrix(validationdata[, -ncol(validationdata)]))
xgboost_predictions <- ifelse(xgboost_probabilities > 0.5, 1, 0)
xgboost_metrics <- calculate_metrics(xgboost_predictions, validationdata$label, xgboost_probabilities)
validation_labels <- as.numeric(validationdata$label)  # 转换为数值
validation_labels <- ifelse(validation_labels == 0, 1, 2)  # 将 0 转换为 1，1 转换为 2
calculate_auprc <- function(probabilities, true_labels, n_boot = 1000) {
  true_labels <- as.numeric(true_labels)
  true_labels <- ifelse(true_labels == 1, 1, 2)  # 确保正类是 1，负类是 2
  
  # Calculate AUPRC
  pr_curve <- pr.curve(scores.class0 = probabilities, weights.class0 = true_labels - 1)  # 注意：weights.class0 需要是非负的
  auprc <- pr_curve$auc.integral
  
  # 
  auprc_boot <- numeric(n_boot)
  for (i in 1:n_boot) {
    boot_idx <- sample(1:length(probabilities), replace = TRUE)
    boot_pr <- pr.curve(scores.class0 = probabilities[boot_idx], weights.class0 = true_labels[boot_idx] - 1)
    auprc_boot[i] <- boot_pr$auc.integral
  }
  auprc_ci <- quantile(auprc_boot, c(0.025, 0.975))
  
  return(list(auprc = auprc, auprc_ci = auprc_ci))
}
# AUPRC
xgboost_auprc <- calculate_auprc(xgboost_probabilities, validation_labels)
cat("XGBoost AUPRC:", xgboost_auprc$auprc, "95% CI:", xgboost_auprc$auprc_ci, "\n")


# ann model
ann_model <- nnet(label ~ ., data = traindata, size = 5, decay = 0.2, maxit = 100)
ann_probabilities <- predict(ann_model, newdata = validationdata, type = "raw")
validationdata$label <- as.factor(validationdata$label)
print(levels(validationdata$label))  
ann_predictions <- ifelse(ann_probabilities > 0.5, levels(validationdata$label)[2], levels(validationdata$label)[1])
print(head(ann_predictions))
ann_probabilities <- as.numeric(ann_probabilities)
calculate_metrics <- function(predictions, true_labels, probabilities = NULL) {
  conf_matrix <- table(predictions, true_labels)
  TP <- conf_matrix[1, 1]
  FP <- conf_matrix[1, 2]
  TN <- conf_matrix[2, 2]
  FN <- conf_matrix[2, 1]
  
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision <- TP / (TP + FP)
  recall <- sensitivity
  f1_score <- 2 * (precision * recall) / (precision + recall)
  npv <- TN / (TN + FN)
  
  # Calculate AUC
  auc <- NA
  if (!is.null(probabilities)) {
    probabilities <- as.numeric(probabilities)
    roc_curve <- roc(true_labels, probabilities)
    auc <- auc(roc_curve)
  }
  
  
  metrics_ci <- list(
    accuracy_ci = BinomCI(TP + TN, TP + TN + FP + FN, conf.level = 0.95),
    sensitivity_ci = BinomCI(TP, TP + FN, conf.level = 0.95),
    specificity_ci = BinomCI(TN, TN + FP, conf.level = 0.95),
    precision_ci = BinomCI(TP, TP + FP, conf.level = 0.95),
    npv_ci = BinomCI(TN, TN + FN, conf.level = 0.95)
  )
  
  return(list(
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    npv = npv,
    auc = auc,
    metrics_ci = metrics_ci
  ))
}
ann_metrics <- calculate_metrics(ann_predictions, validationdata$label, ann_probabilities)
ann_auprc <- calculate_auprc(ann_probabilities, validationdata$label)




print(ann_metrics)
cat("AUPRC:", ann_auprc$auprc, "95% CI:", ann_auprc$auprc_ci, "\n\n")

print(svm_metrics)
cat("AUPRC:", svm_auprc$auprc, "95% CI:", svm_auprc$auprc_ci, "\n\n")

print(gbdt_metrics)
cat("AUPRC:", gbdt_auprc$auprc, "95% CI:", gbdt_auprc$auprc_ci, "\n\n")

print(xgboost_metrics)
cat("AUPRC:", xgboost_auprc$auprc, "95% CI:", xgboost_auprc$auprc_ci, "\n\n")

print(rf_metrics)
cat("AUPRC:", rf_auprc$auprc, "95% CI:", rf_auprc$auprc_ci, "\n\n")





# Calculate the AUC confidence interval for each model.
ann_auc_ci <- ci.auc(roc(validationdata$label, ann_probabilities));ann_auc_ci
svm_auc_ci <- ci.auc(roc(validationdata$label, svm_probabilities));svm_auc_ci
gbdt_auc_ci <- ci.auc(roc(validationdata$label, gbdt_probabilities));gbdt_auc_ci
xgboost_auc_ci <- ci.auc(roc(validationdata$label, xgboost_probabilities));xgboost_auc_ci
rf_auc_ci <- ci.auc(roc(validationdata$label, rf_probabilities));rf_auc_ci


library(showtext)

font_add("Arial", "arial.ttf")  
showtext_auto()

# Define colors and line types
colors <- c("ANN" = "#37AB78", "SVM" = "#F3B169","GBDT" = "#808080",  "XGBoost" = "#589FF3", "RF" = "#941414")
linetypes <- c("ANN" = "dotted", "SVM" = "dashed",  "XGBoost" = "dotdash", "GBDT" = "longdash", "RF" = "solid")



legend_labels <- c(
  paste0("ANN: ", round(ann_metrics$auc, 3), " (", round(ann_auc_ci[1], 3), ",", round(ann_auc_ci[3], 3), ")"),
  paste0("SVM: ", round(svm_metrics$auc, 3), " (", round(svm_auc_ci[1], 3), ",", round(svm_auc_ci[3], 3), ")"),
  paste0("GBDT: ", round(gbdt_metrics$auc, 3), " (", round(gbdt_auc_ci[1], 3), ",", round(gbdt_auc_ci[3], 3), ")"),
  paste0("XGBoost: ", round(xgboost_metrics$auc, 3), " (", round(xgboost_auc_ci[1], 3), ",", round(xgboost_auc_ci[3], 3), ")"),
  paste0("RF: ", round(rf_metrics$auc, 3), " (", round(rf_auc_ci[1], 3), ",", round(rf_auc_ci[3], 3), ")")
)


print(table(validationdata$label))

# Plot ROC curve
p <- ggroc(list(
  ANN = roc(validationdata$label, ann_probabilities),
  SVM = roc(validationdata$label, svm_probabilities),
  XGBoost = roc(validationdata$label, xgboost_probabilities),
  GBDT = roc(validationdata$label, gbdt_probabilities),
  RF = roc(validationdata$label, rf_probabilities)
), legacy.axes = TRUE, size = 1) +
  xlab("False-positive rate") +
  ylab("True-positive rate") +
  scale_x_continuous(expand = c(0, 0)) + 
  scale_y_continuous(
    expand = c(0, 0),  
    labels = function(x) ifelse(x == 0, "", sprintf("%.2f", x))  
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +  
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", size = 0.5) +  
  scale_color_manual(values = colors, breaks = c("ANN",  "SVM", "GBDT","XGBoost", "RF"), labels = legend_labels) + 
  scale_linetype_manual(values = linetypes, breaks = c("ANN",  "SVM", "GBDT","XGBoost", "RF"), labels = legend_labels) +  
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, face = "bold", family = "Times New Roman", hjust = 0.5),
    axis.title.y = element_text(size = 12, family = "Times New Roman", margin = unit(c(0, 10, 0, 0), "pt")),  
    axis.title.x = element_text(size = 12, family = "Times New Roman", margin = unit(c(10, 0, 0, 0), "pt")),  
    legend.title = element_blank(),
    legend.text = element_text(size = 10, family = "Times New Roman"),
    legend.position = c(0.83, 0.05),  
    legend.justification = c(1, 0),   
    legend.direction = "vertical",    
    panel.grid = element_blank(),     
    axis.line = element_line(colour = "black"),  
    panel.border = element_blank(),   
    axis.ticks = element_line(colour = "black"), 
    axis.line.x.top = element_blank(), 
    axis.line.y.right = element_blank(), 
    plot.margin = unit(c(1, 1, 1, 1), "cm")  
  );p




#Plot the AUPRC curve
library(ggplot2)
library(pROC)
library(PRROC)
library(boot)


validationdata$label <- as.numeric(as.character(validationdata$label))

colors <- c("ANN" = "#37AB78", "SVM" = "#F3B169","GBDT" = "#808080", "XGBoost" = "#589FF3", "RF" = "#941414")
linetypes <- c("ANN" = "solid", "SVM" = "solid", "GBDT" = "solid", "XGBoost" = "solid", "RF" = "solid")

legend_labels <- c(
  paste0("ANN: ", round(ann_auprc$auprc, 3), " (", round(ann_auprc$auprc_ci[1], 3), ",", round(ann_auprc$auprc_ci[3], 3), ")"),
  paste0("SVM: ", round(svm_auprc$auprc, 3), " (", round(svm_auprc$auprc_ci[1], 3), ",", round(svm_auprc$auprc_ci[3], 3), ")"),
  paste0("GBDT: ", round(gbdt_auprc$auprc, 3), " (", round(gbdt_auprc$auprc_ci[1], 3), ",", round(gbdt_auprc$auprc_ci[3], 3), ")"),
  paste0("XGBoost: ", round(xgboost_auprc$auprc, 3), " (", round(xgboost_auprc$auprc_ci[1], 3), ",", round(xgboost_auprc$auprc_ci[3], 3), ")"),
  paste0("RF: ", round(rf_auprc$auprc, 3), " (", round(rf_auprc$auprc_ci[1], 3), ",", round(rf_auprc$auprc_ci[3], 3), ")")
)

ann_pr <- pr.curve(scores.class0 = ann_probabilities, weights.class0 = validationdata$label, curve = TRUE)
svm_pr <- pr.curve(scores.class0 = svm_probabilities, weights.class0 = validationdata$label, curve = TRUE)
gbdt_pr <- pr.curve(scores.class0 = gbdt_probabilities, weights.class0 = validationdata$label, curve = TRUE)
xgboost_pr <- pr.curve(scores.class0 = xgboost_probabilities, weights.class0 = validationdata$label, curve = TRUE)
rf_pr <- pr.curve(scores.class0 = rf_probabilities, weights.class0 = validationdata$label, curve = TRUE)

pr_data <- rbind(
  data.frame(recall = ann_pr$curve[,1], precision = ann_pr$curve[,2], model = "ANN"),
  data.frame(recall = svm_pr$curve[,1], precision = svm_pr$curve[,2], model = "SVM"),
  data.frame(recall = gbdt_pr$curve[,1], precision = gbdt_pr$curve[,2], model = "GBDT"),
  data.frame(recall = xgboost_pr$curve[,1], precision = xgboost_pr$curve[,2], model = "XGBoost"),
  data.frame(recall = rf_pr$curve[,1], precision = rf_pr$curve[,2], model = "RF")
)

calculate_auprc_ci <- function(scores, labels, n_boot = 50) {
  pr_original <- pr.curve(scores.class0 = scores, weights.class0 = labels)
  auprc <- pr_original$auc.integral
  
  
  boot_results <- boot::boot(
    data = data.frame(scores = scores, labels = labels),
    statistic = function(data, indices) {
      pr <- pr.curve(scores.class0 = data$scores[indices], 
                     weights.class0 = data$labels[indices])
      return(pr$auc.integral)
    },
    R = n_boot
  )
  
  # 
  ci <- boot::boot.ci(boot_results, type = "perc")
  ci_lower <- ci$percent[4]
  ci_upper <- ci$percent[5]
  
  return(list(auprc = auprc, auprc_ci = c(ci_lower, auprc, ci_upper)))
}

# 
set.seed(123)
ann_auprc <- calculate_auprc_ci(ann_probabilities, validationdata$label, n_boot = 50)
svm_auprc <- calculate_auprc_ci(svm_probabilities, validationdata$label, n_boot = 50)
gbdt_auprc <- calculate_auprc_ci(gbdt_probabilities, validationdata$label, n_boot = 50)
xgboost_auprc <- calculate_auprc_ci(xgboost_probabilities, validationdata$label, n_boot = 50)
rf_auprc <- calculate_auprc_ci(rf_probabilities, validationdata$label, n_boot = 50)

legend_labels <- c(
  paste0("ANN: ", round(ann_auprc$auprc, 3), " (", round(ann_auprc$auprc_ci[1], 3), ",", round(ann_auprc$auprc_ci[3], 3), ")"),
  paste0("SVM: ", round(svm_auprc$auprc, 3), " (", round(svm_auprc$auprc_ci[1], 3), ",", round(svm_auprc$auprc_ci[3], 3), ")"),
  paste0("GBDT: ", round(gbdt_auprc$auprc, 3), " (", round(gbdt_auprc$auprc_ci[1], 3), ",", round(gbdt_auprc$auprc_ci[3], 3), ")"),
  paste0("XGBoost: ", round(xgboost_auprc$auprc, 3), " (", round(xgboost_auprc$auprc_ci[1], 3), ",", round(xgboost_auprc$auprc_ci[3], 3), ")"),
  paste0("RF: ", round(rf_auprc$auprc, 3), " (", round(rf_auprc$auprc_ci[1], 3), ",", round(rf_auprc$auprc_ci[3], 3), ")")
)


calculate_pr_ci <- function(scores, labels, n_boot = 50) {
  data <- data.frame(scores = scores, labels = labels)
  recall_points <- seq(0, 1, length.out = 100)
  calc_pr <- function(data, indices) {
    pr <- pr.curve(scores.class0 = data$scores[indices], 
                   weights.class0 = data$labels[indices],
                   curve = TRUE)
    if (is.null(pr$curve)) return(rep(NA, length(recall_points)))
        approx_pr <- approx(pr$curve[,1], pr$curve[,2], xout = recall_points, 
                        rule = 2, ties = mean)$y
    return(approx_pr)
  }
  set.seed(123)
  boot_results <- boot(data, statistic = calc_pr, R = n_boot)
  ci_data <- data.frame(
    recall = recall_points,
    ci_lower = apply(boot_results$t, 2, function(x) quantile(x, 0.025, na.rm = TRUE)),
    ci_upper = apply(boot_results$t, 2, function(x) quantile(x, 0.975, na.rm = TRUE))
  )
  
  return(ci_data)
}

calculate_pr_ci <- function(scores, labels, n_boot = 50) {
  data <- data.frame(scores = scores, labels = labels)
  if (any(data$labels < 0)) {
    stop("Labels vectors cannot contain negative numbers.")
  }
  recall_points <- seq(0, 1, length.out = 100)
  calc_pr <- function(data, indices) {
    pr <- pr.curve(scores.class0 = data$scores[indices], 
                   weights.class0 = data$labels[indices],
                   curve = TRUE)
        if (is.null(pr$curve)) return(rep(NA, length(recall_points)))
        approx_pr <- approx(pr$curve[,1], pr$curve[,2], xout = recall_points, 
                        rule = 2, ties = mean)$y
    
    return(approx_pr)
  }
  
    set.seed(123)
  boot_results <- boot(data, statistic = calc_pr, R = n_boot)
  
  # 计算置信区间
  ci_data <- data.frame(
    recall = recall_points,
    ci_lower = apply(boot_results$t, 2, function(x) quantile(x, 0.025, na.rm = TRUE)),
    ci_upper = apply(boot_results$t, 2, function(x) quantile(x, 0.975, na.rm = TRUE))
  )
  
  return(ci_data)
}

ann_ci <- calculate_pr_ci(ann_probabilities, validationdata$label)
svm_ci <- calculate_pr_ci(svm_probabilities, validationdata$label)
gbdt_ci <- calculate_pr_ci(gbdt_probabilities, validationdata$label)
xgboost_ci <- calculate_pr_ci(xgboost_probabilities, validationdata$label)
rf_ci <- calculate_pr_ci(rf_probabilities, validationdata$label)


ci_data <- rbind(
  data.frame(recall = ann_ci$recall, ci_lower = ann_ci$ci_lower, ci_upper = ann_ci$ci_upper, model = "ANN"),
  data.frame(recall = svm_ci$recall, ci_lower = svm_ci$ci_lower, ci_upper = svm_ci$ci_upper, model = "SVM"),
  data.frame(recall = gbdt_ci$recall, ci_lower = gbdt_ci$ci_lower, ci_upper = gbdt_ci$ci_upper, model = "GBDT"),
  data.frame(recall = xgboost_ci$recall, ci_lower = xgboost_ci$ci_lower, ci_upper = xgboost_ci$ci_upper, model = "XGBoost"),
  data.frame(recall = rf_ci$recall, ci_lower = rf_ci$ci_lower, ci_upper = rf_ci$ci_upper, model = "RF")
)


p <- ggplot() +
  geom_ribbon(data = ci_data, 
              aes(x = recall, ymin = ci_lower, ymax = ci_upper, fill = model),
              alpha = 0.2, show.legend = FALSE) +
  geom_line(data = pr_data, 
            aes(x = recall, y = precision, color = model, linetype = model),
            size = 1) +
  xlab("Recall (sensitivity)") +
  ylab("Precision (PPV)") +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(
    expand = c(0, 0),
    labels = function(x) ifelse(x == 0, "", sprintf("%.2f", x))
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_color_manual(values = colors, breaks = c("ANN", "SVM", "GBDT", "XGBoost", "RF"), labels = legend_labels) +
  scale_linetype_manual(values = linetypes, breaks = c("ANN", "SVM", "GBDT", "XGBoost", "RF"), labels = legend_labels) +
  scale_fill_manual(values = colors) +  
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, face = "bold", family = "Arial", hjust = 0.5),
    axis.title.y = element_text(size = 12, family = "Arial", margin = unit(c(0, 10, 0, 0), "pt")),
    axis.title.x = element_text(size = 12, family = "Arial", margin = unit(c(10, 0, 0, 0), "pt")),
    axis.text = element_text(size = 10, family = "Arial"),
    legend.title = element_blank(),
    legend.text = element_text(size = 10, family = "Arial"),
    legend.position = c(0.83, 0.00),
    legend.justification = c(1, 0),
    legend.direction = "vertical",
    panel.grid = element_blank(),
    axis.line = element_line(colour = "black"),
    panel.border = element_blank(),
    axis.ticks = element_line(colour = "black"),
    axis.line.x.top = element_blank(),
    axis.line.y.right = element_blank(),
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    legend.background = element_blank() 
  )

print(p)

#Draw calibration curves

models <- list(
  "ANN" = ann_model,
  "SVM" = svm_model,
  "GBDT" = gbdt_model,
  "XGBoost" = xgboost_model,
  "RF" = rf_model
)
# 
colors <- c("ANN" = "#37AB78", "SVM" = "#F3B169", "GBDT" = "#808080", "XGBoost" = "#589FF3", "RF" = "#941414")
linetypes <- c("ANN" = "dotted", "SVM" = "dashed", "XGBoost" = "dotdash", "GBDT" = "longdash", "RF" = "solid")

# Draw calibration curves
par(mfrow = c(1, 1))
plot(0:1, 0:1, type = "l", lty = 2, xlab = "Mean predicted probability", ylab = "Observed probability",
     main = NA)
for (model_name in names(models)) {
  model <- models[[model_name]]
  
  if (model_name == "RF") {
    probs <- rf_probabilities
  } else if (model_name == "SVM") {
    probs <- svm_probabilities
  } else if (model_name == "ANN") {
    probs <- ann_probabilities
  } else if (model_name == "XGBoost") {
    probs <- xgboost_probabilities
  } else if (model_name == "GBDT") {
    probs <- gbdt_probabilities
  }
  
  bins <- cut(probs, breaks = seq(0, 1, length.out = 11), include.lowest = TRUE)
  bin_means <- tapply(probs, bins, mean)
  bin_obs <- tapply(as.numeric(validationdata$label) - 1, bins, mean)
  
  lines(bin_means, bin_obs, type = "b", pch = 16, 
        col = colors[model_name],  
        lty = linetypes[model_name])  
}

current_cex <- par("cex")
legend(x = 0.80, y = 0.35, legend = names(models), 
       col = colors[names(models)],  
       lty = linetypes[names(models)], 
       pch = 16, bty = "n",
       cex = 0.8)

#decision curve
calculate_net_benefit <- function(probs, labels, threshold) {
  preds <- ifelse(probs >= threshold, 1, 0)
  tp <- sum(preds == 1 & labels == 1)
  fp <- sum(preds == 1 & labels == 0)
  n <- length(labels)
  net_benefit <- tp / n - fp / n * (threshold / (1 - threshold))
  return(net_benefit)
}

ann_probabilities <- predict(ann_model, newdata = validationdata, type = "raw")
svm_probabilities <- attr(predict(svm_model, newdata = validationdata, probability = TRUE), "probabilities")[, 2]
gbdt_probabilities <- predict(gbdt_model, newdata = validationdata[, -which(names(validationdata) == "label")], type = "response")
xgboost_probabilities <- predict(xgboost_model, as.matrix(validationdata[, -ncol(validationdata)]))
rf_probabilities <- predict(rf_model, newdata = validationdata, type = "prob")[, 2]

thresholds <- seq(0, 1, by = 0.01)

ann_net_benefit <- sapply(thresholds, function(p) calculate_net_benefit(ann_probabilities, validationdata$label, p))
svm_net_benefit <- sapply(thresholds, function(p) calculate_net_benefit(svm_probabilities, validationdata$label, p))
xgb_net_benefit <- sapply(thresholds, function(p) calculate_net_benefit(xgboost_probabilities, validationdata$label, p))
gbdt_net_benefit <- sapply(thresholds, function(p) calculate_net_benefit(gbdt_probabilities, validationdata$label, p))
rf_net_benefit <- sapply(thresholds, function(p) calculate_net_benefit(rf_probabilities, validationdata$label, p))

all_net_benefit <- sapply(thresholds, function(p) {
  tp <- sum(validationdata$label == 1)
  fp <- sum(validationdata$label == 0)
  n <- length(validationdata$label)
  net_benefit <- tp / n - fp / n * (p / (1 - p))
  return(net_benefit)
})

none_net_benefit <- rep(0, length(thresholds))

nb <- data.frame(threshold = thresholds)
nb$ann <- ann_net_benefit
nb$svm <- svm_net_benefit
nb$gbdt <- gbdt_net_benefit
nb$xgb <- xgb_net_benefit
nb$rf <- rf_net_benefit
nb$all <- all_net_benefit
nb$none <- none_net_benefit

smooth <- TRUE
if (smooth) {
  require(stats)
  lws_ann <- loess(data.matrix(nb[!is.na(nb$ann), "ann"]) ~ data.matrix(nb[!is.na(nb$ann), "threshold"]), span = 0.10)
  nb[!is.na(nb$ann), "ann_sm"] <- lws_ann$fitted
  lws_svm <- loess(data.matrix(nb[!is.na(nb$svm), "svm"]) ~ data.matrix(nb[!is.na(nb$svm), "threshold"]), span = 0.10)
  nb[!is.na(nb$svm), "svm_sm"] <- lws_svm$fitted
  lws_xgb <- loess(data.matrix(nb[!is.na(nb$xgb), "xgb"]) ~ data.matrix(nb[!is.na(nb$xgb), "threshold"]), span = 0.10)
  nb[!is.na(nb$xgb), "xgb_sm"] <- lws_xgb$fitted
  lws_gbdt <- loess(data.matrix(nb[!is.na(nb$gbdt), "gbdt"]) ~ data.matrix(nb[!is.na(nb$gbdt), "threshold"]), span = 0.10)
  nb[!is.na(nb$gbdt), "gbdt_sm"] <- lws_gbdt$fitted
  lws_rf <- loess(data.matrix(nb[!is.na(nb$rf), "rf"]) ~ data.matrix(nb[!is.na(nb$rf), "threshold"]), span = 0.10)
  nb[!is.na(nb$rf), "rf_sm"] <- lws_rf$fitted
}

require(graphics)
par(mar = c(5, 5, 4, 2) + 0.1)  
legendlabel <- c("None", "All", "ANN", "SVM", "GBDT", "XGBoost",  "RF")
legendcolor <- c('black', '#293890', '#37AB78', '#F3B169', '#808080', '#589FF3', '#941414')
legendwidth <- rep(2, length(legendlabel))  
legendpattern <- rep(1, length(legendlabel))  
ymax <- max(nb[names(nb) != "threshold"], na.rm = TRUE) + 0.1
ymin <- -0.05
plot(x = nb$threshold, y = nb$all, type = "l", col = "#293890", lwd = 2, xlim = c(0, 1), ylim = c(ymin, ymax),
     xlab = "Threshold Probability", ylab = "Net Benefit", 
     main = NA,  # 主标题"Decision Curve Analysis for Multiple Models"
     cex.lab = 1, cex.axis = 1, cex.main = 1, font.lab = 1, font.axis = 1, font.main = 1) 
lines(x = nb$threshold, y = nb$none, lwd = 2, col = "black")
if (smooth) {
  lines(nb$threshold, data.matrix(nb$ann_sm), col = "#37AB78", lty = 1, lwd = 2)
  lines(nb$threshold, data.matrix(nb$svm_sm), col = "#F3B169", lty = 1, lwd = 2)
  lines(nb$threshold, data.matrix(nb$gbdt_sm), col = "#808080", lty = 1, lwd = 2)
  lines(nb$threshold, data.matrix(nb$xgb_sm), col = "#589FF3", lty = 1, lwd = 2)
  lines(nb$threshold, data.matrix(nb$rf_sm), col = "#941414", lty = 1, lwd = 2)
} else {
  lines(nb$threshold, data.matrix(nb$ann), col = "#37AB78", lty = 1, lwd = 2)
  lines(nb$threshold, data.matrix(nb$svm), col = "#F3B169", lty = 1, lwd = 2)
  lines(nb$threshold, data.matrix(nb$gbdt), col = "#808080", lty = 1, lwd = 2)
  lines(nb$threshold, data.matrix(nb$xgb), col = "#589FF3", lty = 1, lwd = 2)
  lines(nb$threshold, data.matrix(nb$rf), col = "#941414", lty = 1, lwd = 2)
}
current_cex <- par("cex")
legend(x = 0.77, y = 0.56, legendlabel, cex = 0.8, col = legendcolor, lwd = legendwidth, lty = legendpattern, 
       bty = "n",  
       text.font = 1,
       y.intersp = 0.8
) 














######保存训练好的模型进行外部验证
# 保存所有模型到一个文件
save(rf_model, svm_model, xgboost_model, ann_model, gbdt_model,file = "E:/WLP/WHU/article-idea/article/code/5-16all_models.RData")



