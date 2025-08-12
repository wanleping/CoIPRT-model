
#1. Prepare models and dependencies
#install.packages(c("shiny", "plumber", "dplyr", "caret"))
library(plumber)
library(caret)
library(shiny)
library(httr)
library(jsonlite)
library(shiny)
#2. Create Plumber API service 

# Load the trained model
model <- readRDS("E:/WLP/WHU/article-idea/article/code/CoIPRT_model.rds")


# UI 
ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("Cognitive Impairment Population Risk Tool (CoIPRT)", windowTitle = "Cognitive impairment prediction"),
  
  sidebarLayout(
    sidebarPanel(
      width = 4,
      selectInput("Gender", "Gender", choices = c("male" = "1", "female" = "2")),
      sliderInput("Age", "Age", value = 65, min = 45, max = 120),
      selectInput("Education", "Education", choices = c("Primary" = "0", "Secondary" = "1", "Tertiary" = "2")),
      selectInput("Work_status", "Work_status", choices = c("Not working" = "0", "Working" = "1", "Retired" = "2")),
      sliderInput("Body_mass_index", "Body_mass_index", min = 0, max = 40, value = 24, step = 0.1),
      selectInput("Drink", "Drink", choices = c("Yes" = "1", "No" = "0")),
      selectInput("Social_activities", "Social_activities", choices = c("Active" = "1", "Inactive" = "0")),
      selectInput("Self_report_of_health", "Self_report_of_health", choices = c("poor or fair" = "0", "good" = "1", "very good or excellent" = "2")),
      selectInput("Depression", "Depression", choices = c("Yes" = "1", "No" = "0")),
      selectInput("Self_reported_memory", "Self_reported_memory", choices = c("poor or fair" = "0", "good" = "1", "very good or excellent" = "2")),
      sliderInput("Living_siblings", "Living_siblings", min = 0, max = 10, value = 1),
      selectInput("diabetes", "diabetes", choices = c("Yes" = "1", "No" = "0")),
      selectInput("stroke", "stroke", choices = c("Yes" = "1", "No" = "0")),
      selectInput("heart_problem", "heart_problem", choices = c("Yes" = "1", "No" = "0")),
      selectInput("chronic_lung", "chronic_lung", choices = c("Yes" = "1", "No" = "0")),
      selectInput("cancer", "cancer", choices = c("Yes" = "1", "No" = "0")),
      actionButton("predict", "Start", class = "btn-primary btn-block"),
      br(), br(),
      wellPanel(
        h5("Operation Manual"),
        p("1. Completion of participant indicators"),
        p("2. Click on the ‘Start’ button"),
        p("3. View predictions on the right"),
        p("#Disclaimer: This system is only used as a clinical auxiliary tool and cannot replace professional medical judgment")
      )
    ),
    
    mainPanel(
      width = 8,
      tabsetPanel(
        tabPanel(
          "Predictions",
          br(),
          fluidRow(
            column(
              6,
              wellPanel(
                h4("Prediction"),
                div(style = "font-size: 24px; color: #2c3e50;", 
                    textOutput("prediction")),
                br(),
                plotOutput("prob_plot", height = "200px")
              )
            ),
            column(
              6,
              wellPanel(
                h4("Clinical Recommendations"),
                uiOutput("recommendation")
              )
            )
          ),
          fluidRow(
            column(
              12,
              wellPanel(
                h4("Importance of the impact of each factor"),
                plotOutput("importance_plot", height = "300px")
              )
            )
          )
        ),
        
        tabPanel(
          "About the System",
          includeMarkdown("about.md")
        )
      )
    )
  )
)

# server logic
server <- function(input, output) {
  prediction <- eventReactive(input$predict, {
    new_data <- data.frame(
      Gender = input$Gender,
      Age = input$Age,
      Education = input$Education,
      Work_status = input$Work_status,
      Body_mass_index = input$Body_mass_index,
      Drink = input$Drink,
      Social_activities = input$Social_activities,
      Self_report_of_health = input$Self_report_of_health,
      Depression = input$Depression,
      Self_reported_memory = input$Self_reported_memory,
      Living_siblings = input$Living_siblings,
      diabetes = input$diabetes,
      stroke = input$stroke,
      heart_problem = input$heart_problem,
      chronic_lung = input$chronic_lung,
      cancer = input$cancer
    )
    
    pred_prob <- predict(model, newdata = new_data, type = "prob")[, 2]
    class_pred <- ifelse(pred_prob > 0.5, "Cognitive Impairment", "Cognitive Normal")
    
    list(
      class = class_pred,
      prob = pred_prob
    )
  })
  
  output$prediction <- renderText({
    prob <- prediction()$prob
    paste("The predicted risk of cognitive impairment is: ", sprintf("%.2f%%", prob * 100))
  })
  
  output$recommendation <- renderUI({
    pred <- prediction()$class
    if (pred == "Cognitive Normal") {
      tagList(
        p(style = "color: #27ae60;", "✔ Cognitive Normal"),
        p("Recommendation:"),
        tags$ul(
          tags$li("Follow-up examination"),
          tags$li("Maintain a Healthy Lifestyle")
        )
      )
    } else {
      tagList(
        p(style = "color: #e74c3c;", "❗ Cognitive Impairment"),
        p("Recommendation:"),
        tags$ul(
          tags$li("Immediate referral to a specialist"),
          tags$li("Family nursing guidance"),
          tags$li("Monthly follow-up evaluation")
        )
      )
    }
  })
  
  output$importance_plot <- renderPlot({
    if (!is.null(model) && inherits(model, "randomForest")) {
      importance_scores <- importance(model)
      importance_gini <- importance(model, type = 2)
      importance_accuracy <- importance(model, type = 1)
      feature_importance <- data.frame(Variable = rownames(importance_scores), Importance = importance_scores[, "MeanDecreaseGini"])
      ordered_features <- feature_importance[order(-feature_importance$Importance), ]
      feature_importance_gini <- data.frame(Variable = rownames(importance_gini), Importance = importance_gini[, "MeanDecreaseGini"])
      ordered_features_gini <- feature_importance_gini[order(-feature_importance_gini$Importance), ]
      feature_importance_accuracy <- data.frame(Variable = rownames(importance_accuracy), Importance = importance_accuracy[, "MeanDecreaseAccuracy"])
      ordered_features_accuracy <- feature_importance_accuracy[order(-feature_importance_accuracy$Importance), ]
      top_gini <- head(ordered_features_gini, 10)
      top_accuracy <- head(ordered_features_accuracy, 10)
      color_range <- c("#D8BFD8", "#8B008B")
      plot_accuracy <- ggplot(top_accuracy, aes(x = reorder(Variable, Importance), y = Importance, fill = Importance, label = round(Importance, 2))) + 
        geom_bar(stat = "identity") +
        scale_fill_gradient(low = color_range[1], high = color_range[2]) + 
        theme_few() + 
        labs(title = "Top 10 Important Variable in Cognitive Impairment Prediction", x = NULL, y = "Importance") + 
        coord_flip() 

      library(gridExtra)
      grid.arrange(plot_accuracy, ncol = 1)
    } else {
      plot.new()
      text(0.5, 0.5, "Can't get variable significance, check if the model is of type randomForest")
    }
  })
}

shinyApp(ui, server)     
shiny::runApp("E:/WLP/WHU/article-idea/article/code/app.R")

library(rsconnect)
rsconnect::deployApp('E:/WLP/WHU/article-idea/article/code/app.R')
