# Main file
# Florian UNGER
# 07/03/2019
# Amazon Web Scraping

### Packages 

pacman::p_load(plot3Drgl, rgl, car, ggplot2,
               plotly, rstudioapi, corrplot, 
               rgl, manipulateWidget, reshape, 
               reshape2, Rfast, randomForest, esquisse)

### Github

current_path = rstudioapi::getActiveDocumentContext()$path #save working directory
setwd(dirname(current_path))
setwd("..")








