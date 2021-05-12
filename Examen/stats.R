library(tidyverse)

data  <- read_csv('export.csv')

data %>%
    summarise(
        media = mean(Final),
        sd = sd(Final)
    )

data %>%
    select(Final) %>%
    ggplot(aes(Final))+
    geom_histogram(bins=15)

data %>%
    select(P1) %>%
    ggplot(aes(P1))+
    geom_histogram(bins=10)

data %>%
    select(P2) %>%
    ggplot(aes(P2))+
    geom_histogram(bins=10)

data %>%
    select(P3) %>%
    ggplot(aes(P3))+
    geom_histogram(bins=10)
