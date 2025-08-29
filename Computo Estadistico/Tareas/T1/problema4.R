library(ggplot2)

# Data
h_counts <- c(396, 568, 1212, 171, 554, 1104, 257, 435, 295, 397,
              288, 1004, 431, 795, 1621, 1378, 902, 958, 1283, 2415)
noh_counts <- c(375, 375, 752, 208, 151, 116, 736, 192, 315, 1252,
                675, 700, 440, 771, 688, 426, 410, 979, 377, 503)

# Combine data into a single data frame
df <- data.frame(
  group = factor(c(rep("H", length(h_counts)), rep("No-H", length(noh_counts)))),
  counts = c(h_counts, noh_counts)
)

ggplot(data = df) +
  geom_density(aes(x = counts, fill = group), alpha = 0.5) +
  labs(x = "Número de células", y = "Densidad", fill = "Grupos") +
  theme(legend.position="inside",legend.position.inside = c(0.9, 0.9))

boxplot(counts ~ group, data=df, ylab="Número de células", xlab="Grupos")


# Fit the Poisson regression model
df <- data.frame(
  has_h = factor(c(rep(1, length(h_counts)), rep(0, length(noh_counts)))),
  counts = c(h_counts, noh_counts)
)
poisson_model <- glm(counts ~ has_h, data = df, family = poisson)

# Summary of the model
summary(poisson_model)
