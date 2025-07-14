library(corrplot)
library(readr)       # pour lire les CSV
library(dplyr)       # pour la manipulation
library(ggplot2)     # pour les graphiques
library(tidyr)       # pour le nettoyage
library(scales)      # pour la normalisation
library(caret)       # pour preprocessing

data <- read_delim("radiomiques_multislice.csv", delim = ";")

packages <- c("readr", "dplyr", "ggplot2", "tidyr", "scales", "caret", "corrplot")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

invisible(lapply(packages, install_if_missing))
lapply(packages, library, character.only = TRUE)

min_value <- min(data$slice_num, na.rm = TRUE)
max_value <- max(data$slice_num, na.rm = TRUE)

print(min_value)
print(max_value)

# Colonnes numériques uniquement
numeric_cols <- data %>%
  select(where(is.numeric))

data <- data %>%
  mutate(slice_zone = case_when(
    slice_num <= 33 ~ "Bas",
    slice_num <= 66 ~ "Milieu",
    TRUE ~ "Haut"
  ))

# Colonne avec uniquement 





unique_slices <- unique(data$slice_zone)

for (s in unique_slices) {
  # Filtrer les données de la slice
  slice_data <- subset(data, slice_zone == s)

  slice_num_data <- slice_data[sapply(slice_data, is.numeric)]
  slice_num_data <- slice_num_data[, sapply(slice_num_data, function(x) length(unique(na.omit(x))) > 1)]
  
  # Vérifier qu'on a au moins 2 colonnes numériques
  if (ncol(slice_num_data) >= 2) {
    # Calcul de la corrélation
    cor_matrix <- cor(slice_num_data, use = "pairwise.complete.obs")
    
    
    # Sauvegarder l’image
    png(paste0("correlation_slice_", s, ".png"), width = 3000, height = 3000, res = 300)
    corrplot(cor_matrix, method = "color", type = "upper",
             order = "hclust", tl.cex = 0.5, tl.srt = 45)
    dev.off()
  } else {
    message(paste("Slice", s, ": trop peu de colonnes numériques non constantes"))
  }
}

