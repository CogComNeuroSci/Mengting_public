library(readxl)
library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(rstudioapi)

base_dir <- normalizePath(
  file.path(dirname(rstudioapi::getActiveDocumentContext()$path),
            "..", "Experiment_1_data", "results"),
  winslash = "/",
  mustWork = FALSE
)
input_xlsx <- file.path(base_dir, "z_peak_fre_acc_long_e1.xlsx")

dat <- readxl::read_excel(input_xlsx)

dat <- dat %>%
  mutate(
    sub_id = factor(sub_id),
    hemi   = factor(hemi, levels = c("L","R")),
    hand   = factor(hand, levels = c("L","R"))
  )

# --- Type-III ANOVA ---
options(contrasts = c("contr.sum","contr.poly"))

# --- LMM: full 2×2 (hemi × hand) ---
m1 <- lmer(peak_z ~ hemi * hand + (1 | sub_id), data = dat)
cat("==== LMM (hemi × hand) summary ====")
print(summary(m1))
cat("==== Type-III ANOVA (hemi × hand) ====")
print(anova(m1, type = 3))

# --- LMM: Congruency (LL/RR vs LR/RL) ---
dat <- dat %>% mutate(
  congruency = factor(ifelse(hemi == hand, "Congruent", "Incongruent"),
                      levels = c("Incongruent","Congruent"))
)

m2 <- lmer(peak_z ~ congruency + (1 | sub_id), data = dat)
cat("==== LMM (Congruency) summary ====")
print(summary(m2))
cat("==== Type-III ANOVA (Congruency) ====")
print(anova(m2, type = 3))
