library(readxl)
library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(rstudioapi)
library(glmmTMB)
library(ggeffects)
library(ggplot2)

base_dir <- normalizePath(
  file.path(
    dirname(rstudioapi::getActiveDocumentContext()$path),
    "..", "Experiment_2_data", "results"
  ),
  winslash = "/",
  mustWork = FALSE
)

input_xlsx <- file.path(base_dir, "z_peak_fre_acc_long.xlsx")

# for LMM
options(contrasts = c("contr.sum", "contr.poly"))

# ------------------------------------------------------------
# LMM: peak_z ~ condition + (1 | subject)
# ------------------------------------------------------------
dat_peak <- read_xlsx(input_xlsx, sheet = "z_peak_long") %>%
  mutate(
    subject = factor(subject),
    condition = factor(condition, levels = c("SS", "SSiDS", "DSi"))
  )

m_peak <- lmer(peak_z ~ condition + (1 | subject), data = dat_peak, REML = FALSE)
anova_res <- anova(m_peak)

cat("\n=== LMM summary ===\n"); print(summary(m_peak))
cat("\n=== LMM omnibus test (condition) ===\n"); print(anova_res)

# ------------------------------------------------------------
# GLMM: acc_beta ~ peak_z + (1 + peak_z | sub_id)
# ------------------------------------------------------------
dat_acc <- read_excel(input_xlsx)
dat_acc$sub_id <- factor(dat_acc$sub_id)

dat_acc$peak_z <- dat_acc$peak_z
dat_acc$acc_beta <- dat_acc$acc_mean

m_acc <- glmmTMB(
  acc_beta ~ peak_z + (1 + peak_z | sub_id),
  family = beta_family(link = "logit"),
  data = dat_acc
)

cat("\n=== GLMM summary ===\n"); print(summary(m_acc))
