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

input_xlsx <- file.path(base_dir, "z_peak_fre_RT_long.xlsx")

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
emm <- emmeans(m_peak, ~ condition)
pairs_res <- pairs(emm, adjust = "tukey")

cat("\n=== LMM summary ===\n"); print(summary(m_peak))
cat("\n=== Omnibus test (condition) ===\n"); print(anova_res)

# ------------------------------------------------------------
# GLMM: RT_beta ~ peak_z + (1 + peak_z | sub_id)
# ------------------------------------------------------------
dat_rt <- read_excel(input_xlsx)
dat_rt$sub_id <- factor(dat_rt$sub_id)

dat_rt$peak_z <- dat_rt$peak_z
dat_rt$RT_beta <- dat_rt$RT_mean

m_rt <- glmmTMB(
  RT_beta ~ peak_z + (1 + peak_z | sub_id),
  family = beta_family(link = "logit"),
  data = dat_rt
)

cat("\n=== GLMM summary ===\n"); print(summary(m_rt))
