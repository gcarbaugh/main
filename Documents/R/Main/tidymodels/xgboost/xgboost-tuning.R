# Libraries ----------
library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(ggplot2)
library(tidymodels)
library(doParallel)
library(xgboost)
library(vip)

# Project Directory
setwd('C:/Users/carba/Documents/R/Main/tidymodels/xgboost/')

# Load Data ----------
vb_matches <- readr::read_csv("vb_matches.csv", guess_max = 76000)


# Preprocess data -----
vb_parse <- vb_matches %>% 
  transmute(
    circuit,
    gender,
    year,
    # Winner Mutates
    w_attacks = w_p1_tot_attacks + w_p2_tot_attacks,
    w_kills = w_p1_tot_kills + w_p2_tot_kills,
    w_errors = w_p1_tot_errors + w_p2_tot_errors,
    w_aces  = w_p1_tot_aces + w_p2_tot_aces,
    w_serve_errors  = w_p1_tot_serve_errors + w_p2_tot_serve_errors,
    w_blocks  = w_p1_tot_blocks + w_p2_tot_blocks,
    w_digs  = w_p1_tot_digs + w_p2_tot_digs,
    # Loser Mutate
    l_attacks = l_p1_tot_attacks + l_p2_tot_attacks,
    l_kills = l_p1_tot_kills + l_p2_tot_kills,
    l_errors = l_p1_tot_errors + l_p2_tot_errors,
    l_aces  = l_p1_tot_aces + l_p2_tot_aces,
    l_serve_errors  = l_p1_tot_serve_errors + l_p2_tot_serve_errors,
    l_blocks  = l_p1_tot_blocks + l_p2_tot_blocks,
    l_digs  = l_p1_tot_digs + l_p2_tot_digs,
  ) %>% 
  na.omit()

winners <- vb_parse %>% 
  dplyr::select(circuit, gender, year, w_attacks:w_digs) %>% 
  rename_with(~ stringr::str_remove_all(.,"w_"), .cols = w_attacks:w_digs) %>% 
  mutate(win = 'win')
  
losers <- vb_parse %>% 
  dplyr::select(circuit, gender, year, l_attacks:l_digs) %>% 
  rename_with(~ stringr::str_remove_all(.,"l_"), .cols = l_attacks:l_digs) %>% 
  mutate(win = 'lose')

vb_df <- dplyr::bind_rows(winners,losers) %>% 
  mutate_if(is.character, factor)

vb_df %>% 
  tidyr::pivot_longer(attacks:digs, names_to = 'stat', values_to = "value") %>% 
  ggplot2::ggplot(aes(gender, value, fill = win)) + 
  geom_boxplot(alpha=0.4) +
  facet_wrap(~stat, scales='free_y',nrow = 2)+
  labs(x=NULL,y=NULL,fill=NULL)
  
# Build a Model ------------

# Split the data
set.seed(123)
vb_split <- rsample::initial_split(vb_df, strata = win)
vb_train <- rsample::training(vb_split)
vb_test <- rsample::testing(vb_split)

# Model Specification

xgb_spec <- parsnip::boost_tree(
  mtry = tune::tune(),
  #> the number (or proportion) of predictors that will be randomly sampled at each split when creating the tree models
  trees = 1000,
  min_n = tune::tune(),
  #> minimum number of data points in a node that is required for the node to be split further.
  tree_depth = tune::tune(),
  #> max depth of the tree (number of splits)
  loss_reduction = tune::tune(),
  #> reduction in the loss function required for the tree to be split further
  #> effectively is a residual sum of squares optimization
  learn_rate = tune::tune(),
  sample_size = tune::tune()
) %>% 
  set_engine('xgboost') %>% 
  set_mode('classification')

xgb_grid <- dials::grid_latin_hypercube(
  dials::tree_depth(),
  dials::min_n(),
  dials::loss_reduction(),
  sample_size = dials::sample_prop(),
  finalize(mtry(), vb_train),
  #> we use finalize to determine all possible mtry variables.
  dials::learn_rate(),
  size = 20
)

# Create workflow for Tidymodels
xgb_wf <- workflows::workflow() %>% 
  workflows::add_formula(win ~ .) %>% 
  workflows::add_model(xgb_spec)
  
# Further split the data
set.seed(123)
vb_folds <- rsample::vfold_cv(vb_train, strata = win)
#> CV or Cross Validation provides more robust estimates on models with unseen data
#> Assess model performance by splitting the sample into more samples or folds.
#> Reduction in overfitting by training on many samples
#> Optimizing hyper param tuning (this exercise) to find the best.
#> Maximize data utilization

doParallel::registerDoParallel()
set.seed(234)

# To prevent running the large estimate, the if protects
# rerun <- F
if(!file.exists('vb_xgb.rds') | rerun == T){
  xgb_res <- tune::tune_grid(
    # The Workflow is the object.
    object = xgb_wf,
    # The Cross Validations
    resamples = vb_folds, 
    grid =xgb_grid,
    control = tune::control_grid(save_pred = T)
  )
  readr::write_rds(xgb_res, 'vb_xgb.rds')
}
if(!exists('xgb_res')){xgb_res <- readr::read_rds('vb_xgb.rds')}

# Run garbage collector to free up some memory
gc()

# Results --------------------

xgb_res %>% 
  workflowsets::collect_metrics() %>% 
  filter(.metric == 'roc_auc') %>% 
  select(mean, mtry:sample_size) %>% 
  tidyr::pivot_longer(mtry:sample_size, 
               names_to = "parameter",
               values_to = "value"
               ) %>% 
  ggplot2::ggplot(ggplot2::aes(value,mean,color=parameter)) +
  ggplot2::geom_point(show.legend = F) + 
  ggplot2::facet_wrap(~parameter, scales = "free_x") +
  ggplot2::labs(y = 'Area Under the Curve',title = 'See how the Data is Dispersed')
  



# Returns the best performing model estimates by AUC
tune::show_best(xgb_res, metric = "roc_auc") %>% View()
# Pulls out the best performing estimate
best_auc <- tune::select_best(xgb_res, metric = "roc_auc"); best_auc
# Replaces the WF tuners with the parameters from the estimate with the best AUC above
final_xgb <- tune::finalize_workflow(xgb_wf, best_auc)  

final_xgb %>% 
  parsnip::fit(data = vb_train) %>%
  workflows::pull_workflow_fit() %>% 
  vip::vip(geom="point")

final_result <- tune::last_fit(final_xgb, vb_split)

final_result %>% workflowsets::collect_metrics()

final_result %>% 
  workflowsets::collect_predictions() %>% 
  yardstick::conf_mat(win, .pred_class)

final_result %>% 
  workflowsets::collect_predictions() %>% 
  yardstick::roc_curve(win, .pred_win) %>% 
  autoplot()
















