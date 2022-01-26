##########################################################
# Beginning of the code supplied in the course by edX.
##########################################################
#
# This code generates two datasets that must be used in the project:
#     1. The "edx" dataset, which should be used to train, develop and select the algorithm to predict moving ratings.
#     2. The "validation" dataset, which should only be used to evaluate the root mean squared error (RMSE) of the final, selected algorithm.
# 
# The datasets generated are based on the MovieLens 10M dataset.
#
# Note:
# R Version 4.1.1 was used in this project. Any code required for earlier versions of R has been retained but commented out.

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# End of the code supplied in the course by edX.
##########################################################


##########################################################
# Beginning of code to predict movie ratings in the 
# validation set.
##########################################################

# Install any packages required.
if(!require(scales)) 
  install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) 
  install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(scales)
library(lubridate)

# Examine both of the "edx" and "validation" sets to understand their structure.
str(edx)
str(validation)

# To see some example data, view the first few rows in the "edx" and "validation" sets.
head(edx)
head(validation)


##########################################################
# Do some simple checks for missing values.
##########################################################

# Check for any NA values in any column in either set.
apply(edx, 2, function(x) any(is.na(x)))
apply(validation, 2, function(x) any(is.na(x)))

# Check columns of type "character" for any empty strings.
edx %>% filter(title =="")
edx %>% filter(genres =="")
validation %>% filter(title =="")
validation %>% filter(genres =="")


##########################################################
# Explore the larger of the two sets - "edx" - 
# to further understand the nature of the data.
##########################################################

# Calculate the number of unique movies and unique users in the set.
edx %>% summarize(n_movies = n_distinct(movieId),
                  n_users = n_distinct(userId))

# Calculate how many combinations of movieId and title are in the data.
# It should be the same as the number of distinct movieIds; ie, there
# should be only one title for each movieId.
n_distinct(edx$movieId, edx$title)

# View the different ratings given and the number of movies for each rating. 
edx %>% count(rating)

# Plot the number of times each rating was given.
edx %>% ggplot(aes(rating)) +
  geom_bar(fill = "purple") +
  scale_y_continuous(labels = comma) +
  xlab("Rating") +
  ylab("Count") +
  ggtitle("Count of Ratings")


##########################################################
# Reformat the data to make it easier to work with.
##########################################################

# Add a column to the edx set that contains the timestamp 
# in a human-readable format.
edx <- edx %>% mutate(rating_datetime = as_datetime(timestamp))

# Add a column to the edx set that contains the week the
# rating was given.(This will be used in one of the models.)
edx <- edx %>% mutate(week_of_rating = round_date(rating_datetime, unit = "week")) 

# Add a column to the edx set that contains the year of 
# release of the movie.(This will be used in one of the
# models.)
edx <- edx %>% mutate(year_of_release = as.numeric(str_sub(title,-5,-2)))

# Repeat the above steps for the validation set.
validation <- validation %>% mutate(rating_datetime = as_datetime(timestamp))
validation <- validation %>% mutate(week_of_rating = round_date(rating_datetime, unit = "week")) 
validation <- validation %>% mutate(year_of_release = as.numeric(str_sub(title,-5,-2)))

# Examine the first few rows of the modified sets to view the new columns.
head(edx)
head(validation)


##########################################################
# Set up the objects required for the modelling.
##########################################################

# Create Training and Test Sets from the "edx" set.

# The following line of code assumes that R 3.6 or later
# is being used.
set.seed(1, sample.kind = "Rounding") 

# Create a training set that is 90% of the edx set,
# and a test set that is 10% of the edx set.
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train_set <- edx %>% slice(-edx_test_index)
temp <- edx %>% slice(edx_test_index)

# For some of the calculations we make later, the users, 
# movies, week_of_ratings, year of release, and genres
# that are in the test set must also be in the training set.
# Therefore, at this point we remove any rows in the test
# set with a user, movie, week_of_rating, year of release,
# or genres that are not in the training set.
edx_test_set <- temp %>%
  semi_join(edx_train_set, by = "movieId") %>%
  semi_join(edx_train_set, by = "userId") %>%
  semi_join(edx_train_set, by = "week_of_rating") %>%
  semi_join(edx_train_set, by = "year_of_release") %>%
  semi_join(edx_train_set, by = "genres")

# Add rows removed from the test set back into the
# training set.
removed <- anti_join(temp, edx_test_set)
edx_train_set <- rbind(edx_train_set, removed)

# Define a function to calculate the root mean squared
# error (RMSE) of a model.
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


##########################################################
# Naive Average Rating Model
# --------------------------
# We start with a very simple model, where every rating is
# predicted to be the same as the average rating of all 
# movies in the training set.
##########################################################

# Calculate the average rating for the training set.
mu <- mean(edx_train_set$rating)
mu

# Use this as the predicted rating for all rows in the test set,
# and calculate the RMSE (distance) between it and the 
# actual ratings in the test set.
naive_rmse <- RMSE(edx_test_set$rating, mu)
naive_rmse
rmse_results <- tibble(
  Method = "Naive Model - Average Rating of All Movies", 
  RMSE = naive_rmse)

knitr::kable(rmse_results)


##########################################################
# Movie Effect Model
# -------------------
# Augments the previous model by incorporating the average
# rating for each movie.
##########################################################

# Compute the average rating for each movie in the training
# set, and plot the distribution.
edx_train_set %>%
  group_by(movieId) %>%
  summarize(avg_i = mean(rating)) %>%
  ggplot(aes(avg_i)) +
  geom_histogram(bins = 30, color = "white", fill = "magenta") +
  ggtitle("Average Rating per Movie") + xlab("Average Rating") + 
  ylab("Count of Movies") + 
  scale_y_continuous(labels = comma)+
  scale_x_continuous(limits = c(0, 5))

# For each movie in the training set: subtract the overall
# average rating across all movies (mu) from each rating
# that movie was given, and calculate the average. This is
# denoted as "b_i", and gives us the "movie effect" for 
# each movie in the training set.
movie_avgs <- edx_train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Use the movie effects calculated above to predict the
# rating for each row in the test set.
predicted_ratings <- mu + edx_test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# Calculate the RMSE using the actual ratings in the test
# set and the predicted ratings calculated above.
movie_effects_rmse <- RMSE(edx_test_set$rating, predicted_ratings)
movie_effects_rmse
rmse_results <- bind_rows(rmse_results,
  tibble(Method="Movie Effect Model",
         RMSE = movie_effects_rmse))

knitr::kable(rmse_results)


##########################################################
# Movie + User Effects Model
# --------------------------
# Augments the previous model by incorporating the average
# rating for each user.
##########################################################

# Compute the average rating for each user in the training
# set, and plot the distribution.
edx_train_set %>%
  group_by(userId) %>%
  summarize(avg_u = mean(rating)) %>%
  ggplot(aes(avg_u)) +
  geom_histogram(bins = 30, color = "white", fill = "blue") +
  ggtitle("Average Rating per User") + xlab("Average Rating") + 
  ylab("Count of Users") + 
  scale_y_continuous(labels = comma)+
  scale_x_continuous(limits = c(0, 5))

# Apply a similar approach to that used for movie effects
# to calculate the "user effect" for each user ("b_u")
# in the training set. 
# Incorporate the user effects into the model we used
# for movie effects. This new model includes both the 
# movie effects (b_i), and user effects (b_u).
user_avgs <- edx_train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Use this model to predict ratings for all rows in the
# test set, and calculate the RMSE.
predicted_ratings <- edx_test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

user_effects_rmse <- RMSE(edx_test_set$rating, predicted_ratings)
user_effects_rmse
rmse_results <- bind_rows(rmse_results, 
  tibble(Method="Movie and User Effects Model",
         RMSE = user_effects_rmse))

knitr::kable(rmse_results)


##########################################################
# Movie + User + Time Effects Model
# ---------------------------------
# Augments the previous model by incorporating the average
# rating for each week.
##########################################################

# Compute the average rating per week based on the 
# timestamp in the training set, and plot the distribution.
edx_train_set %>%
  group_by(week_of_rating) %>%
  summarize(avg_w = mean(rating)) %>%
  ggplot(aes(avg_w)) +
  geom_histogram(bins = 30, color = "white", fill = "dark cyan") +
  ggtitle("Average Rating per Week") + xlab("Average Rating") + 
  ylab("Count of Weeks") + 
  scale_y_continuous(labels = comma) +
  scale_x_continuous(limits = c(0, 5))

# Using the week of the timestamp for each rating in the 
# training set, calculate the "time effect" for each week
# ("b_w").
# Incorporate the time effects into the model we used
# above. This new model includes the movie effects (b_i),
# user effects (b_u), and time effects (b_w).
week_avgs <- edx_train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(week_of_rating) %>%
  summarize(b_w = mean(rating - mu - b_i - b_u))

# Use this model to predict ratings for all rows in the
# test set, and calculate the RMSE.
predicted_ratings <- edx_test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(week_avgs, by='week_of_rating') %>%
  mutate(pred = mu + b_i + b_u + b_w) %>%
  pull(pred)

time_effects_rmse <- RMSE(edx_test_set$rating, predicted_ratings)
time_effects_rmse
rmse_results <- bind_rows(rmse_results,
  tibble(Method="Movie, User and Time Effects Model",
         RMSE = time_effects_rmse))

knitr::kable(rmse_results)


##########################################################
# Movie + User + Time + Year of Release Effects Model
# ---------------------------------------------------
# Augments the previous model by incorporating the average
# rating for the year of release of the movies.
##########################################################

# Compute the average rating for each year of release 
# in the training set, and plot the distribution.
edx_train_set %>%
  group_by(year_of_release) %>%
  summarize(avg_y = mean(rating)) %>%
  ggplot(aes(avg_y)) +
  geom_histogram(bins = 30, color = "white", fill = "gold") +
  ggtitle("Average Rating per Year of Release") + xlab("Average Rating") + 
  ylab("Count of Years") + 
  scale_y_continuous(labels = comma) +
  scale_x_continuous(limits = c(0, 5))

# Using the year of release for each movie in the 
# training set, calculate the "year effect" for each year
# of release ("b_y").
# Incorporate the year effects into the model we used
# above. This new model includes the movie effects (b_i),
# user effects (b_u), time effects (b_w), and year
# effects (b_y).
year_avgs <- edx_train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(week_avgs, by='week_of_rating') %>%
  group_by(year_of_release) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u - b_w))

# Use this model to predict ratings for all rows in the
# test set, and calculate the RMSE.
predicted_ratings <- edx_test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(week_avgs, by='week_of_rating') %>%
  left_join(year_avgs, by='year_of_release') %>%
  mutate(pred = mu + b_i + b_u + b_w + b_y) %>%
  pull(pred)

year_effects_rmse <- RMSE(edx_test_set$rating, predicted_ratings)
year_effects_rmse
rmse_results <- bind_rows(rmse_results,
  tibble(Method="Movie, User, Time and Year of Release Effects Model",
         RMSE = year_effects_rmse))

knitr::kable(rmse_results)


##########################################################
# Movie + User + Time + Year + Genres Effects Model
# -------------------------------------------------
# Augments the previous model by incorporating the average
# rating for each value of "genres". 
# The value of "genres" may be a single genre or a
# combination of two or more genres.
# For movies that did not have a genre recorded, the value
# of "genres" was set to "(no genres listed)".
##########################################################

# Compute the average rating for each combination of genres in the training
# set, and plot the distribution.
edx_train_set %>%
  group_by(genres) %>%
  summarize(avg_g = mean(rating)) %>%
  ggplot(aes(avg_g)) +
  geom_histogram(bins = 30, color = "white", fill = "dark green") +
  ggtitle("Average Rating for Genres") + xlab("Average Rating") + 
  ylab("Count of Genres") + 
  scale_y_continuous(labels = comma) +
  scale_x_continuous(limits = c(0, 5))

# Using the "genres" for each movie in the training set,
# calculate the "genres effect" for each value of "genres"
# ("b_g").
# Incorporate the genres effects into the model we used
# above. This new model includes the movie effects (b_i),
# user effects (b_u), time effects (b_w), year effects
# (b_y), and genres effect (b_g).
genre_avgs <- edx_train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(week_avgs, by='week_of_rating') %>%
  left_join(year_avgs, by='year_of_release') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_w - b_y))

# Use this model to predict ratings for all rows in the
# test set, and calculate the RMSE.
predicted_ratings <- edx_test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(week_avgs, by='week_of_rating') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year_of_release') %>%
  mutate(pred = mu + b_i + b_u + b_w + b_y + b_g) %>%
  pull(pred)

genre_effects_rmse <- RMSE(edx_test_set$rating, predicted_ratings)
genre_effects_rmse
rmse_results <- bind_rows(rmse_results,
  tibble(Method="Movie, User, Time, Year and Genre Effects Model",
         RMSE = genre_effects_rmse))

knitr::kable(rmse_results)


##########################################################
# Predict Ratings in the Validation Set
# ----------------------------------------
# Re-calculates the effects used in the final model based on
# the entire edx set, then uses the final model to produce 
# predicted ratings for the validation set.
##########################################################

# Calculate the average rating for the entire edx set.
edx_mu <- mean(edx$rating)

# Calculate the movie effect for the entire edx set.
edx_movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - edx_mu))

# Calculate the user effect for the entire edx set.
edx_user_avgs <- edx %>%
  left_join(edx_movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - edx_mu - b_i))

# Calculate the time effect for the entire edx set.
edx_week_avgs <- edx %>%
  left_join(edx_movie_avgs, by='movieId') %>%
  left_join(edx_user_avgs, by='userId') %>%
  group_by(week_of_rating) %>%
  summarize(b_w = mean(rating - edx_mu - b_i - b_u))

# Calculate the year of release effect for the entire edx set.
edx_year_avgs <- edx %>%
  left_join(edx_movie_avgs, by='movieId') %>%
  left_join(edx_user_avgs, by='userId') %>%
  left_join(edx_week_avgs, by='week_of_rating') %>%
  group_by(year_of_release) %>%
  summarize(b_y = mean(rating - edx_mu - b_i - b_u - b_w))

# Calculate the genre effect for the entire edx set.
edx_genre_avgs <- edx %>%
  left_join(edx_movie_avgs, by='movieId') %>%
  left_join(edx_user_avgs, by='userId') %>%
  left_join(edx_week_avgs, by='week_of_rating') %>%
  left_join(edx_year_avgs, by='year_of_release') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - edx_mu - b_i - b_u - b_w - b_y))

# Use the values calculated above to predict ratings for
# the validation set.
validation_predicted_ratings <- validation %>%
  left_join(edx_movie_avgs, by='movieId') %>%
  left_join(edx_user_avgs, by='userId') %>%
  left_join(edx_week_avgs, by='week_of_rating') %>%
  left_join(edx_year_avgs, by='year_of_release') %>%
  left_join(edx_genre_avgs, by='genres') %>%
  mutate(pred = edx_mu + b_i + b_u + b_w + b_y + b_g) %>%
  pull(pred)

# Calculate the RMSE using the actual ratings in the
# validation set and the predicted ratings calculated
# above.
validation_final_rmse <-RMSE(validation$rating, validation_predicted_ratings)
validation_final_rmse
rmse_results <- bind_rows(rmse_results,
  tibble(Method="Final Model on Validation Set",
         RMSE = validation_final_rmse))

knitr::kable(rmse_results)

