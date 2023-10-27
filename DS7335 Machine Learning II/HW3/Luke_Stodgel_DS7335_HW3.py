# -*- coding: utf-8 -*-
"""
Feb 18 2023

@author: Luke
"""
import numpy as np

# Despite the simplicity of the process you will need to make decisions 
#regarding how to process the data.
  
# This process was thoroughly investigated in the operation research community.
# This approach can prove helpful on any number of decision making problems 
# that are currently not leveraging machine learning.  

# You asked your 10 work friends to answer a survey. They gave you back 
#the following dictionary object.  

#1
#the higher the number, the higher the weight of the category in their eyes
people  = {'Jane': {'distance': 5,
                    'novelty': 2,
                    'cost': 1,
                    'average rating': 4,
                    'cuisine': 1,
                    'vegetarian': 5},
           'John': {'distance': 1,
                    'novelty': 3,
                    'cost': 1,
                    'average rating': 5,
                    'cuisine': 5,
                    'vegetarian': 1},
            'Joe': {'distance': 5,
                    'novelty': 1,
                    'cost': 4,
                    'average rating': 3,
                    'cuisine': 1,
                    'vegetarian': 1},
           'Jack': {'distance': 1,
                    'novelty': 2,
                    'cost': 3,
                    'average rating': 4,
                    'cuisine': 5,
                    'vegetarian': 4},
            'Jon': {'distance': 2,
                    'novelty': 3,
                    'cost': 4,
                    'average rating': 5,
                    'cuisine': 4,
                    'vegetarian': 3},
           'Jose': {'distance': 3,
                    'novelty': 4,
                    'cost': 5,
                    'average rating': 4,
                    'cuisine': 3,
                    'vegetarian': 2},
         'Joseph': {'distance': 4,
                    'novelty': 5,
                    'cost': 4,
                    'average rating': 3,
                    'cuisine': 2,
                    'vegetarian': 1},
        'Jessica': {'distance': 5,
                    'novelty': 4,
                    'cost': 3,
                    'average rating': 2,
                    'cuisine': 1,
                    'vegetarian': 2},
          'Julia': {'distance': 4,
                    'novelty': 3,
                    'cost': 2,
                    'average rating': 1,
                    'cuisine': 2,
                    'vegetarian': 3},
          'Julie': {'distance': 3,
                    'novelty': 2,
                    'cost': 1,
                    'average rating': 2,
                    'cuisine': 3,
                    'vegetarian': 4}}

print("#1 Transform the user data into a matrix (M_people). "
      "Keep track of column and row IDs.")
# Convert the nested dictionary to a list of dictionaries
people_list = []
for key in people:
    person_data = people[key]
    person_dict = {}
    for k, v in person_data.items():
        person_dict[k] = v
    people_list.append(person_dict)

# Convert the list of dictionaries to a NumPy array
my_list = []
for d in people_list:
    values_list = list(d.values())
    my_list.append(values_list)
M_people = np.array(my_list)

print("M_people:\n", M_people, "\n")

# Next you collected data from an internet website. You got the following information.
restaurants  = {'flacos': {'distance': 3,
                           'novelty': 2,
                           'cost': 3,
                           'average rating': 4,
                           'cuisine': 2,
                           'vegetarian': 4},
                 'pizza': {'distance': 2,
                           'novelty': 2,
                           'cost': 3,
                           'average rating': 4,
                           'cuisine': 2,
                           'vegetarian': 5},
            'steakhouse': {'distance': 1,
                           'novelty': 3,
                           'cost': 5,
                           'average rating': 4,
                           'cuisine': 2,
                           'vegetarian': 1},
                'buffet': {'distance': 4,
                           'novelty': 3,
                           'cost': 1,
                           'average rating': 3,
                           'cuisine': 5,
                           'vegetarian': 5}}

#2
print("#2 Transform the restaurant data into a matrix (M_resturants) " 
      "using the same column index.")
restaurant_list = []
for key in restaurants:
    restaurant_data = restaurants[key]
    restaurant_dict = {}
    for k, v in restaurant_data.items():
        restaurant_dict[k] = v
    restaurant_list.append(restaurant_dict)

# Convert the list of dictionaries to a NumPy array
my_list = []
for d in restaurant_list:
    values_list = list(d.values())
    my_list.append(values_list)
M_restaurants = np.array(my_list)

print("M_restaurants:\n", M_restaurants, "\n") # each row represents a restaurant

#3
print("#3 The most important idea in this project is the idea of a linear",
      "combination. Informally describe what a linear combination is and",
      "how it will relate to our restaurant matrix.\n"
      "A linear combination is a mathematical operation in which you multiply",
      "each element in a vector by a scalar and add up all of the resulting "
      "products. Each restaurant will have a vector of scores based on",
      "descriptive categories like distance, novelty, cost, average rating, "
      "cuisine, vegetarian and for each person, we will use their scores as",
      "the weights or scalars to find out which restaurant they like the most.",
      "Whichever restaurant has the highest linear combination for that user",
      "will be that user's top rated restaurant. We can also use this "
      "information to find the most popular restaurants amongst the group.\n")

#4
print("#4 Choose a person and compute(using a linear combination) the top",
      "restaurant for them. What does each entry in the resulting vector ""represent?")
user_1 = M_people[0] # we will use Jane for our example
print("Jane's preferences (weights):", user_1) # each row represents one person
print("Jane's restaurant ratings:", np.dot(M_restaurants,
                                           user_1)) # output: [60 60 39 69]
print("Each entry in the resulting vector represents the score the person",
      "gave each restaurant.")

#5
print("\n#5 Next, compute a new matrix (M_usr_x_rest i.e. a user by "
      "restaurant) from all people. What does the a_ij matrix represent?")

#M_people and M_restaurants = M_usr_x_rest
print("M_usr_x_rest:")
M_usr_x_rest = np.dot(M_people, M_restaurants.T)
print(M_usr_x_rest)
print("Each row represents a user and each column represents a restaurant.",
      "Each user rated each of the four restaurants.\n")

#6
print("#6 Sum all columns in M_usr_x_rest to get the optimal restaurant for "
      "all users. What do the entries represent?")
print("sum(M_usr_x_rest):")
print(sum(M_usr_x_rest))
print("These entries represent the total user scores for each restaurant.\n")

#7
print("#7 Now convert each row in the M_usr_x_rest into a ranking for "
      "each user and call it M_usr_x_rest_rank. Do the same as above to "
      "generate the optimal restaurant choice.")
from scipy.stats import rankdata

def get_real_rank(data):
    return rankdata(len(data)-rankdata(data))

#ranks per user
my_list = []
for item in M_usr_x_rest:
    my_list.append(get_real_rank(item))
M_usr_x_rest_rank = np.array(my_list)
print("Ranks Per User, M_usr_x_rest_rank:\n", M_usr_x_rest_rank)

#Overall restaurant ranks
M_usr_x_rest_rank_overall = get_real_rank(sum(M_usr_x_rest))
print("Overall restaurant ranks,  get_real_rank(sum(M_usr_x_rest)):\n", 
      M_usr_x_rest_rank_overall, "\n")

#8
print("#8 Why is there a difference between the two?  What problem arrives?",  
      "What does it represent in the real world?\n"

      "The difference between the two resulting lists is that one contains",
      "the overall rank of each restaurant (taking into account the scores of",
      "each user) and the other contains ranks for each restaurant, per",
      "each user.\n"

      "A problem I ran into was that when I calculated the rank for each",
      "user, the resulting array was one dimensional. This would make it hard",
      "to distinguish between the users and their ranks. To fix this, I "
      "iterated through each item in M_usr_x_rest and stored lists of each",
      "user's ranks in a larger list.\n"
      
      "Another problem that arrises when you go from having scores for each "
      "user to overall restaurant scores is that you lose information "
      "specific to each user.\n")

#9
print("#9 How should you preprocess your data to remove this problem?\n" 
      "This was answered in question #8, paragraph 2.\n")

#10
print("#10 Find user profiles that are problematic, explain why?\n"
      "Some users have restaurant rankings that are tied. This could be "
      "problematic when trying to pick one single best restaurant to go to. "
      "If every member on the team rated every restaurant as tied for 1st "
      "place, it would be impossible to decide which restaurant is the "
      "most popular amongst the group.\n"
 
      "\nAlso, some user profiles contained preferences for specific cuisines"
      " such "
      "as indian_food, mexican_food, italian_food, and american_food which "
      "could cause problems because if the restaurant is missing one"
      " of these attributes, the user and restaurant matrices will be "
      "incompatibly shaped for matrix multiplication.\n\n"

      "To fix this problem, I combined all cuisine preferences"
      " into one category called cuisine.\n")

#11
print("#11 Think of two metrics to compute the dissatisfaction with the "
      "group.\nOne metric could be 'average "
      "lowest restaurant score/rank'.\nAnother "
      "metric could be 'count' of restaurants that consistently appear in the "
      "bottom half of restaurant rankings.\n")

#12
print("#12 Should you split into two groups today?\n"
      "No. It looks like the buffet is popular amongst all but one person"
      " in the group.\n")

#13
print("#13 Ok. Now you just found out the boss is paying for the meal. " 
      "How should you adjust? Now what is the best restaurant?\n"
      "I would remove 'cost' from both the user weights and restaurant"
      " matrices. Then, I would recompute the restaurant rankings.\n")

people  = {'Jane': {'distance': 5,
                    'novelty': 2,
                    'average rating': 4,
                    'cuisine': 1,
                    'vegetarian': 5},
           'John': {'distance': 1,
                    'novelty': 3,
                    'average rating': 5,
                    'cuisine': 5,
                    'vegetarian': 1},
            'Joe': {'distance': 5,
                    'novelty': 1,
                    'average rating': 3,
                    'cuisine': 1,
                    'vegetarian': 1},
           'Jack': {'distance': 1,
                    'novelty': 2,
                    'average rating': 4,
                    'cuisine': 5,
                    'vegetarian': 4},
            'Jon': {'distance': 2,
                    'novelty': 3,
                    'average rating': 5,
                    'cuisine': 4,
                    'vegetarian': 3},
           'Jose': {'distance': 3,
                    'novelty': 4,
                    'average rating': 4,
                    'cuisine': 3,
                    'vegetarian': 2},
         'Joseph': {'distance': 4,
                    'novelty': 5,
                    'average rating': 3,
                    'cuisine': 2,
                    'vegetarian': 1},
        'Jessica': {'distance': 5,
                    'novelty': 4,
                    'average rating': 2,
                    'cuisine': 1,
                    'vegetarian': 2},
          'Julia': {'distance': 4,
                    'novelty': 3,
                    'average rating': 1,
                    'cuisine': 2,
                    'vegetarian': 3},
          'Julie': {'distance': 3,
                    'novelty': 2,
                    'average rating': 2,
                    'cuisine': 3,
                    'vegetarian': 4}}

# Convert the nested dictionary to a list of dictionaries
people_list = []
for key in people:
    person_data = people[key]
    person_dict = {}
    for k, v in person_data.items():
        person_dict[k] = v
    people_list.append(person_dict)

# Convert the list of dictionaries to a NumPy array
my_list = []
for d in people_list:
    values_list = list(d.values())
    my_list.append(values_list)
M_people2 = np.array(my_list)

restaurants  = {'flacos': {'distance': 3,
                           'novelty': 2,
                           'average rating': 4,
                           'cuisine': 2,
                           'vegetarian': 4},
                 'pizza': {'distance': 2,
                           'novelty': 2,
                           'average rating': 4,
                           'cuisine': 2,
                           'vegetarian': 5},
            'steakhouse': {'distance': 4,
                           'novelty': 4,
                           'average rating': 5,
                           'cuisine': 2,
                           'vegetarian': 1},
                'buffet': {'distance': 4,
                           'novelty': 3,
                           'average rating': 2,
                           'cuisine': 5,
                           'vegetarian': 5}}

restaurant_list = []
for key in restaurants:
    restaurant_data = restaurants[key]
    restaurant_dict = {}
    for k, v in restaurant_data.items():
        restaurant_dict[k] = v
    restaurant_list.append(restaurant_dict)

# Convert the list of dictionaries to a NumPy array
my_list = []
for d in restaurant_list:
    values_list = list(d.values())
    my_list.append(values_list)
M_restaurants2 = np.array(my_list)
### Setup complete, now we have M_restaurants and M_people

#M_people and M_restaurants = M_usr_x_rest
print("M_usr_x_rest2 (M_usr_x_rest minus the cost rating): ")
M_usr_x_rest2 = np.dot(M_people2, M_restaurants2.T)
print(M_usr_x_rest2)

# Sum all columns in M_usr_x_rest to get the optimal restaurant for all users.
# What do the entries represent?
print("Sum of all columns in M_usr_x_rest2:")
print(sum(M_usr_x_rest2))

#ranks per user
my_list = []
for item in M_usr_x_rest:
    my_list.append(get_real_rank(item))
M_user_rankings = np.array(my_list)
print("Ranks Per User, M_user_rankings (minus cost rating):\n", 
      M_user_rankings)

#restaurant ranks
M_usr_x_rest_rank = get_real_rank(sum(M_usr_x_rest))
print("Overall restaurant ranks (get_real_rank(sum(M_usr_x_rest)):", 
      M_usr_x_rest_rank)
print("Even after taking cost out of the equation, the fourth restaurant",
      "is still the most popular.\n")

#14 
print("#14 Tomorrow you visit another team. You have the same restaurants and " 
      "they told you their optimal ordering for restaurants."
      " Can you find their weight matrix?\n")

print("For this problem I used linear regression to predict "
      "user weights based on the information we were given (M_restaurants "
      "and M_user_rankings).\n"
      "After predicting the user weights, to check our work, I "
      "found the dot product of user_weights and M_restaurants"
      " and then turned that matrix into user_rankings. If the user "
      "rankings were similar "
      "to the user_rankings the other team gave us, then our "
      "weight predictions must have been at least proportionally accurate.\n")

print("Example M_user_rankings from another team:\n", M_user_rankings)
print("\nuser_weights = get_user_weights(M_user_rankings.T, M_restaurants)\n")

from sklearn.linear_model import LinearRegression

def get_user_weights(M_user_rankings, M_restaurants2):
    # perform linear regression to solve for user weights
    model = LinearRegression()
    model.fit(M_restaurants2, M_user_rankings)
    user_weights = model.coef_
    
    # return negative user_weights so that we can ensure that rank 1 still 
    # means the most popular restaurant
    return -user_weights

user_weights = get_user_weights(M_user_rankings.T, M_restaurants)

print("Predicted user_weights matrix:\n", user_weights)

M_usr_x_rest14 = np.dot(user_weights, M_restaurants.T)

print("\nM_usr_x_rest14 using predicted user weights, "
      "\nnp.dot(user_weights, M_restaurants.T):\n", M_usr_x_rest14)

#ranks per user
my_list = []
for item in M_usr_x_rest14:
    my_list.append(get_real_rank(item.round(3)))
M_user_rankings = np.array(my_list)

print("\nRanks Per User (to check if the rankings still match the original " 
      "rankings), M_user_rankings:\n",
      M_user_rankings)

print("\nUsing the predicted user weights, we were still able to retain the"
      " restaurant ranking order for each user.")