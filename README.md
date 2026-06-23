# Masters-Thesis-LLM
Welcome to my final masters thesis project ! For this project I created large language model built with a supervised learning technique to sort text based on the most prevelant emotion present out of the four most basic emotions higlighted in the Basic Emotion Theory (anger, fear, sadness, joy). 

**Overview**

Within this project I had the goal of tagging each tweet with the strongest emotion creating, training, and using a machine learning model to tag the tweets with said emotion. In order to do so I used the BERT encoding algorithm to encode my text which I then numerically associated with one of the four above emotions. I then passed this trained data, with the tagged emotions having come from crowd sourced annotations, through two different machine learning models to determine which model would be the most effective. I used both a decision tree model and a random forests model, training both of them and tuning the hyperparameters. I then used the completed random forests model to compare the sentiment of tweets from the 2020 US presidential election and the 2024 US presidential election. Out of all the above I used the BERT model to embed the text but the rest of the data preparation was done by myself.  

**Step by step what I did** 
- I prepped the data: imported the embedding model, encoded my data, and normalized my encoded vector sizes
- I tested the data for normalcy 
- I split the data into training data and testing data
- I fed the data through two models types (random forests and decision tree) to train them with no hyperparameters present 
- I tuned the hyperparameters of both of my models using a grid search resulting in a 99% accuracy rate and minimal overfitting
- I decided on my most reliable model of the two (random forests won!) 

**My data analysis**
- Include screenshots of the outcomes, include more info about the data, maybe do the same analysis with a different test set since the test set I used showed kinda crazy outcomes
- For my data analytics I compared tweets about the 2020 US presidential election to tweets about the 2024 US presidential election. I looked at the change in emotions between the two elections and tested for statistical significance at the change in emotion. I created multiple visualizations to compare the two election sentiments and highlight the overall changes. My main finding is that fear was the strongest emotion during the time of both elections though the amount of fear decreased between the 2020 and 2024 election by 34.8% and instead the levels of anger and joy rose, both by around 8%.

**include a section that highlights why this project is usefull** 
It's useful becaussseee
- other people can use the tagger to show sentiment
- can be used for google reviews, yelp reviews, to focus on improvement areas
- can be used for marketing 
