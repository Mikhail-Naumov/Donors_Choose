
# <center> LightGBM for Predicting DonorsChoose</center>

--------

## Introduction


![d_c_1](https://user-images.githubusercontent.com/36013672/38519542-26d8e118-3c0e-11e8-9125-46f7a7c13481.png)

Since the recession of 2008 US public schools have grown drastically underfunded, further complicated by the rise in costs of class resources & the growing need for computer heavy instruction. 
-img-

As a result, teachers increasingly spend personal capital to keep their classes afloat, but these teachers often need additional support. 

DonorsChoose is a United States based nonprofit organization that facilitates the direct donation to a public school in need. 

Applicants supply two essays, one about the project they want funded  and another about the students who will use those resources, and wait to see if their kickstarter like request will be funed.

</a>

## Objectives
***
#### Local Objective:
***

- Generate a predictive model that estimates the liklihood of a proposed project being fully funded.

#### Global Objective:
***
- Use this predictive model to:
    - Simulate the liklihood of success for a proposed project:
        - Finding the most favorable conditions, with the highest rate of success
    - Determine which elements & features have the highest impact on a project's success.
- Using the 180,000+ submissions to learn how to write and when to send a successful proposal.
    - Use the same methodology for other written application (NIH, Grant Submissions, ect)

## EDA

From the collected data, any project is roughly 86%, when no accounting for application variablities.

### Statewide
***

California due to its high application frequency, doubling that of the following states was ommited to see a further trends.

![screen shot 2018-06-06 at 4 27 46 pm](https://user-images.githubusercontent.com/36013672/41063417-9420676c-69a6-11e8-96a7-7108c2f61bf0.png)


By removing California we can see a bit more of a seperation between heavy applicants and non, however this is not a very strong indicator.


![screen shot 2018-06-06 at 4 29 13 pm](https://user-images.githubusercontent.com/36013672/41063482-c61402a6-69a6-11e8-88c5-33a93a66c6ab.png)


Unsurprisingly the frequency of applications peak just before the school season begins.

![screen shot 2018-06-06 at 4 30 04 pm](https://user-images.githubusercontent.com/36013672/41063531-e85ff1da-69a6-11e8-80e3-e0b32496d42d.png)

Further unsurprisingly, likely due to the increase in appliation volume, the % successful applications drops at the very start of the school season.
Additionally, while the gender of the applying teacher do seem to track, it may make for an interesting feature to further examine.

A very surprising finding is that the mean price for successful projects is consistantly higher than that of un-successful projects. These too follow the regular trend of success over time.



![screen shot 2018-04-08 at 11 44 27 pm](https://user-images.githubusercontent.com/36013672/38519782-f031edfc-3c0e-11e8-89ea-2a30751224a0.png)



## Preprocessing

![screen shot 2018-06-06 at 4 26 04 pm](https://user-images.githubusercontent.com/36013672/41063584-0a462634-69a7-11e8-94e4-df6b6d964e0e.png)


### Non_Text Preprocessing
***

- The only missing values were in that of 4 teacher's prefixs, these were easily remideied, by simply changing those values to the pre-existing ambiuous ```'Teacher'``` prefix.

| Original ```Teacher_prefix``` | Updated ```Teacher_prefixes``` |
|---|-----|
| ```"Nan"``` |  ```Teacher``` | 


- While there were 51 ```States``` labeled, this was because the District of Columbia was assigned as a state. I had originally thought these were missing values and had attempted to generate a groupby dataset on the Teacher IDs, as to see if those teachers may have had their states for other projects as to impute those values, however that was not the case. I left the ```States``` column as it was, leaving ```"DC"``` as its own unique entity.


- The `Gender` feature was engineered by extracting ```Male, Female, Unknown(Unk)``` from the teacher's prefixes, assigning ```"Teacher"``` & ```"Dr."``` as ```"Unk"``` and ```"Male"``` & ```"Female"``` accordingly. During preliminary EDA it seemed that there was a non-zero relationship between the target variable ```"Project_was_approved"``` and these genders, additionally I wanted to see if there were variable trends, which applied to teachers differently based on their gender.

| New Feature ```Gender``` | ```Teacher_prefixes``` included: |
|---|-----|
| ```"Female"``` |  ```Mrs.```   or    ```"Ms."```| 
| ```"Male"``` |  ```"Mr."```| 
| ```"Unk"``` | ```"Teacher"```   or      ```"Dr."```| 

### Text Preprocessing
***

| <center>Original Text Columns|
|---|
|<center>```"Project Title"```|
|<center>```"Resource Summary"```|
|<center>```"Essay 1"```|
|<center>```"Essay 2"```|
|<center>```"Essay 3"```|
|<center>```"Essay 4"```|

| <center>Final Text Columns|
|---|
|<center>```"Processed Project Title"```|
|<center>```"Processed Resource Summary"```|
|<center>```"Processed Student Description"```|
|<center>```"Processed Project Description"```|

#### DonorsChoose changed their submission essays requirements in May of 2016.
#### Before May 17th, 2016:

- **project_essay_1**: ```"Introduce us to your classroom"```

- **project_essay_2**: ```"Tell us more about your students"```

- **project_essay_3**: ```"Describe how your students will use the materials you're requesting"```

- **project_essay_4**: ```"Close by sharing why your project will make a difference"```

#### May 17th, 2016 and beyond:
- **project_essay_1**: ```"Describe your students: What makes your students special? Specific details about their background, your neighborhood, and your school are all helpful."```

- **project_essay_2**: ```"About your project: How will these materials make a difference in your students' learning and improve their school lives?"```

As a result of the fundimental similarities and differences from these times, I aggregated conditional essays to make new features:

| New Feature | before May 17th | after May 17th |
|---|---|---|
| ```student_description``` | Combined ```essay_1``` and ```essay_2``` | ```essay_1```| 
| ```project_description``` | Combined ```essay_2``` and ```essay_4``` | ```essay_2```| 


For each of these features, general text scrubbing was preformed, removing punctuation,
single letters and maintaining only the words of note.

| Text Prep Removed: | ```single letters 'a-z','puncuation','basic stopwords']``` |
|---|---|

Another list of stopwords was generated that removes other words like ```'student'``` and ```'education'``` as because all these essays are about students and education, they provide as much as any other stopword.

| Updated Stopwords included: | ```['student','students','education']``` |
|---|---|---|

In preliminary models I used a lemmetizer, however found that it resulted in lemmetized essays being extremely similar, only their 'gist' was maintained. And as that 'gist' was to answer the given question, they were essentally identical.

### Word Cloud

![screen shot 2018-06-06 at 4 35 13 pm](https://user-images.githubusercontent.com/36013672/41063779-9e0a45da-69a7-11e8-8d9a-36c03c46e9d4.png)


## Model Generation

![screen shot 2018-06-06 at 4 26 24 pm](https://user-images.githubusercontent.com/36013672/41063614-22594152-69a7-11e8-857d-19a1b4d4e17d.png)


### Tfidf
***
A tfidf vectorizor was preformed on the text features, while a hashing vectorizor would be far faster and less computationally expensive, I felt it was important to see the feature importance of the essay's content. In short, if essays are significantly impactful on success what qualities of the essays cause that impact?

### LightGBM Tree
***
A lightGBM model was used because of the need for:
- Interpretability
- Customiziblity
- Handling Large Datasets (>4gb after preprocessing & tfidf)
- Handles Many iterations (>10,000 iters)


Initally an XGB & Neural Network model were used, however they were both computationally more taxing (requiring up to 10x the solve time) with no significant increase in preformance.

Because of the unbalanced classes:
- K-fold validation was preformed as opposed to CV
- Auc was used a metric, as to measure the False Positives & Negatives


## Findings 

Ultimately our model and an auc-roc of .77 meaning we can comfortable say the model can better predict the success of a program than the baseline.

### Important Features

Because of the nature of Random Tree Models, feature extraction is very easy and interpretable. 

![download copy](https://user-images.githubusercontent.com/36013672/41063732-786c08b8-69a7-11e8-879f-0ddf614015c4.png)
![screen shot 2018-06-06 at 4 37 15 pm](https://user-images.githubusercontent.com/36013672/41063861-e619857a-69a7-11e8-82da-e623e4016141.png)


#Future direction:
Non aggregated sentiment analysis,
see if essays with high pos & high neg scores do better than those with both low pos and low neg
