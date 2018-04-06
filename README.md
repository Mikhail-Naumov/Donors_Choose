
# <center> LightGBM for Predicting DonorsChoose</center>

--------

## Introduction


-img D_C_1

DonorsChoose is a United States based nonprofit organization that facilitates the direct donation to a public school in need. Founded in 2000, by Charles Best has quickly grown to one of the most largest public school nonprofits, breaking 1 million funded projects in January of 2018. 

- The process for Donations is rather simple, and individuals simply go on their website:
    - https://www.donorschoose.org/donors/search.html
- Find a project:
    - https://www.donorschoose.org/project/got-skillsorganizational-skills/3157618/
- And give a value donation.
<font color=blue|red|green|pink|yellow>Text</font>


</a>

## Imports

### The required imports included:
***
#### Pythonic Implimentation:
***
- pandas : for easy of organization
- numpy : for accelerated implimentation & dependancies
- tqdm : for progress tracking
- os : dependancies
- re : regex for ease of text cleaning
- gc : Garbage Collection, due to the size of the dataset and memory errors
<br>
#### Statistical Implimentations:
***
- matplotlib & seaborn : for visualizations and EDA
    - _Presented visualizations were generated in Tabealu_
    - _Inital visualizations were preformed with seaborn_
- sklearn: 
    - StandardScalar
        - Scaling Numerical Values
    - roc & auc scores
        - Evaluation Metric
    - K_fold validation
        - Validation Method
    - LabelEncoder
        - Categorical Encoder
<br>
#### NLP Implimentations:
***
- gensim : For topic modeling
- nltk :
    - Stopwords : English
    - Lemmatizer : Preformed for EDA -see 'Text Preprocessing'


## Preprocessing

### Importing & Structuring
***

 The intital data was extracted from Kaggle arrived seperated into 3 csv files:
 
 | File | Contents |
 |---|---|
 | train.csv | Project Success, Teacher Data, State Data, Project ID |
 | test.csv | Teacher Data, State Data, Project ID |
 | resources.csv | Project Quanity, Project Price, & Project ID |
 
While this inital project was _NOT_ a Kaggle submission, later I will preform a prediction using the ```test.csv``` dataset, however for this examination it was ommitted. 
<br>

The values in ```resources``` were aggrogated, and organized by the mean & sum cost, as well as the sum quanity. The thinking here was to determine how 'big' of a project each requested project was. These values were merged to the ```train``` on the ```Project ID``` that those projects were assigned to, as to generated a primary dataset to work off of.
<br>
- The project descriptions were dropped as the project's description should not have an effect on the successfulness of that project, it was uniform across all of the same projects and not a variable that changes from each individual applying to the same project.
<br>
``` eg. If we are trying to see if person A or B is more likely win a raffle, the description of that raffle which is uniform the between the two, should not have an effect on either's success ```

### Non_Text Preprocessing
***

- The only missing values were in that of 4 teacher's prefixs, these were easily remideied, by simply changing those values to the pre-existing ambiuous ```'Teacher'``` prefix.

| Original ```Teacher_prefix``` | Updated ```Teacher_prefixes``` |
|---|-----|
| ```"Nan"``` |  ```Teacher``` | 

<br>
<br>
- While there were 51 ```States``` labeled, this was because the District of Columbia was assigned as a state. I had originally thought these were missing values and had attempted to generate a groupby dataset on the Teacher IDs, as to see if those teachers may have had their states for other projects as to impute those values, however that was not the case. I left the ```States``` column as it was, leaving ```"DC"``` as its own unique entity.
<br>
<br>
- The `Gender` feature was engineered by extracting ```Male, Female, Unknown(Unk)``` from the teacher's prefixes, assigning ```"Teacher"``` & ```"Dr."``` as ```"Unk"``` and ```"Male"``` & ```"Female"``` accordingly. During preliminary EDA it seemed that there was a non-zero relationship between the target variable ```"Project_was_approved"``` and these genders, additionally I wanted to see if there were variable trends, which applied to teachers differently based on their gender.

| New Feature ```Gender``` | ```Teacher_prefixes``` included: |
|---|-----|
| ```"Female"``` |  ```Mrs.```   or    ```"Ms."```| 
| ```"Male"``` |  ```"Mr."```| 
| ```"Unk"``` | ```"Teacher"```   or      ```"Dr."```| 


-other graph-
<br>
<br>
- All the categorical values were encoded, however they were not encoded with one_hot_encoding, as one_hot_encoding was too computationally taxing. Because the model uses binary trees & boolean indexing, and so the encoded variables were valid and assigned as string. 

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
<br>

#### DonorsChoose changed their submission essays requirements in May of 2016.
#### Before May 17th, 2016:

- **project_essay_1**: ```"Introduce us to your classroom"```
<br>
<br>
- **project_essay_2**: ```"Tell us more about your students"```
<br>
<br>
- **project_essay_3**: ```"Describe how your students will use the materials you're requesting"```
<br>
<br>
- **project_essay_4**: ```"Close by sharing why your project will make a difference"```

#### May 17th, 2016 and beyond:
- **project_essay_1**: ```"Describe your students: What makes your students special? Specific details about their background, your neighborhood, and your school are all helpful."```
<br>
<br>
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
<br>
Another list of stopwords was generated that removes other words like ```'student'``` and ```'education'``` as because all these essays are about students and education, they provide as much as any other stopword.

| Updated Stopwords included: | ```['student','students','education']``` |
|---|---|---|
<br>
In preliminary models I used a lemmetizer, however found that it resulted in lemmetized essays being extremely similar, only their 'gist' was maintained. And as that 'gist' was to answer the given question, they were essentally identical.



## Model Generation

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
<br>
<br>
Initally an XGB & Neural Network model were used, however they were both computationally more taxing (requiring up to 10x the solve time) with no significant increase in preformance.

<br>
Because of the unbalanced classes:
- K-fold validation was preformed as opposed to CV
- Auc was used a metric, as to measure the False Positives & Negatives






#Future direction:
Non aggregated sentiment analysis,
see if essays with high pos & high neg scores do better than those with both low pos and low neg
