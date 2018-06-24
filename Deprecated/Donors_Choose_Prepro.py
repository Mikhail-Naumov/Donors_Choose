def Pre_Pro(kaggle=False,One_Hot=False):
    """
    #Kaggle : Generate an X Kaggle test set

    One_Hot = False
    #Default - Label Encoding for LGBM
    #One Hot - for Neural Nets
    """
    
    import gc
    import re
    
    import numpy as np
    import pandas as pd

    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    from tqdm import tqdm_notebook
    
    
    
    
    def date_prep(train):
        train['project_submitted_datetime'] = pd.to_datetime(train['project_submitted_datetime'])
        train['datetime_year'] = train['project_submitted_datetime'].dt.year
        train['datetime_month'] = train['project_submitted_datetime'].dt.month
        return(train)

    def gender_features(train):
        conditions = [(train.teacher_prefix == 'Mr.'), 
                      (train.teacher_prefix == 'Mrs.') | (train.teacher_prefix == 'Ms.')]
        choices = ['Male', 'Female']
        train['gender'] = np.select(conditions, choices, default='Unk')
        return(train)

    def scrub(text):
        text = text.strip().lower()
        text = re.sub('\W+',' ', text)    
        text = re.sub(r'_', ' ', text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r'\:', ' ', text)
        text = re.sub(r'\+', ' ', text)
        text = re.sub(r'\=', ' ', text)
        text = re.sub(r'(\")', ' ', text)
        text = re.sub(r'(\r)', ' ', text)
        text = re.sub(r'(\n)', ' ', text)
        text = re.sub(r'(\\)', ' ', text)
        text = re.sub('n t ','n\'t ', text)
        text = re.sub(' re ',' are ', text)
        text = re.sub(r'(\r\n)', ' ', text)
        text = re.sub(r'\"\"\"\"', ' ', text)
        text = re.sub(' i m ',' i\'m ', text)
        return(text)

    def swords(text):
        text = ' '.join([word for word in text.split() if word not in (all_stopwords)])
        return(text)

    def cat(text):
        text =[word for word in text.split(',')]
        text =[word.strip() for word in text]
        return(text)

    def essay_convert(train):
        # Making the First essay : student_description
        train['student_description']=train['project_essay_1']
        train.loc[train.project_essay_3.notnull(),'student_description']=train.loc[train.project_essay_3.notnull(),'project_essay_1']+train.loc[train.project_essay_3.notnull(),'project_essay_2']

        # Making the second essay : project_description
        train['project_description']=train['project_essay_2']
        train.loc[train.project_essay_3.notnull(),'project_description']=train.loc[train.project_essay_3.notnull(),'project_essay_3']+train.loc[train.project_essay_3.notnull(),'project_essay_4']

        # Removing original essays
        del train['project_essay_1']
        del train['project_essay_2']
        del train['project_essay_3']
        del train['project_essay_4']
        return(train)




    print("Importing Datasets")
    train = pd.read_csv('./Input/train.csv', low_memory=False, index_col='id')
    if kaggle: 
        test = pd.read_csv('./Input/test.csv', low_memory=False, index_col='id')

    res = pd.read_csv('./Input/resources.csv', low_memory=False, index_col='id')



    print('Precessing Resources & Merging Datasets')
    res['cost'] = res['quantity'] * res['price']
    res_agg = res.groupby('id').agg({'description': ['nunique'], 'quantity': ['sum'], 'cost': ['mean', 'sum','median','max','min']})
    res_agg.columns = ['unique_items','total_quantity', 'mean_cost', 'total_cost','median_cost','most_exp_cost','least_exp_cost']
    res_agg.reset_index(inplace=True)

    #description was dropped because the description of the project 
    #should not have an effect on its liklihood of success,
    #outside of the effect of the project essay

    train = train.merge(res_agg, left_index=True, right_on='id')
    if kaggle: 
        test =  test.merge(res_agg, left_index=True, right_on='id')

    del res, res_agg




    print('Date & Prefix Preprocessing')

    date_prep(train)
    train.teacher_prefix[train.teacher_prefix.isnull()] = 'Teacher'
    train = gender_features(train)

    if kaggle: 
        date_prep(test)
        test.teacher_prefix[test.teacher_prefix.isnull()] = 'Teacher'
        test = gender_features(test)
        del test['project_submitted_datetime'], test['project_subject_subcategories']

    del train['project_submitted_datetime'], train['project_subject_subcategories']
    del date_prep, gender_features




    print('Encoding Categorical Features')
    # Because of trees do no need onehot encoding, label encoding is used
    cols  = ['gender',
            'teacher_prefix', 
            'school_state',
            'datetime_year',
            'datetime_month',
            'project_grade_category']

    for c in tqdm_notebook(cols):
        encod = LabelEncoder()
        encod.fit(train[c].astype(str))
        train[c] = encod.transform(train[c].astype(str))
        if kaggle:        
            test[c] = encod.transform(test[c].astype(str))




    print('Scaling Numeric Features')
    num_features  = ['teacher_number_of_previously_posted_projects',
                     'total_quantity', 'mean_cost', 'total_cost','unique_items']

    SS = StandardScaler()
    train[num_features] = SS.fit_transform(train[num_features])
    if kaggle: test[num_features] = SS.transform(test[num_features])

    del num_features, StandardScaler, SS




    print('Text Processing')

    all_stopwords = [x for x in 'abcdefghijklmnopqrstuvwxyz']
    for j in ['student','students','education',]:
        all_stopwords.append(j)
    all_stopwords += stopwords.words("english")


    ##############
    #because there are some entries list as having multiple categories,
    #they are one hot encoded to allow for the acceptance of those features
    #
    #this is an awkward place in the code to put this process....

    clean_cats = train.project_subject_categories.apply(lambda x:cat(x))
    p = pd.get_dummies(clean_cats.apply(pd.Series).stack()).sum(level=0).reset_index(drop=True)
    del p['Warmth']

    train = train.reset_index(drop=True)
    train = train.merge(p,left_index=True,right_index=True)

    if kaggle:
        clean_cats = test.project_subject_categories.apply(lambda x:cat(x))
        for i in p.columns:
            test[i]=0
        p = pd.get_dummies(clean_cats.apply(pd.Series).stack()).sum(level=0).reset_index(drop=True)
        if p.columns.contains('Warmth'):
            del(p['Warmth'])
        for i in p.columns:
            test[i] = p[i]
        del test['project_subject_categories']

    del train['project_subject_categories'], p, cat

    #######################




    print('Essay Conversion & Scrub')

    text_features = ['project_title', 'project_resource_summary',
                    'project_description', 'student_description']

    essay_convert(train)
    if kaggle: essay_convert(test)

    for j in tqdm_notebook(text_features):
        n_col = 'processed_'+j
        train[n_col] = train[j].apply(lambda x: scrub(x)).apply(lambda x: swords(x))
    #    del train[i]
        if kaggle: 
            test[n_col] = test[j].apply(lambda x: scrub(x)).apply(lambda x: swords(x))
    #        del test[i]

    for i in text_features:
        del train[i]
        if kaggle: del test[i]


    gc.collect()
    del essay_convert, stopwords, all_stopwords, text_features, swords




    print('Tf-idf processing')

    cols = [
        'processed_project_title',
        'processed_project_resource_summary', 
        'processed_project_description',
        'processed_student_description']

    n_features = [100, 100, 2000, 2000]

    for c_i, c in tqdm_notebook(enumerate(cols)):
        tfidf = TfidfVectorizer(
            ngram_range=(1,2),
            max_features=n_features[c_i])

        tfidf.fit(train[c])

        tfidf_train = np.array(tfidf.transform(train[c].values).toarray(), dtype=np.float16)
        for i in range(n_features[c_i]):
            train[c + '_contains_-' + tfidf.get_feature_names()[i]] = tfidf_train[:, i]
        if kaggle:
            tfidf_test = np.array(tfidf.transform(test[c].values).toarray(), dtype=np.float16)
            for i in range(n_features[c_i]):
                test[c + '_contains_-' + tfidf.get_feature_names()[i]] = tfidf_test[:, i]

    for i in cols:
        del train[i]
        if kaggle: del test[i]

    if kaggle: del tfidf_test
    del tfidf_train, tfidf



    print('Assigning X & y')
    drop_cols = ['project_is_approved','id','teacher_id']

    X = train.drop(drop_cols, axis=1)
    y = train['project_is_approved']
    feature_names = list(X.columns)

    if kaggle: 
        return X, y, feature_names, kaggle
    else: 
        return X, y, feature_names