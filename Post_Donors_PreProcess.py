def Post_Donor_PrePro(Tf_Features=100,N_Gram=1,Sample=.1,One_Hot=True,Standard_Scale=True):
    """
    Tf_Features : Max TFIDF Features (100)
    Sample : Only Use x% of submissions (.1)
    One_Hot : One Hot Encode (True)
                Label Encode (False)
                
    Standard_Scale : Standard Scale (True)
                     MinMaxScalar  (False)
    N_Gram : Max_Length of Searched Term Combinations (1)
                     
    """


    #### Func

    def cat_cleaner(df,col):
        """
        Takes in the df and the column containing messy, sub cats.

        pd.get_dummies:
           col                 A | A,B | B | B,A
        0|  A              0|  1    0    0    0
        1| A,B  -onehot->  1|  0    1    0    0
        2|  B              2|  0    0    1    0
        3| B,A             3|  0    0    0    1

        this:

           col                       A | B
        0|  A                    0|  1   0
        1| A,B  -cat_cleaner ->  1|  1   1
        2|  B                    2|  0   1
        3| B,A                   3|  1   1



        """
        def cat(text):
            text =[word for word in text.split(',')]
            text =[word.strip() for word in text]
            return(text)


        df[col] = df[col].astype('str')
        clean_cats = df[col].apply(lambda x:cat(x))
        p = pd.get_dummies(clean_cats.apply(pd.Series).stack()).sum(level=0).reset_index(drop=True)
        return p

    def text_cleaner(text,all_stop):
        """
        clean_str = text_cleaner(dirty_string)
        """
        regex = re.compile('[%s]' % re.escape(string.punctuation))

        text = text.lower()
        text = text.replace('<!--DONOTREMOVEESSAYDIVIDER-->',' ')
        text = text.replace('\n' , ' ')
        text = regex.sub('',text)
        text = ' '.join([word for word in text.split() if word not in all_stop])
        return(text)

    def plot_empties(bad_form):
        plt.figure(figsize=(15,15))

        i = 1
        for j in bad_form:
            plt.subplot(230+i)
            sns.heatmap(pd.DataFrame(bad_form[j].isnull().sum()/bad_form[j].shape[0]*100),
                        annot=True,cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")
            plt.title(j)
            i+=1

        plt.subplots_adjust(wspace = 1.6)
        return

    def compress(df,encode=[],num=[],byte_str = False):
        pre_ = df[num+encode].memory_usage(deep=True)

        for i in num:
            if df[i].astype(np.float16).memory_usage(deep=True)<df[i].memory_usage(deep=True):
                df[i] = df[i].astype(np.float16)
            else:
                pass

        if byte_str:
            for j in encode:
                if df[j].astype(np.string_).memory_usage(deep=True)<df[j].memory_usage(deep=True):
                    df[j] = df[j].astype(np.string_)
                else:
                    pass
        else:
            for j in encode:
                if df[j].astype(str).memory_usage(deep=True)<df[j].memory_usage(deep=True):
                    df[j] = df[j].astype(str)
                else:
                    pass



        post_ = df[num+encode].memory_usage(deep=True)
        print("Data Usage - change:{}".format((post_-pre_)))
        print("Total Change: {}".format((post_-pre_).sum()))
        return

    ### Imports

    import gc
    import re
    import string
    import calendar
    import numpy as np
    import pandas as pd
    from nltk.corpus import stopwords

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.preprocessing import LabelEncoder

    from sklearn.feature_extraction.text import TfidfVectorizer

    bad_form = {#'Donations' : pd.read_csv('./Input/New/Donations.csv'),
                #'Donors'    : pd.read_csv('./Input/New/Donors.csv'),

                'Projects'  : pd.read_csv('./Input/New/Projects.csv'),
                'Resources' : pd.read_csv('./Input/New/Resources.csv'),
                'Schools'   : pd.read_csv('./Input/New/Schools.csv'),
                'Teachers'  : pd.read_csv('./Input/New/Teachers.csv')}


        #Treating Nulls in 'Teachers'
    bad_form['Teachers']['Teacher Prefix'] = bad_form['Teachers']['Teacher Prefix'].apply(
        lambda x: 'Teacher' if x in ['Mx.', np.nan] else x)

    #for i in bad_form:
    #    bad_form[i] = bad_form[i].dropna()
    
    ### Data Aggregation

    #How much of the data do you want to use?
    
    df = bad_form['Projects'][:int(bad_form['Projects'].shape[0] * Sample)].copy(deep=True)
    #if Sample: df = bad_form['Projects'][:1000].copy(deep=True)
    #else: df = bad_form['Projects'].copy(deep=True)

    #Projects
    print('DataFrame Init')

    #Structure Target
    df = df[df['Project Current Status'] != 'Live']
    df['Project Current Status'] = df['Project Current Status'].apply(lambda x: 1 if x == 'Fully Funded' else 0)

    #Adding dt features
    df['Project Posted Date'] = pd.to_datetime(df['Project Posted Date'])
    df['Project Posted Year'] = df['Project Posted Date'].dt.year.astype(str)
    df['Project Posted Month'] = df['Project Posted Date'].dt.month
    df['Project Posted Month'] = df['Project Posted Month'].apply(lambda x: calendar.month_abbr[x])

    #cols
    encode_cols = ['Project Type','Project Posted Year','Project Posted Month',
                   'Project Grade Level Category','Project Resource Category',]
    num_cols    = ['Project Cost']

    #del
    for i in ['Teacher Project Posted Sequence', 'Project Fully Funded Date', 
              'Project Expiration Date', 'Project Subject Subcategory Tree', 
              'Project Posted Date','Project Title','Project Short Description']:
        del df[i]
    del calendar
    gc.collect()

    
    #Null values from all data sources

    #plot_empties(bad_form)
    del plot_empties

    #Teachers
    print('Merging Teacher Information')

    #Treating Nulls in 'Teachers'
    bad_form['Teachers']['Teacher Prefix'] = bad_form['Teachers']['Teacher Prefix'].apply(
        lambda x: 'Teacher' if x in ['Mx.', np.nan] else x)

    #merging teacher to df
    df = df.merge(bad_form['Teachers'],on='Teacher ID').copy(deep=True)

    #cols
    encode_cols += ['Teacher Prefix']
    num_cols    += []

    #del
    for i in ['Teacher ID','Teacher First Project Posted Date']:
        del df[i]
    gc.collect()

    #School
    print('Merging School Information')

    #very few, dropping nulls
    bad_form['Schools'] = bad_form['Schools'].dropna()

    #merging school to df
    df = df.merge(bad_form['Schools'],on='School ID')
    df['School Zip'] = df['School Zip'].astype(str)

    #cols
    encode_cols += ['School Metro Type','School State','School District','School Zip']
    num_cols    += ['School Percentage Free Lunch']

    #del
    for i in ['School Name','School ID','School City','School County']:
        del df[i]
    gc.collect()

    # Resource Management
    print('Merging Resource Managment')

    #more funding features
    bad_form['Resources']['Resource Total Cost'] = bad_form['Resources']['Resource Quantity'] * bad_form['Resources']['Resource Unit Price']
    funding_agg = bad_form['Resources'].groupby('Project ID').agg(
        {'Resource Item Name': ['nunique'], 
         'Resource Quantity': ['sum'], 
         'Resource Total Cost': ['mean', 'sum','median','max','min']})

    funding_agg.columns = ['Project num Unique Resources',
                       'Total Resource Quantity', 
                       'Mean Resource Cost', 
                       'Total Project Cost',
                       'Median Resource Cost',
                       'Most exp Resource Cost',
                       'Least exp Resource Cost']

    #cols
    encode_cols += []
    num_cols    += list(funding_agg.columns.values)

    #merging funding to df
    funding_agg.reset_index(inplace=True)
    df = df.merge(funding_agg,on='Project ID')

    del funding_agg
    gc.collect()

    del df['Project ID']

    #s = set(encode_cols+num_cols)
    #df[[x for x in df.columns if x not in s]].head(5)



    #compressing data
    #print('Compressing Data')
    #compress(df,encode_cols,[])
    del compress
    gc.collect()





    ### Encoding / Scaling

    #Preprocessing
    print('Preprocessing')
    df = df.dropna()

    #Encoding
    le_dict = {}
    if One_Hot:
        df = df.merge(pd.get_dummies(df[encode_cols])
                 ,left_index=True,right_index=True)
        for i in encode_cols:
            del df[i]
    else:
        for c in encode_cols:
            encod = LabelEncoder()
            encod.fit(df[c].astype(str))
            df[c] = encod.transform(df[c].astype(str))
            le_dict[c] = dict(zip(encod.classes_, encod.transform(encod.classes_)))
        del encod

    p = cat_cleaner(df,'Project Subject Category Tree')
    df = df.merge(p,left_index=True,right_index=True)
    del df['Project Subject Category Tree'], p

    #Scaling
    print('Scaling')
    if Standard_Scale:
        Scalar = StandardScaler()
    else:
        Scalar = MinMaxScaler()

    df[num_cols] = Scalar.fit_transform(df[num_cols])

    del LabelEncoder, Standard_Scale, Scalar
    gc.collect()




    ### Text Editing

    print('Text Processing')

    text_cols = ['Project Essay','Project Need Statement']

    #adding more words to 'stopwords'
    extra_words = ['student','students','education']
    single_l = [x for x in 'abcdefghijklmnopqrstuvwxyz']
    for j in single_l:
        extra_words.append(j)
    extra_words += stopwords.words("english")

    for i in text_cols:
        print("{} Processing".format(str(i)))
        df[i] = df[i].apply(lambda x: text_cleaner(x,extra_words))

    del text_cleaner, extra_words, single_l, stopwords

    #Tfidf
    tfidf = TfidfVectorizer(max_features=Tf_Features, ngram_range = (1,N_Gram))

    for i in text_cols:
        print("{} TFIDF".format(str(i)))
        tfidf.fit(df[i])
        tf_cols = [str(i)+' contains: "'+str(x)+'"' for x in list(tfidf.vocabulary_.keys())]
        df = df.merge(pd.DataFrame(tfidf.transform(df[i]).todense(),columns=tf_cols), left_index=True, right_index=True)

    for i in text_cols:
        del df[i]
    del tfidf, tf_cols


    X = df.drop(['Project Current Status'],axis=1)
    y = df['Project Current Status']
    df_cols = df.columns
       
    if One_Hot:
        return(X,y,df_cols)
    else:
        return(X,y,df_cols,le_dict)
    
    
    
    
    
    
def Diet_Prepro(One_Hot = False, Standard_Scale = True, Tf_Features = 100, N_Gram = 1):

    import gc
    import re
    import string
    import numpy as np
    import pandas as pd
    from nltk.corpus import stopwords

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.preprocessing import LabelEncoder

    from sklearn.feature_extraction.text import TfidfVectorizer

    def text_cleaner(text,all_stop):
        """
        clean_str = text_cleaner(dirty_string)
        """
        regex = re.compile('[%s]' % re.escape(string.punctuation))

        text = text.lower()
        text = text.replace('<!--DONOTREMOVEESSAYDIVIDER-->',' ')
        text = text.replace('\n' , ' ')
        text = regex.sub('',text)
        text = ' '.join([word for word in text.split() if word not in all_stop])
        return(text)

    df_types = {'Project Type':str,'Project Essay':str, 'Project Need Statement':str, 'Project Subject Category Tree':str, 
        'Project Grade Level Category':str, 'Project Resource Category':str, 'Project Cost':float, 'Project Current Status':int,
        'Project Posted Year':int, 'Project Posted Month':str, 'Teacher Prefix':str, 'School Metro Type':str, 
        'School Percentage Free Lunch':float, 'School State':str, 'School Zip':int,
        'School District':str, 'Project num Unique Resources':int, 'Total Resource Quantity':float, 'Mean Resource Cost':float,
        'Total Project Cost':float, 'Median Resource Cost':float, 'Most exp Resource Cost':float, 'Least exp Resource Cost':float, 
        'Applied Learning':int, 'Care & Hunger':int, 'Health & Sports':int, 'History & Civics':int, 'Literacy & Language':int,
        'Math & Science':int, 'Music & The Arts':int, 'Special Needs':int, 'Warmth':int}

    df = pd.read_csv("./Input/Processed/df_no_encode_yes_essay.csv",index_col=0,dtype=df_types)

    encode_cols = ['Project Type',
                 'Project Posted Year',
                 'Project Posted Month',
                 'Project Grade Level Category',
                 'Project Resource Category',
                 'Teacher Prefix',
                 'School Metro Type',
                 'School State',
                 'School District',
                 'School Zip']

    num_cols    = ['Project Cost',
                 'School Percentage Free Lunch',
                 'Project num Unique Resources',
                 'Total Resource Quantity',
                 'Mean Resource Cost',
                 'Total Project Cost',
                 'Median Resource Cost',
                 'Most exp Resource Cost',
                 'Least exp Resource Cost']

    #Preprocessing
    print('Encoding')

    #Encoding
    le_dict = {}

    #cat_col
    del df['Project Subject Category Tree'], df_types

    if One_Hot:
        df = df.merge(pd.get_dummies(df[encode_cols]),left_index=True,right_index=True)
        for i in encode_cols:
            del df[i]
    else:
        for c in encode_cols:
            encod = LabelEncoder()
            encod.fit(df[c].astype(str))
            df[c] = encod.transform(df[c].astype(str))
            le_dict[c] = dict(zip(encod.classes_, encod.transform(encod.classes_)))
        del encod
    
    del encode_cols, One_Hot, LabelEncoder,
    
    
    #Scaling
    print('Scaling')
    if Standard_Scale:
        Scalar = StandardScaler()
    else:
        Scalar = MinMaxScaler()

    df[num_cols] = Scalar.fit_transform(df[num_cols])

    del Standard_Scale, Scalar, StandardScaler, MinMaxScaler, num_cols
    gc.collect()

    
    #Text Processing
    print('Text Processing')

    text_cols = ['Project Essay','Project Need Statement']

    #adding more words to 'stopwords'
    extra_words = ['student','students','education','learning','school','schools', 'parent','parents','like',
                   'donotremoveessaydividerthis', 'teachers','teacher', 'children', 'kids','kid','classroom','donotremoveessaydividermy']
    single_l = [x for x in 'abcdefghijklmnopqrstuvwxyz']
    for j in single_l:
        extra_words.append(j)
    extra_words += stopwords.words("english")

    for i in text_cols:
        df[i] = df[i].apply(lambda x: text_cleaner(x,extra_words))

    del text_cleaner, extra_words, single_l, stopwords
    gc.collect()

    #Tfidf
    print('Tfidf')
    tfidf = TfidfVectorizer(max_features=Tf_Features, ngram_range=(1,N_Gram))
    for i in text_cols:
        tfidf.fit(df[i])
        tf_cols = [str(i)+' contains: "'+str(x)+'"' for x in list(tfidf.vocabulary_.keys())]
        df = df.merge(pd.DataFrame(tfidf.transform(df[i]).todense(),columns=tf_cols), left_index=True, right_index=True)

    for i in text_cols:
        del df[i]
    del tfidf, tf_cols

    X = df.drop(['Project Current Status'],axis=1)
    y = df['Project Current Status']
    df_cols = df.columns
    #le_dict

    return(X, y, df_cols, le_dict)