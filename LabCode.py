import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statistics import mean 

#Data sampling
def Rand_DF(df ,rng_seed):
    rng_df = df.sample(n=1000, random_state = rng_seed)
    return rng_df

#for sampling observe
def Test_output_handel():
    i = 0
    for i in range(3):
        Rand_DF(df, 1+i).to_csv('rngdata{}.csv'.format(i+1), index= False) 
        print(i)

#make linear model with all data
def LinRe_AllData(df, Y_col):
    X = df.drop(columns=[Y_col])
    Y = df[Y_col]

    #init model with all data
    model = LinearRegression()
    model.fit(X, Y)

    #RMSE
    Y_pred = model.predict(X)
    RMSE = mean_squared_error(Y, Y_pred, squared=False)

    return RMSE

def HoldOut(df, Y_col, testsize):
    X = df.drop(columns=[Y_col])
    Y = df[Y_col]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testsize, random_state=0)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)

    RMSE = mean_squared_error(Y_test, y_pred, squared=False)
    return RMSE

def cross_val(df, Y_col, fold):
    X = df.drop(columns=[Y_col])
    Y = df[Y_col]

    kf = KFold(n_splits=fold, shuffle = True, random_state=0)

    rmse_values = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate RMSE for this fold
        rmse_fold = mean_squared_error(y_test, y_pred, squared=False)
        rmse_values.append(rmse_fold)
    RMSE = mean(rmse_values)
    return RMSE

def run_test3(df, seed, HoldoutSplit, Number_Kfold):
    Y = "Weight"

    #average of holdout
    holdOut_log = []
    for i in range(seed):
        df = Rand_DF(df, i)
        holdOut_log.append(HoldOut(df, Y, HoldoutSplit))
    avr_holdOut = mean(holdOut_log)
    
    #average of holdout
    cross_log = []
    for i in range(seed):
        df = Rand_DF(df, i)
        cross_log.append(cross_val(df, Y, Number_Kfold))
    avr_cross = mean(cross_log)
    

    return holdOut_log, cross_log, avr_holdOut, avr_cross

if __name__ == "__main__":
    #Prepare data set
    #get data frame from csv
    df = pd.read_csv('HeightWeight.csv')
    AllData_RMSE = LinRe_AllData(df, "Weight")
    print(AllData_RMSE)

    #runn_test3(dataframe, seed_round, holdout ratio, cross folds)
    ###################
    ##____Warning____##
    ###################
    ## if seed round > 1000 craeful for lack
    print(run_test3(df, 1000, 0.5, 10))
         
    