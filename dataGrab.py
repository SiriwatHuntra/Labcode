import pandas as pd
import numpy as np

#get data frame from csv
df = pd.read_csv('HeightWeight.csv')
df.info()
print(df)

def Rand_DF(df ,rng_seed):
    rng_df = df.sample(n=10, random_state = rng_seed)
    return rng_df

def holdout():
    

for i in range(3):
    Rand_DF(df, 1+i).to_csv('rngdata{}.csv'.format(i+1), index= False) 
    print(i)