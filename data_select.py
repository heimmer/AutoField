import pandas as pd
train = pd.read_csv(r'D:\研究生\Semester B\dissertation\AutoField\AutoField\avazu-ctr-prediction\train.csv')
train_p = train[:1000]
train_p.to_csv(r'D:\研究生\Semester B\dissertation\AutoField\AutoField\avazu-ctr-prediction\train_p.csv')