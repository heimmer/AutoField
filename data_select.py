import pandas as pd
test = pd.read_csv(r'D:\研究生\Semester B\dissertation\AutoField\AutoField\avazu-ctr-prediction\test.csv')
test_p = test[:1000]
test_p.to_csv(r'D:\研究生\Semester B\dissertation\AutoField\AutoField\avazu-ctr-prediction\test_p.csv')