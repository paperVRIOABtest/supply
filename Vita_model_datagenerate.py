import pandas as pd
import numpy as np
import datetime
import time

#fct def
#count data difference
def Caltime(date1,date2): 
    date1=datetime.datetime.strptime(date1,"%Y-%m-%d")
    date2=datetime.datetime.strptime(date2,"%Y-%m-%d")
    return (date2-date1).days

##count target date
def Caldate(date1,interval): 
    date1=datetime.datetime.strptime(date1,"%Y-%m-%d")
    end_date = date1 + datetime.timedelta(days=interval)
    return end_date.strftime('%Y-%m-%d')


def Gen_single_ad(initium='2021-01-01',user=1000,quality=15,seed=0,length=1000,theta=[10,50],eta=[0.1,0.1],q=[-0.2,1.2]):
    np.random.seed(seed)
    State=pd.DataFrame()
    State['id']=Caltime('2020-12-31',initium)*100000+np.arange(1,user+1)
    State['treatment']=np.random.binomial(1,0.5,user)
    State['activation_date']=initium
    State['action_date']=initium
    State['loyalty']=pd.Series(np.random.exponential(quality,user))+1
    State['strength']=pd.Series(np.random.exponential(1,user)) 
    State['vita']=1
    State['alive']=State['vita']>np.random.uniform(0,1,user)
    State['v_hidden']=(State['strength']*np.sqrt(State['vita'])*np.random.exponential(1,user))
    State['v_view']=State['v_hidden']*np.random.normal(theta[0],theta[0]*eta[0],user).astype(int)
    State['v_time']=State['v_hidden']*np.random.normal(theta[1],theta[1]*eta[1],user)
    Record=State.copy(deep=1)
    for t in range(1,length):
        State['action_date']=Caldate(initium,t)
        # if length-t <= 29: #####third kind of treatment needs to change this additionally
        #     State['vita']=(State['vita']-(1/State['loyalty']>np.random.uniform(0,1,user))
        #                    *(np.random.uniform(q[0],q[1],user)*State['vita']-0.02*State['treatment'])).clip(0,1)
        # else:
        #     State['vita']=(State['vita']-(1/State['loyalty']>np.random.uniform(0,1,user))
        #                    *np.random.uniform(q[0],q[1],user)*State['vita']).clip(0,1)

        State['vita']=(State['vita']-(1/State['loyalty']>np.random.uniform(0,1,user))
                           *np.random.uniform(q[0],q[1],user)*State['vita']).clip(0,1)
        State['vita']=State['vita']*(State['vita']>0)
        State=State[State['vita']>0]
        user=len(State)
        State['alive']=State['vita']>np.random.uniform(0,1,user)
        
        # #####no treatment:
        # State['v_hidden']=(State['strength']*np.sqrt(State['vita'])*np.random.exponential(1,user))
        # State['v_view']=State['v_hidden']*np.random.normal(theta[0],theta[0]*eta[0],user).astype(int)
        # State['v_time']=State['v_hidden']*np.random.normal(theta[1],theta[1]*eta[1],user)
        # Record=Record.append(State)
        
        ####first kind of treatment
        # if length-t <= 29:
        #     State['v_hidden']=((State['strength']+0.5*State['treatment']*np.random.exponential(1,user))*np.sqrt(State['vita'])*np.random.exponential(1,user))
        # else:
        #     State['v_hidden']=(State['strength']*np.sqrt(State['vita'])*np.random.exponential(1,user))
        # State['v_view']=State['v_hidden']*np.random.normal(theta[0],theta[0]*eta[0],user).astype(int)
        # State['v_time']=State['v_hidden']*np.random.normal(theta[1],theta[1]*eta[1],user)
        # Record=Record.append(State)
        
        #####second kind of treatment
        # State['v_hidden']=(State['strength']*np.sqrt(State['vita'])*np.random.exponential(1,user))
        # if length-t <= 29:
        #    State['v_view']=State['v_hidden']*np.random.normal(theta[0],theta[0]*eta[0],user).astype(int)+0.1*State['treatment']*State['v_hidden']*np.random.normal(theta[0],theta[0]*eta[0],user)
        # else:
        #    State['v_view']=State['v_hidden']*np.random.normal(theta[0],theta[0]*eta[0],user).astype(int)
        # State['v_time']=State['v_hidden']*np.random.normal(theta[1],theta[1]*eta[1],user)
        # Record=Record.append(State)
        
        #####third kind of treatment
        State['v_hidden']=(State['strength']*np.sqrt(State['vita'])*np.random.exponential(1,user))
        State['v_view']=State['v_hidden']*np.random.normal(theta[0],theta[0]*eta[0],user).astype(int)
        State['v_time']=State['v_hidden']*np.random.normal(theta[1],theta[1]*eta[1],user)
        Record=Record.append(State)
    return Record

Base=pd.DataFrame()
np.random.seed(0)
quality=20
user=5000
t1=Caldate('2021-01-01',365-56)
t2=Caldate('2021-01-01',365-28)
t3=Caldate('2021-01-01',365)
for t in range(366):
    timer=time.time()
    quality=quality*(1+np.random.normal(0,0.01))
    Local=Gen_single_ad(initium=Caldate('2021-01-01',t),user=user,quality=quality,seed=t,length=366-t)
    Base=Base.append(Local[(Local.action_date>=t1)&(Local.action_date<t3)])
    print('Day {} is done. {:.0f}s used.  '.format(t,time.time()-timer),end='\r')
print('Done.                      ')
Base=Base.reset_index(drop=True)
Observe=Base[(Base.alive==1)&(Base.activation_date<=t1)][['id','treatment','activation_date','action_date','v_view','v_time']].reset_index(drop=True)
#Base.to_csv('Hidden_truth.csv',index=False)
#Observe.to_csv('Observed_data.csv',index=False)

####control and treatment groups
### generate X/Y
def generateXY(Base,hist_length=28,future_length=28,n_feat=2):
    dates=Base.action_date.drop_duplicates()
    if (len(dates)!=hist_length+future_length)|(Caltime(min(dates),max(dates))!=hist_length+future_length-1):
        print('Wrong Input.')
    infms=['id']
    base_date=min(dates)
    Time_acc=pd.DataFrame(columns=infms)
    for i in range(hist_length):
        Time_local=Base[Base['action_date']==Caldate(base_date,i)].drop('action_date',axis=1)
        Time_acc=pd.merge(Time_acc,Time_local,on=infms,how='outer',suffixes=('',str(i)))  ##user may not be alive today，supply with NA
    Y=Base[Base['action_date']>=Caldate(base_date,hist_length)][['id','v_view']].groupby(['id'],as_index=False)['v_view'].sum()
    Y.columns=['id','y']
    Y['y']/=future_length
    full_XY=pd.merge(Time_acc.fillna(-99),Y,on=['id'],how='outer').fillna(0)
    
    ####
    target_column=['y']
    predictors = [x for x in list(full_XY.columns) if x not in target_column+infms]
    full_XY[predictors] = full_XY[predictors]/full_XY[predictors].max()
    Infm=full_XY[infms]
    X=np.array(full_XY[predictors]).reshape(-1,hist_length,n_feat)
    Y=np.array(full_XY[target_column])
    return X,Y

backward=28
forward=28
n_feat=2
Train=Observe[Observe.treatment==0]
Test=Observe[Observe.treatment==1]
X_train,Y_train=generateXY(Train[['id','action_date','v_view','v_time']], hist_length=28,future_length=28,n_feat=2)
X_test,Y_test=generateXY(Test[['id','action_date','v_view','v_time']], hist_length=28,future_length=28,n_feat=2)

a = np.mean(Y_test)-np.mean(Y_train)
print('groundtruth:',a)

print(np.shape(Y_train),np.shape(X_train))
print(np.shape(Y_test),np.shape(X_test))
Y_train_pd = pd.DataFrame(Y_train)

np.save(file='X_train_T1.npy', arr=X_train)
Y_train_pd.to_csv('Y_train_T1.csv',index=False)

Y_test_pd = pd.DataFrame(Y_test)
np.save(file='X_test_T1.npy', arr=X_test)
Y_test_pd.to_csv('Y_test_T1.csv',index=False)

print('over')