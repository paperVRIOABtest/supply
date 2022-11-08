from socket import NI_MAXSERV
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shutil
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
s = 77
## load the data：

X_train = np.load(file = 'X_train_T1.npy')  
X_test = np.load(file='X_test_T1.npy')      
Y_train_pd = pd.read_csv('Y_train_T1.csv')
Y_test_pd = pd.read_csv('Y_test_T1.csv')

Y_train = np.array(Y_train_pd).squeeze(-1)  #77039,
Y_test = np.array(Y_test_pd).squeeze(-1)    #76690,

print(np.shape(Y_train),np.shape(Y_test))


## random choose samples from all samples ##
nc,nt = 2000,8000
np.random.seed(1)
index_choose_train = np.random.permutation(np.shape(Y_train)[0])[0:nc]
X_train = X_train[index_choose_train]
Y_train = Y_train[index_choose_train]
np.random.seed(1)
index = np.random.permutation(np.shape(Y_test)[0])[0:nt] 
X_test = X_test[index]
Y_test = Y_test[index]

print('test:',np.mean(Y_test)-np.mean(Y_train))
print(np.shape(Y_test),np.shape(Y_train))
print(np.shape(X_test),np.shape(X_train))
nc,nt = int(np.shape(Y_train)[0]),int(np.shape(Y_test)[0])

np.random.seed(42)
indexc = np.random.permutation(nc)
indext = np.random.permutation(nt)
ave_choice = 'no'
pool_choice = 'yes'
epoch_num_train = 50
epoch_num_treat = 100

# LSTM structure：
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.emb = nn.Sequential(
            nn.Linear(2,500), 
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(500,64),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=28,         
            num_layers=2,
            batch_first=True,      
        )
        self.out1 = nn.Sequential(
            nn.Linear(28*28,64),
            #nn.Linear(d*28,64),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.out2 = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.out3 = nn.Sequential(
            nn.Linear(32,1)
        )

    def forward(self, x):
        x = self.emb(x)  #x:(batchsize,28,2)
        r_out, (h_n, h_c) = self.rnn(x, None)  #r_out:(batchsize,28,28)
        out = self.out3(self.out2(self.out1(torch.reshape(r_out,((-1,28*28)))))) 
        return out

def evaluation(epoch):
    logit = []   
    lstm.eval() 
    for i,(data, targets) in enumerate(validation_datac):  
        data = data.cuda()
        targets = targets.cuda()
        logits = lstm.forward(data)
        loss = criteon(logits.detach().squeeze(-1), targets).item()  
        logit.append(loss)

    print('epoch_c:',epoch,'validation_loss:',np.mean(np.array(logit)))

    return np.mean(np.array(logit))
    
def evaluationt(epoch):
    logit = []  
    lstmt.eval() 
    for i,(data, targets) in enumerate(validation_datat):  
        data = data.cuda()
        targets = targets.cuda()  
        logits = lstmt.forward(data)  
        loss = criteon(logits.detach().squeeze(-1), targets).item() 
        logit.append(loss)

    print('epoch_t:',epoch,'validation_loss:',np.mean(np.array(logit)))

    return np.mean(np.array(logit))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


#copy the samples for adjusting
X_train_copy,X_test_copy = X_train.copy(),X_test.copy()
Y_train_copy,Y_test_copy = Y_train.copy(),Y_test.copy()

X_train_copy_GBDT,X_test_copy_GBDT = X_train.copy(),X_test.copy()
Y_train_copy_GBDT,Y_test_copy_GBDT = Y_train.copy(),Y_test.copy()

X_train_copy_CUPED,X_test_copy_CUPED = X_train.copy(),X_test.copy()
Y_train_copy_CUPED,Y_test_copy_CUPED = Y_train.copy(),Y_test.copy()
#batchsize:
batch_size = 32 

K = 2

for i in range(K):
    print('now K:',i)   
    
    index_c_validation = indexc[i*(nc//K):(i+1)*(nc//K)]   
    index_t_validation = indext[i*(nt//K):(i+1)*(nt//K)]
    index_c_train = np.concatenate((indexc[0:i*(nc//K)],indexc[(i+1)*(nc//K):nc]))  
    index_t_train = np.concatenate((indext[0:i*(nt//K)],indext[(i+1)*(nt//K):nt]))

 
    validation_X_c,validation_Y_c = X_train[index_c_validation],Y_train[index_c_validation]    
    validation_X_t,validation_Y_t = X_test[index_t_validation],Y_test[index_t_validation]

    train_X_c,train_Y_c = X_train[index_c_train],Y_train[index_c_train]
    train_X_t,train_Y_t = X_test[index_t_train],Y_test[index_t_train]
    #print(np.shape(train_X_c),np.shape(train_Y_c))


    ## LSTM ##
    # one fold：treatment/control
    #1.
    Covariate_c,Response_c = torch.tensor(train_X_c, dtype=torch.float32),torch.tensor(train_Y_c, dtype=torch.float32)  
    #2.
    torch_dataset_c = torch.utils.data.TensorDataset(Covariate_c, Response_c) 
        
    #1.
    Covariate_t,Response_t = torch.tensor(train_X_t, dtype=torch.float32),torch.tensor(train_Y_t, dtype=torch.float32) 
    #2.
    torch_dataset_t = torch.utils.data.TensorDataset(Covariate_t, Response_t)  
        
    #3.
    train_datac = torch.utils.data.DataLoader(torch_dataset_c,      
                                            batch_size=batch_size,
                                            shuffle=True)
    train_datat = torch.utils.data.DataLoader(torch_dataset_t,
                                            batch_size=batch_size,
                                            shuffle=True)               
    #data not in the fold：treatment/control                          
    Covariate_c,Response_c = torch.tensor(validation_X_c, dtype=torch.float32),torch.tensor(validation_Y_c, dtype=torch.float32)  
    torch_dataset_c = torch.utils.data.TensorDataset(Covariate_c, Response_c)  
    Covariate_t,Response_t = torch.tensor(validation_X_t, dtype=torch.float32),torch.tensor(validation_Y_t, dtype=torch.float32)  
    torch_dataset_t = torch.utils.data.TensorDataset(Covariate_t, Response_t) 
    validation_datac = torch.utils.data.DataLoader(torch_dataset_c,    
                                            batch_size=batch_size, 
                                            shuffle=False)
    validation_datat = torch.utils.data.DataLoader(torch_dataset_t,
                                            batch_size=batch_size,
                                            shuffle=False)  
    
    if pool_choice == 'no':
        #control group for adjusting: f_C
        learning_rate = 0.001  
        epochs = epoch_num_train       
        lstm = RNN().cuda()
        optimizer = optim.Adam(lstm.parameters(), learning_rate, weight_decay=1e-4)  
        criteon = torch.nn.MSELoss()

        best_loss = 999.0
    
        for epoch in range(epochs):  
            lstm.train() 
            losses = []
            for ii, (data, target) in enumerate(train_datac):
                data = data.cuda()
                target = target.cuda()
                logits = lstm.forward(data)  
                loss = criteon(logits.squeeze(-1), target)  
                optimizer.zero_grad()   
                loss.backward()         
                optimizer.step()        
                losses.append(loss.item())

            print("epoch:", epoch, "loss:", np.mean(losses))
            loss_val = evaluation(epoch)


            is_best = loss_val < best_loss
            best_loss = min(loss_val, best_loss)
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': lstm.state_dict(),
                            'best_loss': best_loss,
                            'optimizer' : optimizer.state_dict(),
                            }, is_best)
        checkpoint = torch.load('model_best.pth.tar')
        lstm = RNN()
        lstm.load_state_dict(checkpoint['state_dict'])
        lstm.cuda()

        #adjusting: f_C
        Yc_lstm1 = np.array([])  
        for ii,(data, targets) in enumerate(validation_datac):  
            lstm.eval()
            data = data.cuda()
            targets = targets.cuda()
            logits = lstm.forward(data)
            logits = logits.detach().cpu().squeeze(-1).numpy()
            #targets = targets.cpu().numpy()
            Yc_lstm1 = np.concatenate((Yc_lstm1,logits))

            

        Yt_lstm1 = np.array([])
        for ii,(data, targets) in enumerate(validation_datat): 
            lstm.eval()
            data = data.cuda()
            targets = targets.cuda()
            logits = lstm.forward(data)
            logits = logits.detach().cpu().squeeze(-1).numpy()
            targets = targets.cpu().numpy()
            Yt_lstm1 = np.concatenate((Yt_lstm1,logits))

        ##treatment group for adjusting: f_T

        learning_rate = 0.001  
        epochs = epoch_num_treat
        lstmt = RNN().cuda()   
        optimizert = optim.Adam(lstmt.parameters(), learning_rate, weight_decay=1e-4)  
        criteon = torch.nn.MSELoss()

        best_loss = 999.0

        for epoch in range(epochs):  
            lstmt.train() 
            losses = []
            for ii, (data, target) in enumerate(train_datat):
                data = data.cuda()
                target = target.cuda()
                logits = lstmt.forward(data) 
                loss = criteon(logits.squeeze(-1), target)  
                optimizert.zero_grad()  
                loss.backward()  
                optimizert.step()  
                losses.append(loss.item())

            
            print("epoch:", epoch, "loss:", np.mean(losses))
            
            loss_val = evaluationt(epoch)

            is_best = loss_val < best_loss
            best_loss = min(loss_val, best_loss)
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': lstmt.state_dict(),
                            'best_loss': best_loss,
                            'optimizer' : optimizert.state_dict(),
                            }, is_best)
        checkpoint = torch.load('model_best.pth.tar')
        lstmt = RNN()
        lstmt.load_state_dict(checkpoint['state_dict'])
        lstmt.cuda()

        #adjusting f_T
        Yt_lstm2 = np.array([])
        
        for ii,(data, targets) in enumerate(validation_datat): 
            lstmt.eval()
            data = data.cuda()
            targets = targets.cuda()
            logits = lstmt.forward(data)
            logits = logits.detach().cpu().squeeze(-1).numpy()
            #targets = targets.cpu().numpy()
            Yt_lstm2 = np.concatenate((Yt_lstm2,logits))
            

        Yc_lstm2 = np.array([])
        for ii,(data, targets) in enumerate(validation_datac):  
            lstmt.eval()
            data = data.cuda()
            targets = targets.cuda()
            logits = lstmt.forward(data)
            logits = logits.detach().cpu().squeeze(-1).numpy()
            targets = targets.cpu().numpy()
            Yc_lstm2 = np.concatenate((Yc_lstm2,logits))
        
        Yc_lstm = Yc_lstm1*nt/(nt+nc)+Yc_lstm2*nc/(nt+nc)    
        Yt_lstm = Yt_lstm1*nt/(nt+nc)+Yt_lstm2*nc/(nt+nc)

        Y_train_copy[index_c_validation] = Y_train_copy[index_c_validation]-Yc_lstm
        Y_test_copy[index_t_validation] = Y_test_copy[index_t_validation]-Yt_lstm  

    if pool_choice == 'yes':
        #1.
        train_X = np.concatenate((train_X_c,train_X_t),axis=0)
        train_Y = np.concatenate((train_Y_c,train_Y_t))
        Covariate,Response = torch.tensor(train_X, dtype=torch.float32),torch.tensor(train_Y, dtype=torch.float32)  
        #2.
        torch_dataset = torch.utils.data.TensorDataset(Covariate, Response)  
        #3.
        train_data = torch.utils.data.DataLoader(torch_dataset,      
                                                batch_size=batch_size,
                                                shuffle=True)                
        # 1.                      
        validation_X = np.concatenate((validation_X_c,validation_X_t),axis=0)
        validation_Y = np.concatenate((validation_Y_c,validation_Y_t),axis=0)
        #2.
        Covariate,Response = torch.tensor(validation_X, dtype=torch.float32),torch.tensor(validation_Y, dtype=torch.float32)  
        torch_dataset = torch.utils.data.TensorDataset(Covariate, Response) 
        #3.
        validation_data = torch.utils.data.DataLoader(torch_dataset,   
                                                batch_size=batch_size, 
                                                shuffle=False) 

        learning_rate = 0.001 
        epochs = epoch_num_train        
        lstm = RNN().cuda()
        optimizer = optim.Adam(lstm.parameters(), learning_rate, weight_decay=1e-4) 
        criteon = torch.nn.MSELoss()

        best_loss = 999.0
    
        for epoch in range(epochs):  
            lstm.train() 
            losses = []
            for ii, (data, target) in enumerate(train_data):
                data = data.cuda()
                target = target.cuda()
                logits = lstm.forward(data)  
                loss = criteon(logits.squeeze(-1), target) 
                optimizer.zero_grad()   
                loss.backward()         
                optimizer.step()        
                losses.append(loss.item())

            print("epoch:", epoch, "loss:", np.mean(losses))
            loss_val = evaluation(epoch)


            is_best = loss_val < best_loss
            best_loss = min(loss_val, best_loss)
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': lstm.state_dict(),
                            'best_loss': best_loss,
                            'optimizer' : optimizer.state_dict(),
                            }, is_best)
        checkpoint = torch.load('model_best.pth.tar')
        lstm = RNN()
        lstm.load_state_dict(checkpoint['state_dict'])
        lstm.cuda()

        Yc_lstm1 = np.array([])  
        for ii,(data, targets) in enumerate(validation_datac): 
            lstm.eval()
            data = data.cuda()
            targets = targets.cuda()
            logits = lstm.forward(data)
            logits = logits.detach().cpu().squeeze(-1).numpy()
            #targets = targets.cpu().numpy()
            Yc_lstm1 = np.concatenate((Yc_lstm1,logits))

        Yt_lstm1 = np.array([])
        for ii,(data, targets) in enumerate(validation_datat):  
            lstm.eval()
            data = data.cuda()
            targets = targets.cuda()
            logits = lstm.forward(data)
            logits = logits.detach().cpu().squeeze(-1).numpy()
            targets = targets.cpu().numpy()
            Yt_lstm1 = np.concatenate((Yt_lstm1,logits))
        
        Y_train_copy[index_c_validation] = Y_train_copy[index_c_validation]-Yc_lstm1
        Y_test_copy[index_t_validation] = Y_test_copy[index_t_validation]-Yt_lstm1 

    ############################# GBDT sep ###########################
    if pool_choice == 'no':
        if ave_choice == 'yes':
            X_pre = np.mean(train_X_c,axis=1)  ##1. average over time
        else:
            X_pre = train_X_c.reshape(-1,56)
        Y_res = train_Y_c
        model = GradientBoostingRegressor()
        model.fit(X_pre,Y_res)        
        
        if ave_choice == 'yes':
            X_esti_c,X_esti_t = np.mean(validation_X_c,axis=1),np.mean(validation_X_t,axis=1)   
        else: 
            X_esti_c,X_esti_t = validation_X_c.reshape(-1,56),validation_X_t.reshape(-1,56)
        Y_esti_c,Y_esti_t = validation_Y_c,validation_Y_t

        yc_pred = model.predict(X_esti_c)
        yt_pred = model.predict(X_esti_t)
        yc = Y_esti_c - yc_pred*nt/(nt+nc)      
        yt = Y_esti_t - yt_pred*nt/(nt+nc)
        
        if ave_choice == 'yes':
            X_pre = np.mean(train_X_t,axis=1)  ##2. average over time
        else:
            X_pre = train_X_t.reshape(-1,56)
        Y_res = train_Y_t
        model = GradientBoostingRegressor()
        model.fit(X_pre,Y_res)        

        yc_pred = model.predict(X_esti_c)
        yt_pred = model.predict(X_esti_t)
        yc -= yc_pred*nc/(nt+nc)
        yt -= yt_pred*nc/(nt+nc)               

        Y_train_copy_GBDT[index_c_validation] = yc
        Y_test_copy_GBDT[index_t_validation] = yt   

    ### GBDT pool ###
    if pool_choice == 'yes':
        if ave_choice == 'yes':
            X_c,X_t = np.mean(train_X_c,axis=1),np.mean(train_X_t,axis=1)
        else:
            X_c,X_t = train_X_c.reshape(-1,56),train_X_t.reshape(-1,56)
        Y_c,T_t = train_Y_c,train_Y_t
        X_pre = np.concatenate((X_c,X_t),axis=0)
        Y_res = np.concatenate((Y_c,T_t))
        model = GradientBoostingRegressor()
        model.fit(X_pre,Y_res)      
        
        if ave_choice == 'yes':
            X_esti_c,X_esti_t = np.mean(validation_X_c,axis=1),np.mean(validation_X_t,axis=1) 
        else:
            X_esti_c,X_esti_t = validation_X_c.reshape(-1,56),validation_X_t.reshape(-1,56)
        yc_pred = model.predict(X_esti_c)
        yt_pred = model.predict(X_esti_t)

        Y_esti_c,Y_esti_t = validation_Y_c,validation_Y_t
        yc = Y_esti_c - yc_pred     
        yt = Y_esti_t - yt_pred
        Y_train_copy_GBDT[index_c_validation] = yc
        Y_test_copy_GBDT[index_t_validation] = yt 

    ########################### CUPED sep ############################
    if pool_choice == 'no':
        if ave_choice == 'yes':
            X_pre = np.mean(train_X_c,axis=1)  ##1
        else:
            X_pre = train_X_c.reshape(-1,56)
        Y_res = train_Y_c
        model = LinearRegression()
        model.fit(X_pre,Y_res)
        
        if ave_choice == 'yes':
            X_esti_c,X_esti_t = np.mean(validation_X_c,axis=1),np.mean(validation_X_t,axis=1)
        else:
            X_esti_c,X_esti_t = validation_X_c.reshape(-1,56),validation_X_t.reshape(-1,56)
        Y_esti_c,Y_esti_t = validation_Y_c,validation_Y_t

        yc_pred = model.predict(X_esti_c)
        yt_pred = model.predict(X_esti_t)
        yc = Y_esti_c - yc_pred*nt/(nt+nc)
        yt = Y_esti_t - yt_pred*nt/(nt+nc)

        rc = 1-np.mean((Y_esti_c-yc_pred)**2)/np.var(Y_esti_c)
        listrc_CUPED.append(rc)
        if ave_choice == 'yes':
            X_pre = np.mean(train_X_t,axis=1)  ##2
        else:
            X_pre = train_X_t.reshape(-1,56)
        Y_res = train_Y_t
        model = LinearRegression()
        model.fit(X_pre,Y_res)

        yc_pred = model.predict(X_esti_c)
        yt_pred = model.predict(X_esti_t)
        yc -= yc_pred*nc/(nt+nc)
        yt -= yt_pred*nc/(nt+nc)

        Y_train_copy_CUPED[index_c_validation] = yc
        Y_test_copy_CUPED[index_t_validation] = yt   

    ### CUPED pool ###
    if pool_choice == 'yes':
        if ave_choice == 'yes':
            X_c,X_t = np.mean(train_X_c,axis=1),np.mean(train_X_t,axis=1)
        else:
            X_c,X_t = train_X_c.reshape(-1,56),train_X_t.reshape(-1,56)
        Y_c,T_t = train_Y_c,train_Y_t
        X_pre = np.concatenate((X_c,X_t),axis=0)
        Y_res = np.concatenate((Y_c,T_t))
        model = LinearRegression(fit_intercept=False)
        model.fit(X_pre,Y_res)      
        
        if ave_choice == 'yes':
            X_esti_c,X_esti_t = np.mean(validation_X_c,axis=1),np.mean(validation_X_t,axis=1) 
        else:
            X_esti_c,X_esti_t = validation_X_c.reshape(-1,56),validation_X_t.reshape(-1,56)
        yc_pred = model.predict(X_esti_c)
        yt_pred = model.predict(X_esti_t)

        Y_esti_c,Y_esti_t = validation_Y_c,validation_Y_t
        yc = Y_esti_c - yc_pred     
        yt = Y_esti_t - yt_pred
        
        Y_train_copy_CUPED[index_c_validation] = yc
        Y_test_copy_CUPED[index_t_validation] = yt 

#final results：

## ATE、variance:
a = np.mean(Y_test_copy)-np.mean(Y_train_copy)        #LSTM-ATE
b = np.var(Y_test_copy)/nt+np.var(Y_train_copy)/nc    #LSTM-variance
c = np.mean(Y_test)-np.mean(Y_train)                  #no adjustment
d = np.var(Y_test)/nt+np.var(Y_train)/nc
e = np.mean(Y_test_copy_GBDT)-np.mean(Y_train_copy_GBDT)           #GBDT
f = np.var(Y_test_copy_GBDT)/nt+np.var(Y_train_copy_GBDT)/nc
g = np.mean(Y_test_copy_CUPED)-np.mean(Y_train_copy_CUPED)         #CUPED
h = np.var(Y_test_copy_CUPED)/nt+np.var(Y_train_copy_CUPED)/nc


print('LSTM_ATE_Var:',a,b/d)
print('No_ATE_Var:',c,d/d)
print('GBDT_ATE_Var:',e,f/d)
print('CUPED_ATE_Var:',g,h/d)
print('DiM_Var:',d)

print('over')