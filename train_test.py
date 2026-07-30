import argparse
import math
import time
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib
import random
from util import *
from trainer import Optim
import sys
from random import randrange
from matplotlib import pyplot as plt
import time

plt.rcParams['savefig.dpi'] = 1200

runs=20

def inverse_diff_2d(output, I,shift):
    output[0,:]=torch.exp(output[0,:]+torch.log(I+shift))-shift
    for i in range(1,output.shape[0]):
        output[i,:]= torch.exp(output[i,:]+torch.log(output[i-1,:]+shift))-shift
    return output

def inverse_diff_3d(output, I,shift):
    output[:,0,:]=torch.exp(output[:,0,:]+torch.log(I+shift))-shift
    for i in range(1,output.shape[1]):
        output[:,i,:]=torch.exp(output[:,i,:]+torch.log(output[:,i-1,:]+shift))-shift
    return output


def plot_data(data,title):
    x=range(1,len(data)+1)
    plt.plot(x,data,'b-',label='Actual')
    plt.legend(loc="best",prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03,fontsize=18)
    plt.ylabel("Trend",fontsize=15)
    plt.xlabel("Month",fontsize=15)
    locs, labs = plt.xticks() 
    plt.xticks(rotation='vertical',fontsize=13) 
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    plt.show()


# for figure display, we rename columns
def consistent_name(name):

    if name=='CAPTCHA' or name=='DNSSEC' or name=='RRAM':
        return name

    #e.g., University of london
    if not name.isupper():
        words=name.split(' ')
        result=''
        for i,word in enumerate(words):
            if len(word)<=2: #e.g., "of"
                result+=word
            else:
                result+=word[0].upper()+word[1:]
            
            if i<len(words)-1:
                result+=' '

        return result
    

    words= name.split(' ')
    result=''
    for i,word in enumerate(words):
        if len(word)<=3 or '/' in word or word=='MITM' or word =='SIEM':
            result+=word
        else:
            result+=word[0]+(word[1:].lower())
        
        if i<len(words)-1:
            result+=' '
        
    return result

#computes and saves validation/testing error to a text file given a single node's prediction and actual curve values
def save_metrics_1d(predict, test, title, type):
    #RRSE according to Lai et.al - numerator
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator

    #Relative Absolute Error RAE  - numerator
    sum_absolute_diff= torch.sum(torch.abs(test - predict))

    #RRSE according to Lai et.al - denominator
    test_s=test
    mean_all = torch.mean(test_s) # calculate the mean of each column in test
    diff_r = test_s - mean_all # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    # Handle zero division safely for flat actual data signals
    if root_sum_squared_r == 0:
        rrse = 0.0
    else:
        rrse = root_sum_squared / root_sum_squared_r

    #Relative Absolute Error RAE - denominator
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements
    
    # Handle zero division safely for RAE
    if sum_absolute_r == 0:
        rae = 0.0
    else:
        rae = sum_absolute_diff / sum_absolute_r 
        rae = rae.item()

    title=title.replace('/','_')
    with open('model/Bayesian/'+type+'/'+title+'_'+type+'.txt',"w") as f:
      f.write('rse:'+str(rrse)+'\n')
      f.write('rae:'+str(rae)+'\n')
      f.close()


#plots predicted curve with actual curve. The x axis can be adjusted as needed
def plot_predicted_actual(predicted, actual, title, type,variance, confidence_95):

    #all months
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    M=[]
    for year in range (4,26):   
        for month in months:
            if year==4 and month not in ['Oct','Nov','Dec']:
                continue
            M.append(month+'-'+str(year))   
    M2=[]
    p=[]
    
    #last 3 years
    if type=='Testing':
        M=M[-len(predicted):]
        for index,value in enumerate(M):
            if 'Dec' in M[index] or 'Mar' in M[index] or 'Jun' in M[index] or 'Sep' in M[index]:
                M2.append(M[index])
                p.append(index+1) 
    
    else: ##Validation x axis: September 2018 to August 2021
        M=M[167:203]
        for index,value in enumerate(M):
            if 'Dec' in M[index] or 'Mar' in M[index] or 'Jun' in M[index] or 'Sep' in M[index]:
                M2.append(M[index])
                p.append(index+1) 

    x=range(1,len(predicted)+1)
    plt.plot(x,actual,'b-',label='Actual')
    plt.plot(x,predicted,'--', color='purple',label='Predicted')
    
    # Plot the confidence interval as a shaded region
    # plt.fill_between(x, predicted-confidence_95.numpy(), predicted+confidence_95.numpy(), alpha=0.2, color='green', label='95% Confidence')
    # Move confidence to CPU before converting to NumPy
    conf_np = confidence_95.cpu().numpy()
    # If 'predicted' is also a PyTorch CUDA tensor, you'll need to do the same:
    pred_np = predicted.cpu().numpy() if hasattr(predicted, 'cpu') else predicted
   
    plt.fill_between(x, pred_np - conf_np, pred_np + conf_np, alpha=0.3, color='deeppink', label='95% Confidence')
    plt.legend(loc="best",prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03,fontsize=18)
    plt.ylabel("Trend",fontsize=15)
    plt.xlabel("Month",fontsize=15)
    locs, labs = plt.xticks() 
    plt.xticks(ticks = p ,labels = M2, rotation='vertical',fontsize=13) 
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    title=title.replace('/','_')
    plt.savefig('model/Bayesian/'+type+'/'+title+'_'+type+'.png', bbox_inches="tight")
    plt.savefig('model/Bayesian/'+type+'/'+title+'_'+type+".pdf", bbox_inches = "tight", format='pdf')


    plt.show(block=False)
    plt.close()


#symmetric mean absolute percentage error (optional)
def s_mape(yTrue,yPred):
  mape=0
  for i in range(len(yTrue)):
    mape+= abs(yTrue[i]-yPred[i])/ (abs(yTrue[i])+abs(yPred[i]))
  mape/=len(yTrue)

  return mape

#for testing the model on unseen data, a sliding window can be used when the output period of the model is smaller than the target period to be forecasted.
#The sliding window uses the output from previous step as input of the next step.
#In our case, the window was not slided (we predicted 36 months and the model by default predicts 36 months)
def evaluate_sliding_window(data, test_window, model,
                            evaluateL2, evaluateL1,
                            n_input, is_plot):

    # model.eval()  # keep commented for Bayesian inference

    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0

    predict = None
    test = None
    variance = None
    confidence_95 = None

    sum_squared_diff = 0
    sum_absolute_diff = 0

    r = 0

    print('testing node =', r)

    ####################################################################
    # NEW:
    # test_window is now:
    #
    # [X_test , Y_test]
    #
    ####################################################################
    X_test = test_window[0]
    Y_test = test_window[1]

    print("Number of testing samples =", X_test.shape[0])

    ####################################################################
    # Process EVERY testing sample independently
    ####################################################################
    for sample in range(X_test.shape[0]):

        # print("Testing sample",
        #       sample + 1,
        #       "/",
        #       X_test.shape[0])

        x_input = X_test[sample].clone()

        target = Y_test[sample].clone()

        sample_predict = None
        sample_truth = None
        sample_var = None
        sample_conf = None

        for i in range(0, target.shape[0], data.out_len):
            # print('**************x_input*******************')
            # print(x_input[:,r])#prints 1 random column in the sliding window
            # print('**************-------*******************')

            X = torch.unsqueeze(x_input,dim=0)
            X = torch.unsqueeze(X,dim=1)
            X = X.transpose(2,3)
            X = X.to(torch.float)


            y_true = target[i:i+data.out_len].clone()

            # Bayesian estimation
            num_runs = runs

            # Create a list to store the outputs
            outputs = []


            # Use model to predict next time step
            for _ in range(num_runs):
                with torch.no_grad():
                    X = X.to(device)
                    output = model(X)  
                    y_pred = output[-1, :, :,-1].clone()
                    #if this is the last predicted window and it exceeds the test window range
                    if y_pred.shape[0]>y_true.shape[0]:
                        y_pred=y_pred[:-(y_pred.shape[0]-y_true.shape[0]),]
                outputs.append(y_pred)

            # Stack the outputs along a new dimension
            outputs = torch.stack(outputs)


            y_pred=torch.mean(outputs,dim=0)
            var = torch.var(outputs, dim=0)#variance
            std_dev = torch.std(outputs, dim=0)#standard deviation

            # Calculate 95% confidence interval
            z=1.96
            confidence=z*std_dev/math.sqrt(num_runs)



            #shift the sliding window
            if data.P<=data.out_len:
                x_input = y_pred[-data.P:].clone()
            else:
                x_input = torch.cat([x_input[ -(data.P-data.out_len):, :].clone(), y_pred.clone()], dim=0)


            # print('----------------------------Predicted months', str(i + 1), 'to', str(i + data.out_len), '--------------------------------------------------')            print(y_pred.shape,y_true.shape)
            # y_pred_o=y_pred
            # y_true_o=y_true
            # for z in range(y_true.shape[0]):
            #     print(y_pred_o[z,r],y_true_o[z,r]) #only one col
            # print('------------------------------------------------------------------------------------------------------------')


            if sample_predict is None:
                sample_predict = y_pred
                sample_truth = y_true
                sample_var = var
                sample_conf = confidence
            else:
                sample_predict = torch.cat((sample_predict, y_pred))
                sample_truth = torch.cat((sample_truth, y_true))
                sample_var = torch.cat((sample_var, var))
                sample_conf = torch.cat((sample_conf, confidence))


        ####################################################################
        # Collect predictions from this testing sample
        ####################################################################
        if predict is None:
            predict = sample_predict.unsqueeze(0)
            test = sample_truth.unsqueeze(0)
            variance = sample_var.unsqueeze(0)
            confidence_95 = sample_conf.unsqueeze(0)
        else:
            predict = torch.cat(
                (predict,
                    sample_predict.unsqueeze(0)),
                dim=0)

            test = torch.cat(
                (test,
                    sample_truth.unsqueeze(0)),
                dim=0)

            variance = torch.cat(
                (variance,
                    sample_var.unsqueeze(0)),
                dim=0)

            confidence_95 = torch.cat(
                (confidence_95,
                    sample_conf.unsqueeze(0)),
                dim=0)


    scale = data.scale.expand(
        test.size(0),
        test.size(1),
        data.m
    )
    scale=scale.to(device)
    predict=predict.to(device)
    test = test.to(device)
    variance = variance.to(device)
    confidence_95= confidence_95.to(device)

    #inverse normalisation
    predict *= scale
    test *= scale
    variance *= scale
    confidence_95 *= scale


    #Relative Squared Error RSE according to Lai et.al - numerator
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    #Relative Absolute Error RAE - numerator
    sum_absolute_diff= torch.sum(torch.abs(test - predict))# numerator


    #Root Relative Squared Error RRSE according to Lai et.al - numerator
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator
    
    #Root Relative Squared Error RRSE according to Lai et.al - denominator
    test_s=test
    mean_all = torch.mean(test_s, dim=(0,1))

    diff_r = test_s - mean_all.expand(
        test_s.size(0),
        test_s.size(1),
        data.m
    )
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    rrse=root_sum_squared/root_sum_squared_r
    print('rrse=',root_sum_squared,'/',root_sum_squared_r)

    #Relative Absolute Error RAE - denominator
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements - denominator
    #Relative Absolute Error RAE
    rae=sum_absolute_diff/sum_absolute_r 
    rae=rae.item()
###########################################################################################################


    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)

    # 1. Calculate the denominator first
    denominator = sigma_p * sigma_g

    # 2. Divide safely while ignoring static evaluation zero-division warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0)
        correlation = np.where(denominator != 0, numerator / denominator, 0.0)

    # 3. Filter out the nodes where the actual data was completely flat, then take the average
    correlation = (correlation[index]).mean()

    # s-mape
    smape = 0

    for s in range(Ytest.shape[0]):
        for node in range(Ytest.shape[2]):
            smape += s_mape(
                Ytest[s,:,node],
                predict[s,:,node]
            )

    smape /= (Ytest.shape[0] * Ytest.shape[2])

    #plot predicted vs actual and save errors to file
    counter=0
    if is_plot:
        print("plotting in Test...")
        for v in range(r,r+37):
            col=v%data.m
            
            node_name=DataLoaderS.col[col]
            #node_name=consistent_name(node_name)
            
            #save error to file
            save_metrics_1d(torch.from_numpy(predict[-1,:,col]),torch.from_numpy(Ytest[-1,:,col]),node_name,'Testing')
            #plot
            plot_predicted_actual(predict[-1,:,col],Ytest[-1,:,col],node_name, 'Testing',variance[-1,:,col],confidence_95[-1,:,col])
            counter+=1

    return rrse,rae,correlation, smape



def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot):
    #model.eval()# To get Bayesian estimation, we must comment out this line
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance=None
    confidence_95=None
    sum_squared_diff=0
    sum_absolute_diff=0
    r=0 #we choose any node index for printing (debugging)
    #print('validation r=',str(r))

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)

        # Bayesian estimation
        num_runs = runs

        # Create a list to store the outputs
        outputs = []

        # Run the model multiple times (10)
        with torch.no_grad():
            for _ in range(num_runs):
                X = X.to(device)
                output = model(X)
                output = torch.squeeze(output)
                if len(output.shape) == 1 or len(output.shape) == 2:
                    output = output.unsqueeze(dim=0)
                outputs.append(output)
            

        # Stack the outputs along a new dimension
        outputs = torch.stack(outputs)

        # Calculate mean, variance, and standard deviation
        mean = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)#variance
        std_dev = torch.std(outputs, dim=0)#standard deviation

        # Calculate 95% confidence interval
        z=1.96
        confidence=z*std_dev/math.sqrt(num_runs)

        output=mean #we will consider the mean to be the prediction

        scale = data.scale.expand(Y.size(0), Y.size(1), data.m) #scale will have the max of each column (37 max values)
        
        #inverse normalisation
        output*=scale
        Y*=scale
        var*=scale
        confidence*=scale

        if predict is None:
            predict = output
            test = Y
            variance=var
            confidence_95=confidence
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
            variance= torch.cat((variance, var))
            confidence_95=torch.cat((confidence_95,confidence))


        # print('EVALUATE RESULTS:')
        # scale = data.scale.expand(Y.size(0), Y.size(1), data.m) #scale will have the max of each column (37 max values)
        # y_pred_o=output
        # y_true_o=Y
        # for z in range(Y.shape[1]):
        #     print(y_pred_o[0,z,r],y_true_o[0,z,r]) #only one col
        
        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()
        n_samples += (output.size(0) * output.size(1) * data.m)

        #RRSE according to Lai et.al
        sum_squared_diff += torch.sum(torch.pow(Y - output, 2))
        #Relative Absolute Error RAE - numerator
        sum_absolute_diff+=torch.sum(torch.abs(Y - output))

    #The below 2 lines are not used
    rse = math.sqrt(total_loss / n_samples) / data.rse 
    rae = (total_loss_l1 / n_samples) / data.rae 

    #RRSE according to Lai et.al - numerator
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator
    
    #RRSE according to Lai et.al - denominator
    test_s=test
    mean_all = torch.mean(test_s, dim=(0,1)) # calculate the mean of each column in test
    diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1), data.m) # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    rrse=root_sum_squared/root_sum_squared_r #RRSE

    #Relative Absolute Error RAE
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements - denominator
    rae=sum_absolute_diff/sum_absolute_r # RAE
    rae=rae.item()


    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    
    # Calculate denominator safely to handle flat/constant signals
    denominator = sigma_p * sigma_g
    
    # Pearson's correlation coefficient with safe division handling wrapped to suppress static evaluation warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0)
        correlation = np.where(denominator != 0, numerator / denominator, 0.0)
        
    correlation = (correlation[index]).mean()

    #s-mape
    smape = 0
    for x in range(Ytest.shape[0]):
        for z in range(Ytest.shape[2]):
            smape+=s_mape(Ytest[x,:,z],predict[x,:,z])
    smape/=Ytest.shape[0]*Ytest.shape[2]


    #plot actual vs predicted curves and save errors to file
    counter=0
    if is_plot:
        print("plotting in Validation...")
        for v in range(r,r+37):
            col=v%data.m
            node_name=DataLoaderS.col[col]
            #node_name=consistent_name(node_name)
            save_metrics_1d(torch.from_numpy(predict[-1,:,col]),torch.from_numpy(Ytest[-1,:,col]),node_name,'Validation')
            plot_predicted_actual(predict[-1,:,col],Ytest[-1,:,col],node_name, 'Validation', variance[-1,:,col], confidence_95[-1,:,col])
            counter+=1
    return rrse, rae, correlation, smape


# def train(data, X, Y, model, criterion, optim, batch_size):
#     model.train()
#     total_loss = 0
#     n_samples = 0
#     iter = 0

#     for X, Y in data.get_batches(X, Y, batch_size, True):
#         model.zero_grad()
#         #temp_X=X
#         X = torch.unsqueeze(X,dim=1)
#         X = X.transpose(2,3)
#         if iter % args.step_size == 0:
#             perm = np.random.permutation(range(args.num_nodes))
#         num_sub = int(args.num_nodes / args.num_split)

#         for j in range(args.num_split):
#             if j != args.num_split - 1:
#                 id = perm[j * num_sub:(j + 1) * num_sub]
#             else:
#                 id = perm[j * num_sub:]

#             id = torch.tensor(id).to(device)
#             tx = X[:, :, :, :] #id was in third colum
#             ty = Y[:, :, :] #id was in third colum
            
#             # Bayesian estimation
#             num_runs = runs

#             # Create a list to store the outputs
#             outputs = []

#             # Run the model multiple times
#             for _ in range(num_runs):
#                 tx=tx.to(device)
#                 output = model(tx)
#                 output = torch.squeeze(output,3)
#                 outputs.append(output)
            

#             # Stack the outputs along a new dimension
#             outputs = torch.stack(outputs)

#             # print(outputs[0][0][0])
#             # print(outputs[1][0][0])
#             # sys.exit()

#             # Calculate mean, variance, and standard deviation
#             mean = torch.mean(outputs, dim=0)
#             var = torch.var(outputs, dim=0)#variance
#             std_dev = torch.std(outputs, dim=0)#standard deviation
            
#             # Calculate 95% confidence interval
#             z=1.96
#             confidence=z*std_dev/torch.sqrt(torch.tensor(num_runs))

#             output=mean #we will consider the mean to be the prediction
            
   
           
#             scale = data.scale.expand(output.size(0), output.size(1), data.m)
#             scale = scale[:,:,:] #id was in third colum
            
#             output*=scale #by Zaid
#             ty*=scale
            
#             # #inverse diff
#             # output=inverse_diff_3d(output,I,data.shift)
#             # ty=inverse_diff_3d(ty,I,data.shift)

#             loss = criterion(output, ty)
#             loss.backward()
#             total_loss += loss.item()
#             n_samples += (output.size(0) * output.size(1) * data.m)
            
#             # perform gradient clipping
#             #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             grad_norm = optim.step()

#         if iter%1==0:
#             print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * output.size(1)* data.m)))
#         iter += 1
#     return total_loss / n_samples

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]

            id = torch.tensor(id).to(device)
            tx = X[:, :, :, :] 
            ty = Y[:, :, :] 
            model = model.to(device)
            output = model(tx)           
            output = torch.squeeze(output,3)
            scale = data.scale.expand(output.size(0), output.size(1), data.m)
            scale = scale[:,:,:] 
            
            output*=scale 
            ty*=scale


            loss = criterion(output, ty)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * output.size(1) * data.m)
            
            grad_norm = optim.step()

        if iter%1==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * output.size(1)* data.m)))
        iter += 1
    return total_loss / n_samples


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/sm_data.txt',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/Bayesian/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=37,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=10,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=36,help='output sequence length')
parser.add_argument('--horizon', type=int, default=1) 
parser.add_argument('--layers',type=int,default=5,help='number of layers')

parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')

parser.add_argument('--clip',type=int,default=10,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')


args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(3)
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fixed_seed = 123

def main(experiment):
    # Set fixed random seed for reproducibility
    set_random_seed(fixed_seed)

    #model hyper-parameters
    gcn_depths=[1,2,3]
    lrs=[0.01,0.001,0.0005,0.0008,0.0001,0.0003,0.005]#[0.00001,0.0001,0.0002,0.0003]
    convs=[4,8,16,32]
    ress=[8,16,32,64]
    skips=[16,32,64,128,256]
    ends=[128,256,512,1024]
    layers=[1,2,3,4,5]
    ks=[2,3,5,10,15,20,25,30]
    dropouts=[0.2,0.3,0.4,0.5,0.6,0.7]
    dilation_exs=[1,2,3]
    node_dims=[10,20,30,40,50,60,70,80,90,100]
    prop_alphas=[0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8]
    tanh_alphas=[0.05,0.1,0.5,1,2,3,5,7,9]


    best_val = 10000000
    best_rse=  10000000
    best_rae=  10000000
    best_corr= -10000000
    best_smape=10000000
    
    best_test_rse=10000000
    best_test_corr=-10000000

    best_hp=[]


    #random search
    for q in range(60):
        
        #hps
        gcn_depth=gcn_depths[randrange(len(gcn_depths))]
        lr=lrs[randrange(len(lrs))]
        conv=convs[randrange(len(convs))]
        res=ress[randrange(len(ress))]
        skip=skips[randrange(len(skips))]
        end=ends[randrange(len(ends))]
        layer=layers[randrange(len(layers))]
        k=ks[randrange(len(ks))]
        dropout=dropouts[randrange(len(dropouts))]
        dilation_ex=dilation_exs[randrange(len(dilation_exs))]
        node_dim=node_dims[randrange(len(node_dims))]
        prop_alpha=prop_alphas[randrange(len(prop_alphas))]
        tanh_alpha=tanh_alphas[randrange(len(tanh_alphas))]
        

        Data = DataLoaderS(args.data, 0.7, 1, device, args.horizon, args.seq_in_len, args.normalize,args.seq_out_len)
    

        print('train X:',Data.train[0].shape)
        print('train Y:', Data.train[1].shape)
        print('valid X:',Data.valid[0].shape)
        print('valid Y:',Data.valid[1].shape)
        print('test X:',Data.test[0].shape)
        print('test Y:',Data.test[1].shape)
        
        print("------------------------------------------------")
        print("Training samples  :", Data.train[0].shape[0])
        print("Validation samples:", Data.valid[0].shape[0])
        print("Testing samples   :", Data.test[0].shape[0])
        print("------------------------------------------------")

        print('length of training set=',Data.train[0].shape[0])
        print('length of validation set=',Data.valid[0].shape[0])
        print('length of testing set=',Data.test[0].shape[0])
        print('valid=',int(0.2 * Data.n))
        
       
        
        model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
                    device, None, dropout=dropout, subgraph_size=k,
                    node_dim=node_dim, dilation_exponential=dilation_ex,
                    conv_channels=conv, residual_channels=res,
                    skip_channels=skip, end_channels= end,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)
        
        model = model.to(device)

        print(args)
        print('The recpetive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        if args.L1Loss:
            criterion = nn.L1Loss(reduction='sum').to(device)
        else:
            criterion = nn.MSELoss(reduction='sum').to(device)
        evaluateL2 = nn.MSELoss(reduction='sum').to(device) #MSE
        evaluateL1 = nn.L1Loss(reduction='sum').to(device) #MAE

        optim = Optim(
            model.parameters(), args.optim, lr, args.clip, lr_decay=args.weight_decay
        )
        
        es_counter=0 #early stopping
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            print('begin training')
            for epoch in range(1, args.epochs + 1):
                print('Experiment:',(experiment+1))
                print('Iter:',q)
                print('epoch:',epoch)
                print('hp=',[gcn_depth,lr,conv,res,skip,end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch])
                print('best sum=',best_val)
                print('best rrse=',best_rse)
                print('best rrae=',best_rae)
                print('best corr=',best_corr)
                print('best smape=',best_smape)       
                print('best hps=',best_hp)
                print('best test rse=',best_test_rse)
                print('best test corr=',best_test_corr)

                
                es_counter+=1 # feel free to use this for early stopping (not used)

                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                val_loss, val_rae, val_corr, val_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                 args.batch_size,False)
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid smape  {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, val_smape), flush=True)
                # Save the model if the validation loss is the best we've seen so far.
                sum_loss=val_loss+val_rae-val_corr
                if (not math.isnan(val_corr)) and val_loss < best_rse:
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                    best_val = sum_loss
                    best_rse= val_loss
                    best_rae= val_rae
                    best_corr= val_corr
                    best_smape=val_smape

                    best_hp=[gcn_depth,lr,conv,res,skip,end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch]
                    
                    es_counter=0
                    
                    # vtest_acc, vtest_rae, vtest_corr, vtest_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                    #                      args.batch_size, True)
                    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                           args.seq_in_len, False) 
                    print('********************************************************************************************************')
                    print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}| test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape), flush=True)
                    print('********************************************************************************************************')
                    best_test_rse=test_acc
                    best_test_corr=test_corr

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    print('best val loss=',best_val)
    print('best hps=',best_hp)
    #save best hp to desk
    with open('model/Bayesian/hp.txt',"w") as f:
        f.write(str(best_hp))
        f.close()
    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f, weights_only=False)

    vtest_acc, vtest_rae, vtest_corr, vtest_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size, True)

    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                         args.seq_in_len, True) 
    print('********************************************************************************************************')    
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape))
    print('********************************************************************************************************')
    return vtest_acc, vtest_rae, vtest_corr, vtest_smape, test_acc, test_rae, test_corr, test_smape

if __name__ == "__main__":
    vacc = []
    vrae = []
    vcorr = []
    vsmape=[]
    acc = []
    rae = []
    corr = []
    smape=[]
    for i in range(1):
        val_acc, val_rae, val_corr, val_smape, test_acc, test_rae, test_corr, test_smape = main(i)
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        vsmape.append(val_smape)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
        smape.append(test_smape)
    print('\n\n')
    print('1 run average')
    print('\n\n')
    print("valid\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae)))
    print('\n\n')
    print("test\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae)))

