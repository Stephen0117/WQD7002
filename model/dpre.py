import pandas as pd
import numpy as np
import cv2

class prepros:
    
    def data_cleaning(df,fp,sp,cv2):
    
        # Annotate & Join Table
        df['label'] = 1
        fp['label'] = 0
        ff = pd.concat([df,fp])
        
        # Filter Species Focus
        test = ff[ff['species_id']==sp].copy()
        
        # Remove duplicate records
        test = test.drop_duplicates(subset=['recording_id'],keep='first')
        
        # Test_Train Split & Read Image (80% Training, 20% Testing)
        test['recording_id'] = test['recording_id'].apply(lambda x: x+'.flac.png')
        test = test.reset_index()
        train_i=test.sample(frac=0.8, random_state = 1234).copy()
        test_i=test.loc[~test.index.isin(train_i.index)].copy()
        
        # Index Image to be Load
        trimg = train_i['recording_id'].values
        teimg = test_i['recording_id'].values
        
        # Load Images X and Y label
        X_Train = []
        for i in trimg:
            i = f'mel_spec_diagram_fr_rw/{i}'
            X_Train.append(cv2.imread(i))
            
        X_Test = []
        for i in teimg:
            i = f'mel_spec_diagram_fr_rw/{i}'
            X_Test.append(cv2.imread(i))
            
        # Normalize Pixel of Image
        X_Train = np.array(X_Train)/255
        X_Test = np.array(X_Test)/255
        
        # Y Label
        Y_Train = train_i['label'].values
        Y_Tr_arr=[]
        for i in range(len(Y_Train)):
            Y_Tr_arr.append(np.asarray([Y_Train[i]]))
        for i in range(len(Y_Tr_arr)):
            Y_Tr_arr[i] = np.asarray(Y_Tr_arr[i])
        Y_Tr_arr = np.asarray(Y_Tr_arr)
        
        Y_Test = test_i['label'].values
        Y_Te_arr=[]
        for i in range(len(Y_Test)):
            Y_Te_arr.append(np.asarray([Y_Test[i]]))
        for i in range(len(Y_Te_arr)):
            Y_Te_arr[i] = np.asarray(Y_Te_arr[i])
        Y_Te_arr = np.asarray(Y_Te_arr)
        
        # Solve Imbalance Data Issue
        # Scaling by total/2 helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.
        pos = test[test['label']==1].shape[0]
        neg = test[test['label']==0].shape[0]
        total = test.shape[0]
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
    
        return X_Train, X_Test, Y_Tr_arr, Y_Te_arr, class_weight

    def cl_c(X_Train):
        img = []
        for i in range(len(X_Train)):
            img.append(np.asarray([X_Train[i]]))
            
        img = np.asarray(img)
        return img
