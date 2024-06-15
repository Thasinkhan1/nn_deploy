import numpy as np
import os
from src.config import config

class preprosses_data:
    
    
        def fit(self,X,y=None):
            
            self.num_rows = X.shape[0]
            
            
            if len(X.shape)==1:
                
                self.num_input_features = 1 
                #no.of columns
            else:
                self.num_input_features = X.shape[1]
                
            if y is not None:
                
               if len(y.shape) == 1:
                   self.num_target_feauters_dim = 1
               else:
                   self.num_target_feauters_dim = y.shape[1]
                
            else:
                self.num_target_feauters_dim = None
            
        def transform(self,X=None,y=None):
            
            self.X = np.array(X).reshape(self.num_rows,self.num_input_features)
            
            if y is not None:
                self.Y = np.array(y).reshape(self.num_rows,self.num_target_feauters_dim) # converting all the data into numpy array
            else:
                self.Y = None
            
            return self.X,self.Y 
            
        
        
            