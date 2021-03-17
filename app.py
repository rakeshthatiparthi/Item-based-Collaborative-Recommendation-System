from flask import Flask,render_template,request
import pandas as pd
from fuzzywuzzy import fuzz
import pickle

app = Flask(__name__)
model_nn=pickle.load(open('knnpickle_file.pkl','rb'))
df=pd.read_csv(r"item-user-data.csv")
item_user_matrix = pd.pivot_table(data=df,index="title",columns="reviewerID",values="ratings").fillna(0)
'''
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
item_user_matrix_sparse = csr_matrix(item_user_matrix.values) #transform matrix to scipy sparse matrix
model_nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20) #define model
model_nn.fit(item_user_matrix_sparse) #fit
'''
 
def get_product_index(query_product,product_matrix=item_user_matrix):
    product_index=None
    ratio_tuples = []
    for i in product_matrix.index:
        ratio = fuzz.ratio(i.lower(),query_product.lower())
        if ratio>75:
            current_product_index = product_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_product_index))
    try:
        product_index = max(ratio_tuples, key = lambda x: x[1])[2]     
    except:
        product_index=0
    return product_index

def get_recommendations(query_product_index,product_matrix=item_user_matrix,model=model_nn):
    recommendations=[]
    distances, indices = model.kneighbors(product_matrix.iloc[query_product_index,:].values.reshape(1,-1),n_neighbors = 11)
    for i in range(0,len(distances.flatten())):
        if i == 0:
            pass
        else:
            recommendations.append(product_matrix.index[indices.flatten()[i]])  
    return recommendations

      
@app.route('/',methods=['GET','POST'])
def index():
    
    if request.method == 'GET':
        return(render_template('index.html'))
    
    if request.method == "POST":
        product=request.form['product_name']
        
        product_index=get_product_index(product) #Get product index
        if product_index==0:
            return render_template('index.html',
                                   error='Enter the product name correctly or product is not in our dataset')
        else:
            final_results = get_recommendations(product_index)
    return render_template('index.html',
                           product="Recommendation for '{}':".format(product),
                           result=final_results)
   
if __name__ == '__main__':
    app.run(debug=True)