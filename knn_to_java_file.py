import sklearn_porter
from model_network import *
import warnings

warnings.filterwarnings('ignore')

loaded_file = open(f'model_number_multi_0_3d_knn.pickle', 'rb')
model = pickle.load(loaded_file)

porter = sklearn_porter.Porter(model.model, language='java')
java_code=porter.export()
print(java_code)
f=open('KNNClassifier.java', 'w')
f.write(java_code)
f.close()




