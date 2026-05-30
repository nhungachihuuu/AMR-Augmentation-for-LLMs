import torch
import pickle
graph = torch.load("dataset/cnn/dglgraph-cnn-train-v0.pt", map_location='cpu', weights_only=False)
with open('dataset/cnn/cnn_train_v2.pkl', 'rb') as file:
    # print('Load data')
    text = pickle.load(file)

print(len(graph['id']))
print(len(text['doc_keys']))
print("check id : b2661807e28a8175cbfca9fa4f1b85ecd222f8cf")
doc_id = 'b2661807e28a8175cbfca9fa4f1b85ecd222f8cf'


print('pkl')
print(text['doc_keys'].index(doc_id))

print('graph')
print(graph['id'].index(doc_id))