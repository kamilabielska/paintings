import os
import pandas as pd
import pickle

classes12 = ['Salvador Dali', 'Vincent van Gogh', 'Raphael', 'Leonardo da Vinci',
             'Rembrandt', 'Pablo Picasso', 'Francisco Goya', 'Peter Paul Rubens',
             'Claude Monet', 'Edvard Munch', 'Andy Warhol', 'Jackson Pollock']

df = pd.read_csv('artists.csv')
most_common = df[['name','paintings']].sort_values('paintings', ascending=False)[:15]['name']
classes15 = list(most_common.reset_index(drop=True).str.replace('Dürer', 'Durer')) 
classes50 = list(df['name'].str.replace('Dürer', 'Durer'))

if not os.path.exists('test info'):
    os.makedirs('test info')
    
for classes in [classes12, classes15, classes50]:
    test_images, test_labels = [], []
    n = str(len(classes))
    
    for folder in classes:
        folder_path = os.path.join('test', folder)
        for filename in os.listdir(folder_path):
            test_images.append(os.path.join(folder_path, filename))
            test_labels.append(folder)
            
    with open(os.path.join('test info', 'test_images_'+n), 'wb') as file:
        pickle.dump(test_images, file)
    with open(os.path.join('test info', 'test_labels_'+n), 'wb') as file:
        pickle.dump(test_labels, file)