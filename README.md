# TGIN

#### Tensorflow implementation of our method: "Triangle Graph Interest Network for Click-through Rate Prediction".

## Files in the folder

- `dataset/`
  - `electronics/`
    - `uid_voc.pkl`: users;
    - `mid_voc.pkl`: items;
    - `cat_voc.pkl`: categories;
    - `item-info`: mapping dict {item:category};
    - `reviews-info`: interaction records [user, item, rating, timestamp];
    - `local_train_splitByUser`: train data;
    - `local_test_splitByUser`: test data;
    - `wnd3_alpha_01_theta_09_tri_num_10`: triangles data with <nobr aria-hidden="true">α</nobr>=0.1 and <nobr aria-hidden="true">θ</nobr>=0.9;
- `triangle_data/`: processed triangles data of the public datasets.
- `script/`: implementations of TGIN.
- `triangle_mapreduce.zip`: MapReduce implementations of triangle extraction and selection. 

## Prepare data

#### 1. interaction data
We have processed the raw data and upload it to the `electronics/` fold. You can use it directly.

Also, you can get the data from the amazon website and process it using the script:

```
sh prepare_data.sh
```

#### 2. co-occurrence graph
You can use the processed triangles data directly, and just skip this step.

```
python script/gen_wnd_edges.py
```

#### 3. triangle extraction and selection
We have extracted and selected the triangles of both amazon(books) and amazon(electronics) datasets. You can <a href="https://drive.google.com/drive/folders/1gj7aHFjRLVPwmhvK-o1waJKj3Xme7GVM" target="_blank">download</a> and put it into the `triangle_data/` folder.

Next, the triangle indexes should be transformed into the input format of the TGIN model.
```
python process_tridata.py
```

Also, you can refer to the MapReduce source code in 
`triangle_mapreduce.zip` folder to generate triangle indexes.



## Train Model

##### (Recommended) You can skip all the previous steps and run the TGIN model using the script directly.

```

tar xvf triangle_data/electronics_triangle.tar.gz
tar xvf dataset/electronics.tar.gz 
python script/process_tridata.py

sh run.sh
```


## Required packages
The code has been tested running under Python 2.7.18, with the following packages installed (along with their dependencies):

- cPickle == 1.17
- numpy == 1.16.6
- keras == 2.0.8
- tensorflow-gpu == 1.5.0
### 
