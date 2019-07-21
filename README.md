# My notes on Data Science


## Links

[Apache Spark](https://spark.apache.org/)

[Apache Hadoop](https://hadoop.apache.org/)

[Kaggle](https://www.kaggle.com/)

[Matplotlib](https://matplotlib.org/)

[ggplot](http://ggplot.yhathq.com/)

[D3](https://d3js.org/)

[Seaborn](https://seaborn.pydata.org/)

[Flink](https://flink.apache.org/)

[Pandas](https://pandas.pydata.org/)

[Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling)

[SciPy](https://www.scipy.org/)

[Jupyter](https://jupyter.org/)

[Scikit-learn](https://scikit-learn.org/)

[Numpy](https://numpy.org/)

[r-project](https://www.r-project.org/)

## Snippets

##### Jupyter - Pyspark


```shell script

docker run --rm \
-v ~:/home/jovyan/work \
-p 8888:8888 \
-p 4040:4040 \
-p 4041:4041 \
jupyter/pyspark-notebook
```

```python

import pyspark 

sc = pyspark.SparkContext('local[*]')

# do something to prove it works
rdd = sc.parallelize(range(1000))
rdd.takeSample(False, 5)
```