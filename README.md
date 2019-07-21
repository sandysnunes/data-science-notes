# My notes on Data Science


## Frameworks, Tools and libraries

[Apache Hadoop](https://hadoop.apache.org/)

[Apache Spark](https://spark.apache.org/)

[D3](https://d3js.org/)

[Flink](https://flink.apache.org/)

[ggplot](http://ggplot.yhathq.com/)

[Jupyter Docker Images](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)

[Jupyter](https://jupyter.org/)

[Kaggle](https://www.kaggle.com/)

[Matplotlib](https://matplotlib.org/)

[Numpy](https://numpy.org/)

[Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling)

[Pandas](https://pandas.pydata.org/)

[r-project](https://www.r-project.org/)

[Scikit-learn](https://scikit-learn.org/)

[SciPy](https://www.scipy.org/)

[Seaborn](https://seaborn.pydata.org/)


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

## Books

## Articles

## Papers

## Datasets

## Ref Cards

 - [**Tensorflow**](https://github.com/sandysnunes/data-science-notes/blob/master/PDFs/Tensorflow.pdf)<br>
  - [**Keras**](https://github.com/sandysnunes/data-science-notes/blob/master/Keras.jpg)<br>
  - [**Neural Networks Zoo**](https://github.com/sandysnunes/data-science-notes/blob/master/Neural%20Networks%20Zoo.png)<br>
  - [**Numpy**](https://github.com/sandysnunes/data-science-notes/blob/master/Numpy.png)<br>
  - [**Scipy**](https://github.com/sandysnunes/data-science-notes/blob/master/Scipy.png)<br>
  - [**Pandas-1**](https://github.com/sandysnunes/data-science-notes/blob/master/Pandas-1.jpg)<br>
  - [**Pandas-2**](https://github.com/sandysnunes/data-science-notes/blob/master/Pandas-2.jpg)<br>
  - [**Pandas-3**](https://github.com/sandysnunes/data-science-notes/blob/master/Pandas-3.png)<br>
  - [**Scikit-learn**](https://github.com/sandysnunes/data-science-notes/blob/master/Scikit%20Learn.png)<br>
  - [**Matplotlib**](https://github.com/sandysnunes/data-science-notes/blob/master/Matplotlib.png)<br>
  - [**ggplot2-1**](https://github.com/sandysnunes/data-science-notes/blob/master/ggplot2-1.jpg)<br>
  - [**ggplot2-2**](https://github.com/sandysnunes/data-science-notes/blob/master/ggplot2-2.jpg)<br>
  - [**PySpark**](https://github.com/sandysnunes/data-science-notes/blob/master/PySpark.jpg)<br>
  - [**PySpark-RDD**](https://github.com/sandysnunes/data-science-notes/blob/master/PySpark-RDD.png)<br>
  - [**PySpark-SQL**](https://github.com/sandysnunes/data-science-notes/blob/master/PySpark-SQL.png)<br>
  - [**R Studio(dplyr & tidyr)-1**](https://github.com/sandysnunes/data-science-notes/blob/master/Data%20Wrangling%20with%20dplyr%20and%20tidyr%20-%20R%20Studio-1.jpg)<br>
  - [**R Studio(dplyr & tidyr)-2**](https://github.com/sandysnunes/data-science-notes/blob/master/Data%20Wrangling%20with%20dplyr%20and%20tidyr%20-%20R%20Studio-2.jpg)<br>
  - [**Neural Network Cells**](https://github.com/sandysnunes/data-science-notes/blob/master/Neural%20Network%20Cells.png)<br>
  - [**Neural Network Graphs**](https://github.com/sandysnunes/data-science-notes/blob/master/Neural%20Network%20Graphs.png)<br>
  - [**Deep Learning Cheat Sheet**](https://github.com/sandysnunes/data-science-notes/blob/master/Deep%20Learning%20Cheat%20Sheet-Hacker%20Noon.pdf)<br>
  - [**Dask1**](https://github.com/sandysnunes/data-science-notes/blob/master/Dask1.png)
  - [**Dask2**](https://github.com/sandysnunes/data-science-notes/blob/master/Dask2.png)
  - [**Dask3**](https://github.com/sandysnunes/data-science-notes/blob/master/Dask3.png)
  - [**Dask4**](https://github.com/sandysnunes/data-science-notes/blob/master/Dask4.png)
  - [**All Cheat Sheets(PDF)**](https://github.com/sandysnunes/data-science-notes/blob/master/All%20Cheat%20Sheets.pdf)<br>

## Notebooks

## Podcasts
- ### Podcasts for Beginners:
    - [Talking Machines](http://www.thetalkingmachines.com/)
    - [Linear Digressions](http://lineardigressions.com/)
    - [Data Skeptic](http://dataskeptic.com/)
    - [This Week in Machine Learning & AI](https://twimlai.com/)
    - [Machine Learning Guide](http://ocdevel.com/podcasts/machine-learning)

- ### "More" advanced podcasts
    - [Partially Derivative](http://partiallyderivative.com/)
    - [O’Reilly Data Show](http://radar.oreilly.com/tag/oreilly-data-show-podcast)
    - [Not So Standard Deviation](https://soundcloud.com/nssd-podcast)

- ### Podcasts to think outside the box:
    - [Data Stories](http://datastori.es/)

## Communities
- Quora
    - [Machine Learning](https://www.quora.com/topic/Machine-Learning)
    - [Statistics](https://www.quora.com/topic/Statistics-academic-discipline)
    - [Data Mining](https://www.quora.com/topic/Data-Mining)

- Reddit
    - [Machine Learning](https://www.reddit.com/r/machinelearning)
    - [Computer Vision](https://www.reddit.com/r/computervision)
    - [Natural Language](https://www.reddit.com/r/languagetechnology)
    - [Data Science](https://www.reddit.com/r/datascience)
    - [Big Data](https://www.reddit.com/r/bigdata)
    - [Statistics](https://www.reddit.com/r/statistics)

- [Data Tau](http://www.datatau.com/)

- [Deep Learning News](http://news.startup.ml/)

- [KDnuggets](http://www.kdnuggets.com/)

## Conferences
- Neural Information Processing Systems ([NIPS](https://nips.cc/))
- International Conference on Learning Representations ([ICLR](http://www.iclr.cc/doku.php?id=ICLR2017:main&redirect=1))
- Association for the Advancement of Artificial Intelligence ([AAAI](http://www.aaai.org/Conferences/AAAI/aaai17.php))
- IEEE Conference on Computational Intelligence and Games ([CIG](http://www.ieee-cig.org/))
- IEEE International Conference on Machine Learning and Applications ([ICMLA](http://www.icmla-conference.org/))
- International Conference on Machine Learning ([ICML](https://2017.icml.cc/))
- International Joint Conferences on Artificial Intelligence ([IJCAI](http://www.ijcai.org/))
- Association for Computational Linguistics ([ACL](http://acl2017.org/))

## Interview Questions
- [How To Prepare For A Machine Learning Interview](http://blog.udacity.com/2016/05/prepare-machine-learning-interview.html)
- [40 Interview Questions asked at Startups in Machine Learning / Data Science](https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science)
- [21 Must-Know Data Science Interview Questions and Answers](http://www.kdnuggets.com/2016/02/21-data-science-interview-questions-answers.html)
- [Top 50 Machine learning Interview questions & Answers](http://career.guru99.com/top-50-interview-questions-on-machine-learning/)
- [Machine Learning Engineer interview questions](https://resources.workable.com/machine-learning-engineer-interview-questions)
- [Popular Machine Learning Interview Questions](http://www.learn4master.com/machine-learning/popular-machine-learning-interview-questions)
- [What are some common Machine Learning interview questions?](https://www.quora.com/What-are-some-common-Machine-Learning-interview-questions)
- [What are the best interview questions to evaluate a machine learning researcher?](https://www.quora.com/What-are-the-best-interview-questions-to-evaluate-a-machine-learning-researcher)
- [Collection of Machine Learning Interview Questions](http://analyticscosm.com/machine-learning-interview-questions-for-data-scientist-interview/)
- [121 Essential Machine Learning Questions & Answers](https://elitedatascience.com/mlqa-reading-list)