# Data Science Portfolio

This [GitHub Repository](https://github.com/hheejuice/Heeju_Portfolio) includes data analysis and machine learning exploration projects completed by Heeju Son who received a bachelor's degree in Economics from the University of Washington. Each project has been written in R using RStudio, then pushed to the GitHub repository.

README.md will highlight key findings of each data analysis projects. Codes and additional analysis are available in repository's subfolder. They also can be accessed by project title hyperlinks.

For better portfolio browsing experience, please check out [Portfolio](https://hheejuice.github.io/Heeju_Portfolio/).

## Contents
* ### Data Analysis and Visualization
  * [Netflix Analysis](Netflix-Analysis/Netflix-Analysis.md): Netflix has been growing very rapidly over the past few years. They have successfully expanded their business to over 190 countries internationally, and they are now producing many Netflix original contents. This project will investigate the trends in their library of movies and TV shows. 
  
      *Used Packages: tidyverse, dplyr, ggplot2, plotly, rworldmap*
      
      *Key findings:* Netflix has been focusing more on TV shows than on movies in recent years. The number of TV shows available on Netflix has been continuously increasing while the number of movies has been declining since 2019. Continent-wise, Netflix had a preference for movies from America - North and South America - over movies from other continents. This has been visualized by a map presented below. This project also shows that Netflix has made a lot of effort to promote diversity in their movie collection. A percentage stacked barchart for 'Movie Genre by Year' below shows that the percentage for international and LGBTQ films has been increasing since 2014. Out of top five countries by the number of movies on Netflix, France and India showed weaker genre diversity compared to US, UK, and Canada. Given the current trends, Netflix is very likely to increase genre variety in foreign movies in the near future.
  
    <img src="Netflix-Analysis/Netflix-Analysis_files/figure-html/year-2.png" width="250"> <img src="Netflix-Analysis/Netflix-Analysis_files/figure-html/map-1.png" width="250"> 
  
    <img src="Netflix-Analysis/Netflix-Analysis_files/figure-html/genrebyyear-2.png" width="250"> <img src="Netflix-Analysis/Netflix-Analysis_files/figure-html/genrebycountry-2.png" width="250">  

  * [HR Analysis](HR-Analysis/HR-Analytics.md): When hiring data scientists, it is important for companies to know which of their candidates are willing to work for them after training. It helps them reduce costs and time associated with training session. In this project, I will perform EDA (Exploratory Data Analysis) on data scientist candidate data, and look for insights that could be helpful for company with their hiring decision.
  
      *Used Packages: tidyverse, dplyr, ggplot2, Hmisc, scales*
      
      *Key findings:* Data scientists in general often look for company change in five-year intervals. Also, those who have less years of experience are more likely to look for another company to work for. This project also highlights that the funding status of a company is one of the important factors for data scientists in their company decision. Many data scientists in startups were willing to continue working given that the company was well funded. 

    <img src="HR-Analysis/HR-Analytics_files/figure-html/exp-1.png" width="250"> <img src="HR-Analysis/HR-Analytics_files/figure-html/yearexp-1.png" width="250">
  
    <img src="HR-Analysis/HR-Analytics_files/figure-html/type-2.png" width="250"> <img src="HR-Analysis/HR-Analytics_files/figure-html/type-3.png" width="250">
  
  * [Credit Card Customer Analysis](Credit-card-customer/Credit-Card-Customers.md): This project studies behavior of credit card customers. The data used in this project includes customer demographics and credit card information. In this project, I will show the process of outlier removal and data binning.
  
      *Used Packages: tidyverse, dplyr, ggplot2, Hmisc, knitr, kableExtra*
      
      *Key findings:* Credit card customers in their twenties are worst at managing credit compared to other age groups, and women tend to overspend more than men. The project also suggests that the credit card users with many dependents usually have hard time paying off their credit card back, although they are better at managing credit card and keeping the spending low. The customers with blue cards, which is the least prestigious card type, tend to spend more on the card compared to customers with different card types. However, study suggests that the blue card customers are doing better job managing their credit by paying off their debt on time.

* ### Machine Learning
  * [House Price Prediction Project using ML Linear Regression, K-Nearest Neighbors and Decision Tree](Housing-Price-Prediction/Housing-Price-Prediction.md): ndflnfln
    
      *Used Packages: caret, tree, class, psych, corrplot, psych*
      
  * [project using ML Logistic Regression](https://hheejuice.github.io/Heeju_Portfolio/): ndflnfln
  * [project using ML Classification](url): ndflnfln
  * Cluster Analysis
 
* ### Natural Language Processing
  * Tokenization
  * Stopword Removal
  * Lemmatization & Stemming
  https://monkeylearn.com/blog/nlp-ai/
