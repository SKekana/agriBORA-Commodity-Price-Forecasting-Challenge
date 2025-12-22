# agriBORA Commodity Price Forecasting Challenge

## Can you accurately forecast weekly market prices for maize in Kenya?
Smallholder farmers across Africa often face price volatility and post-harvest losses of up to 40%. agriBORA is a startup aiming to help smallholder farmers in Kenya by offering certified warehouses where farmers can safely store their produce and receive digital warehouse certificates, enabling access to loans and the option of delayed selling. This provides flexibility for farmers in deciding the optimal time to sell their produce for maximum returns, and reduces storage losses.

Using historical prices of dry maize in Kenya, your task is to develop a machine learning solution to predict average weekly prices of maize in the counties of Kiambu, Kirinyaga, Mombasa, Nairobi and Uasin-Gishu. At each prediction step, your model should generate forecasts for two consecutive weeks. The forecasting period spans six consecutive weeks, from November 17, 2025 to January 10, 2026.

Accurate forecasts will help farmers time their sales effectively, increase earnings, and strengthen agriBORA’s integrated storage, credit, and market intelligence service to East African farmers.


About Digital Africa

Launched in 2018, the Digital Africa initiative's mission is to strengthen the capacity of African digital entrepreneurs to design and deploy disruptive innovations at scale to serve the real economy. The initiative brings together a range of partners of all nationalities committed to African digital entrepreneurs, at the forefront of which is the French Development Agency (AFD).

## About AgriBORA

agriBORA is an agri-fintech company redefining how grain is stored, financed, and traded across East Africa. Through its agriGHALA platform, the company integrates climate-smart warehousing, digital finance, and structured market linkages to create a transparent and efficient grain ecosystem. By empowering farmers, cooperatives, traders, and large-scale producers to securely store their grain, access instant liquidity, and connect directly with verified buyers, agriBORA is reducing post-harvest losses, unlocking working capital, and enabling a more resilient and inclusive agricultural economy.

Evaluation
This challenge uses multi-metric evaluation. There are two error metrics: Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

Your score on the leaderboard is the weighted average of the two metrics:

MAE (50%): measures the average magnitude of errors between predicted and actual values. This metric is less sensitive to outliers compared to other error metrics such as Mean Squared Error (MSE). It is a suitable choice for financial forecasting.
RMSE (50%): measures the deviation of your predictions from the actual values, but penalises large errors more heavily.
For each of the selected county, submission files should contain 3 columns: ID, Target_RMSE and Target_MAE. Take note of the column names and their order. The values in the ID column are formed by concatenating the county name and the week of year for which the predictions are made. For each entry, the predicted price (target) must be the same in both target columns. This is necessary for multi-metric evaluation.

The RMSE score is calculated from the column Target_RMSE and the MAE score is calculated from the column Target_MAE.

Your submission file should look like this (numbers to show format only):

ID                  Target_RMSE      Target_MAE
Kiambu_Week_50         0.57              0.57    
Kirinyaga_Week_50      0.57              0.57  
Mombasa_Week_50        0.57              0.57
Nairobi_Week_50        0.57              0.57    
Uasin-Gishu_Week_50    0.57              0.57    
Kiambu_Week_51         0.57              0.57    
Kirinyaga_Week_51      0.57              0.57    
Mombasa_Week_51        0.57              0.57    
Nairobi_Week_51        0.57              0.57       
Uasin-Gishu_Week_51    0.57              0.57    
A rolling leaderboard is used for this challenge. It will be updated every week on Tuesday at 15:00 GMT, as the actual maize prices in the selected counties become available.

## About Data

The data consists of historical wholesale and retail prices of maize at different markets in Kenya from 2021 - 2025. The first dataset is from KAMIS. It consists of historical prices for three types of maize (white, yellow and mixed-traditional). You can pull extra data from the KAMIS website to complement the sample that is provided. The second dataset, collected by agriBORA, is the transaction data between businesses showing the wholesale price of white maize for a given week.

Using historical prices of dry maize in Kenya, your task is to develop a machine learning solution to predict average weekly prices of maize in the counties of Kiambu, Kirinyaga, Mombasa, Nairobi and Uasin-Gishu. agriBORA data is the data that needs to be forecasted, however, KAMIS data is provided if you would like to supplement the agriBORA data. At each prediction step, your model should generate forecasts for two consecutive weeks. The forecasting period spans six consecutive weeks, from November 17, 2025 to January 10, 2026.

You are encouraged to explore and incorporate any external data you believe may improve your solution - precipitation, vegetation indices (NDVI), or other relevant data could provide useful signals. Part of the challenge is discovering which features truly help predict prices. Any additional dataset used must be open-source, and must be shared in the chat in order for us to consider your winning submission.

If you choose to use additional data sources, please submit the following:

Scripts for downloading or pulling the data
The raw data itself
Any processing scripts you applied
The processed data used as input to your model
This ensures your solution is fully reproducible and allows others to understand and build upon your approach.