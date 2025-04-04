# China-Inflation-Nowcasting
A model for nowcasting China's inflation based on Dynamic Factor Model.  

This project is based on <https://github.com/HoagieT/Inflation-Nowcast-Model> and improvements were made to produce a reproduction of a research report from HUATAI Securities **《美国CPI的三周期Nowcasting模型》** with some minor differences.

You can run 'main.py' first which performs backtest of monthly forecasting of China's CPI/PPI with 2 methods of estimating the DFM. The 4 backtest will cost proximately 10 minutes and will produce 1 graph comparing the backtest results of CPI/PPI first order differences with Wind CPI/PPI expectation first order differences, 1 graph comparing backtest results of CPI/PPI with Wind CPI/PPI expectation, 1 excel storing the backtest results. So altogether running this file will produce 4 excels and 8 graphs.
After running 'main.py', you can run 'backtest_error_analysis.py' to evaluate the backtest results.
Finally running 'CPI_backtest_daily' to use the 'Nowcasting' function. Note that this file is independent of the 2 files above.

All necessary data are uploaded here in the './data' directory, which are up to 2024/12/31.

If you want to find more details of this project, check the PDF file.

## main.py
Running this file will perform a backtest of forecasting China's inflation at the the ot the month during a selected time period. What the 'forecasting' means here is that since China officially anounces a certain month's CPI/PPI on roughly 8th day in the following month, this model forecast china's inflation at the end of the month, or in other words, one week inadvance. Thus this model can help with making investment decisions.  

This project provides 2 methods of estimating the parameters of DFM, you can change between them by changing the parameter 'alt_method'. The 2 methods differ in how the model converges. The method under 'alt_method=False', or the default method, is illustrated in the PDF. While the method under 'alt_method=True' is described in the research report from HUATAI, with a slight difference in terms of the number of latent factors and the use of Kalman Filter here. 

## backtest_error_analysis.py
Running this file will perform a error analysis of the results of forecasting backtest produced in 'main.py', so you have to run 'main.py' before running this file. This file will produced 3 graphs and 4 lines in your terminal -- 1 error histogram of the backtest result, 1 error histogram of Wind CPI/PPI expectation, 1 line chart comparing Wind CPI/PPI expectation with Actual CPI/PPI, 2 lines of the R2 and direction correctness of forecast for backtest result and Wind expectation each.

## CPI_backtest daily.py
Running this file will performing the 'Nowcasting' of CPI, that is, updating the forecasting value of CPI of this month on a daily basis based on data updated each day. This is also a 'backtest' file since it is a imitation of updating data every day. You can select a certain month to backtest by changing a parameter.
