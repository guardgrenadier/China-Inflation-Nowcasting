# China-Inflation-Nowcasting
A model for nowcasting China's inflation based on Dynamic Factor Model.  
This project is based on <https://github.com/HoagieT/Inflation-Nowcast-Model> and improvements were made to produce a reproduction of a research report from HUATAI Securities **《美国CPI的三周期Nowcasting模型》**.  
##main.py
Running this file will perform a backtest of forecasting China's inflation at the the ot the month during a selected time period. What the 'forecasting' means here is that since China officially anounces a certain month's CPI/PPI on roughly 8th day in the following month, this model forecast china's inflation at the end of the month, or in other words, one week inadvance. Thus this model can help with making investment decisions.  
This project provides 2 methods of forecasting, you can change between them by changing the parameter 'alt_method'. The 2 methods differ in how the model converges. The method under 'alt_method=False', or the default method, is illustrated in the PDF. While the method under 'alt_method=False' is described in the research report from HUATAI, with a slight difference in terms of the number of latent factors. 
