Built in Python using Plotly + Dash. Pick date range, train using first 80% data (window size of five), test on next 20%, outputs the next day's prediction.
Issue #1: If the request takes longer than 30 seconds Heroku will scrap it, I already reduced epochs to 10 but it can error out for many reasons. I wouldn't recommend exceeding 10 years data.
Issue #2: The prediction basically just shifts yesterday's data forwards by period 1.
Issue #3: White background exposed when graph loads and not sure what causes it.

https://stock-dashboard-ml.herokuapp.com/
