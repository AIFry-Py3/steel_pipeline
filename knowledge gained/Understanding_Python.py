# Understanding pd.Series Class!
# print(pd.Series(data='2023-10-23'))
# print(pd.Series(data=3))
# print(pd.Series(data=["hi", "world"]))
# print(pd.Series(data=[3,4,7]))
# print(pd.Series(data=[pd.to_datetime('2023-10-23')]))


# Understanding the df_test['ds'] column in our data!
# try:
#     for date in df_test['ds']:
#         print(type(date))
#         if date=='2023-11-30':
#             print("Exists")
#     for date in future['ds']:
#         print(type(date))
#     for date in df_test['ds']:
#         print(type(pd.to_datetime(date)))
#     
#     for ds in forecast['ds']:
#         print(type(ds))     # <class 'pandas._libs.tslibs.timestamps.Timestamp'>
#     
#         test_dict = dict(zip(df_test['ds'], df_test['y']))
#    test_dict = dict(zip(df_test['ds'], df_test['y']))

#     for ds,yhat in forecast['ds']:
#         if ds in pd.to_dataframe(df_test['ds']):
#               did not continue

#    forecast_dict = dict(zip(forecast['ds'], forecast['yhat']))
#    print(test_dict)
#    print(forecast_dict)

# except Exception as inst:
#     print(type(inst))
#     print(inst.args)
#     print(inst)

    # Mean Squared Error Logic!
    # diff_list = []
    # for ds in test_dict.keys():
    #     diff_list.append(abs(test_dict[ds] - forecast_dict[ds]))
    # loss = sum(diff_list)/len(test_dict.keys()) 