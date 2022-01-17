import holidays
import pandas as pd


kr_holidays = holidays.KR(years=[2020,2021])#주식장이 열리지 않는 공휴일,임시공휴일에 해당하는 행을 삭제하여 준다.
import holidays
'''
{datetime.date(2020, 1, 1): "New Year's Day", datetime.date(2020, 1, 24): "The day preceding of Lunar New Year's Day", datetime.date(2020, 1, 25): "Lunar New Year's Day", datetime.date(2020, 1, 26): 
"The second day of Lunar New Year's Day", datetime.date(2020, 1, 27): "Alternative holiday of Lunar New Year's Day", datetime.date(2020, 3, 1): 'Independence Movement Day', 
datetime.date(2020, 4, 30): 'Birthday of the Buddha', datetime.date(2020, 5, 5): "Children's Day", datetime.date(2020, 5, 1): 'Labour Day', datetime.date(2020, 6, 6): 'Memorial Day', 
datetime.date(2020, 8, 15): 'Liberation Day', datetime.date(2020, 9, 30): 'The day preceding of Chuseok', datetime.date(2020, 10, 1): 'Chuseok', datetime.date(2020, 10, 2): 'The second day of Chuseok', 
datetime.date(2020, 10, 3): 'National Foundation Day', datetime.date(2020, 10, 9): 'Hangeul Day', datetime.date(2020, 12, 25): 'Christmas Day', datetime.date(2020, 8, 17): 'Alternative public holiday'}
{datetime.date(2021, 1, 1): "New Year's Day", datetime.date(2021, 2, 11): "The day preceding of Lunar New Year's Day", datetime.date(2021, 2, 12): "Lunar New Year's Day", datetime.date(2021, 2, 13): 
"The second day of Lunar New Year's Day", datetime.date(2021, 3, 1): 'Independence Movement Day', datetime.date(2021, 5, 19): 'Birthday of the Buddha', datetime.date(2021, 5, 5): "Children's Day", 
datetime.date(2021, 5, 1): 'Labour Day', datetime.date(2021, 6, 6): 'Memorial Day', datetime.date(2021, 8, 15): 'Liberation Day', datetime.date(2021, 8, 16): 'Alternative holiday of Liberation Day', 
datetime.date(2021, 9, 20): 'The day preceding of Chuseok', datetime.date(2021, 9, 21): 'Chuseok', datetime.date(2021, 9, 22): 'The second day of Chuseok', datetime.date(2021, 10, 3): 'National Foundation Day', 
datetime.date(2021, 10, 4): 'Alternative holiday of National Foundation Day', datetime.date(2021, 10, 9): 'Hangeul Day', datetime.date(2021, 10, 11): 'Alternative holiday of Hangeul Day', datetime.date(2021, 12, 25): 'Christmas Day'}
'''
date_list=[]

for date, occasion in kr_holidays.items():
    date = (f'{date}')
    date_list.append(date)
    date_list.sort()
  
print(date_list)
'''
['2020-01-01', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-03-01', '2020-04-30', '2020-05-01', '2020-05-05', 
'2020-06-06', '2020-08-15', '2020-08-17', '2020-09-30', '2020-10-01', '2020-10-02', '2020-10-03', '2020-10-09', '2020-12-25', 
'2021-01-01', '2021-02-11', '2021-02-12', '2021-02-13', '2021-03-01', '2021-05-01', '2021-05-05', '2021-05-19', '2021-06-06', 
'2021-08-15', '2021-08-16', '2021-09-20', '2021-09-21', '2021-09-22', '2021-10-03', '2021-10-04', '2021-10-09', '2021-10-11', '2021-12-25']
'''
date_list.remove('2020-01-01')
date_list.remove('2020-01-25')
date_list.remove('2020-01-26')
date_list.remove('2020-03-01')
date_list.remove('2020-06-06')
date_list.remove('2020-08-15')
date_list.remove('2020-10-03')
date_list.remove('2020-10-09')
date_list.remove('2020-12-25')
date_list.remove('2021-02-13')
date_list.remove('2021-05-01')
date_list.remove('2021-06-06')
date_list.remove('2021-08-15')
date_list.remove('2021-10-03')
date_list.remove('2021-10-09')
date_list.remove('2021-12-25')

print(date_list)
'''
['2020-01-24', '2020-01-27', '2020-04-30', '2020-05-01', '2020-05-05', '2020-08-17', '2020-09-30', '2020-10-01', '2020-10-02',
'2021-01-01', '2021-02-11', '2021-02-12', '2021-03-01', '2021-05-05', '2021-05-19', '2021-08-16', '2021-09-20', '2021-09-21',
'2021-09-22', '2021-10-04', '2021-10-11']
'''
x1.drop(date_list[0:], axis=0, inplace=True)

x1.drop("2020-04-15", axis=0, inplace=True)  #제21대 국회의원선거
x1.drop("2020-12-31", axis=0, inplace=True)  #한국 폐장일
x1.drop("2021-12-31", axis=0, inplace=True)  #한국 폐장일


