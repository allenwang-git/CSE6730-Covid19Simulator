import COVID19Py
def get_covid_data(code: str):
    CovidData = COVID19Py.COVID19(data_source="jhu")
    CovidDataForOneCountry = CovidData.getLocationByCountryCode(code.upper(), timelines=True)
    print(CovidDataForOneCountry)
    timespan = 90
    # get date
    date = list(CovidDataForOneCountry[0]["timelines"]["confirmed"]["timeline"].keys())[-timespan:]
    # get confirmed cumulative cases
    confirmed_cumulative_cases = list(CovidDataForOneCountry[0]["timelines"]["confirmed"]["timeline"].values())[-timespan:]
    # get death cumulative cases
    death_cumulative_cases = list(CovidDataForOneCountry[0]["timelines"]["deaths"]["timeline"].values())[-timespan:]

    # generate dict to store covid data for one specific country
    CovidDataForOneCountryDict = {"date": date, "confirmed": confirmed_cumulative_cases, "deaths": death_cumulative_cases}
    print(CovidDataForOneCountryDict)

    import json
    # create json object from dictionary
    json = json.dumps(CovidDataForOneCountryDict)

    # open file for writing, "w" 
    f = open("data.json","w")

    # write json object to file
    f.write(json)

    # close file
    f.close()

get_covid_data("US")