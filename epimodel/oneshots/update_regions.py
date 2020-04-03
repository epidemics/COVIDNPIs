# this script serves to update the regions.csv file with population, lat, long, continent columns
# it requires:
#   file 'countries_codes_and_coordinates.csv', available at https://gist.github.com/tadast/8827699
#   file 'regions.yaml'
# Done:
#   adding population, lat, long, continent name to countries
# To-Do:
#   finish up gleam subregions updating
#   refactor the regions part


import pandas as pd
import yaml
import csv

# utility function creating dict to look up latitude a nd longitude
# based on ISOa33 later


def get_latlong_dict():
    alpha3_to_lat_long = {}
    with open('countries_codes_and_coordinates.csv', 'r', newline='\n') as file:
        reader = csv.reader(file, delimiter=',',
                            skipinitialspace=True)
        reader.__next__()

        for row in reader:
            if row[2] not in alpha3_to_lat_long.keys():
                alpha3_to_lat_long[row[2]] = (float(row[4]), float(row[5]))

    return alpha3_to_lat_long


# dict with latitude and longitude values, keys are ISOa3 values, returns tuple of (lat,long)
latlong_dict = get_latlong_dict()

# df is the old regions.csv to be updated
# cast M49 code column to string, otherwise pandas converts it to float
df = pd.read_csv('../data/regions.csv', index_col='ISOa3',
                 converters={'M49Code': str})

# new columns or column values for countrie in df
new_columns = pd.DataFrame(
    columns=['ISOa3', 'Continent', 'Lat', 'Long', 'Population'])

# new columns or values for gleam subregions
new_gl_cols = pd.DataFrame(
    columns=['ISOa3', 'Continent', 'Lat', 'Long', 'Population', 'Level'])


with open('regions.yaml', 'r') as f:
    data = yaml.load(f)

    world = data["subregions"]
    with open('output_countries.csv', 'w') as out_file:

        for i in range(len(world)):
            continent_name = world[i]["names"][0]
            continents = world[i]["subregions"]

            for j in range(len(continents)):
                # this is continent -> country route
                if continents[j]["kind"] == 'country':
                    # some countries don't have ISOa3 value - they are excluded
                    if continents[j].get("iso_alpha_3"):
                        country = continents[j]

                        # some countries are missing in dict with lat,long values
                        if latlong_dict.get(
                                country["iso_alpha_3"]):
                            latitude = latlong_dict.get(
                                country["iso_alpha_3"])[0]
                            longitude = latlong_dict.get(
                                country["iso_alpha_3"])[1]

                        ISOa3 = country["iso_alpha_3"]
                        population = country.get("population")

                        # create a row that gets concatenated to new_columns
                        row = pd.DataFrame([[ISOa3, continent_name, latitude, longitude, population]], columns=[
                                           'ISOa3', 'Continent', 'Lat', 'Lon', 'Population'])
                        new_columns = pd.concat([row, new_columns])

                        # gleam subregions are asubregions of countries
                        # this route is the same for continent -> region (EU or Middle East) -> country,
                        # so check for kind == city
                        gleam_subregions = continents[j]["subregions"]
                        for k in range(len(gleam_subregions)):

                            sub_reg = gleam_subregions[k]

                            if sub_reg["kind"] == "city":
                                gl_key = country["iso_alpha_3"] + \
                                    '_' + sub_reg.get("iana")
                            else:
                                gl_key = country["iso_alpha_3"] + \
                                    '_' + sub_reg.get("key")
                            gl_iana = sub_reg.get("iana")

                            gl_continent = continent_name
                            gl_lat = sub_reg.get("lat")
                            gl_lon = sub_reg.get("lon")
                            gl_pop = sub_reg.get("population")
                            gl_level = sub_reg.get("kind")
                            gl_row = pd.DataFrame([[gl_key, gl_continent, gl_lat, gl_lon, gl_pop, gl_level]], columns=[
                                'ISOa3', 'Continent', 'Lat', 'Long', 'Population', 'Level'])

                            new_gl_cols = pd.concat([gl_row, new_gl_cols])

                # the alternative route if there is a region (EU, Middle East) - needs refactoring badly
                elif continents[j]["kind"] == 'region':
                    countries = continents[j]['subregions']
                    for m in range(len(countries)):
                        country = countries[m]
                        print(country["names"])
                        if country.get("iso_alpha_3"):
                            if latlong_dict.get(
                                    country["iso_alpha_3"]):
                                latitude = latlong_dict.get(
                                    country["iso_alpha_3"])[0]
                                longitude = latlong_dict.get(
                                    country["iso_alpha_3"])[1]

                            ISOa3 = country["iso_alpha_3"]
                            if country.get("population"):
                                population = country.get("population")
                            else:
                                population = None

                            row = pd.DataFrame([[ISOa3, continent_name, latitude, longitude, population]], columns=[
                                'ISOa3', 'Continent', 'Lat', 'Lon', 'Population'])
                            new_columns = pd.concat([row, new_columns])

                            gleam_subregions = continents[j]["subregions"]
                            if continents[j]["kind"] == 'region':
                                gleam_subregions = []

                            for k in range(len(gleam_subregions)):

                                sub_reg = gleam_subregions[k]

                                if sub_reg["kind"] == "city":
                                    gl_key = country["iso_alpha_3"] + \
                                        '_' + sub_reg.get("iana")
                                else:
                                    gl_key = country["iso_alpha_3"] + \
                                        '_' + sub_reg.get("key")
                                gl_iana = sub_reg.get("iana")

                                gl_continent = continent_name
                                gl_lat = sub_reg.get("lat")
                                gl_lon = sub_reg.get("lon")
                                gl_pop = sub_reg.get("population")
                                gl_level = sub_reg.get("kind")
                                gl_row = pd.DataFrame([[gl_key, gl_continent, gl_lat, gl_lon, gl_pop, gl_level]], columns=[
                                    'ISOa3', 'Continent', 'Lat', 'Long', 'Population', 'Level'])

                                new_gl_cols = pd.concat([gl_row, new_gl_cols])

    # ISOa3 is the one column shared by all of the datafiles
    new_columns = new_columns.set_index('ISOa3')
    updated_df = df.update(new_columns)

    # outputs udpated regions csv
    df.to_csv('regions_updated.csv')

    # outputs gleam regions
    new_gl_cols.to_csv('out_gl.csv')
